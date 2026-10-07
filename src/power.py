"""Mac power readout for the panel — no root, no subprocess, no new dependency.

Two things a normal user process may read on macOS:

  * IOReport's private "Energy Model" group: accumulated energy per rail, in nJ/uJ/mJ/J.
    A delta over time is watts. Measured on this M2 Pro (macOS 26): only **GPU Energy**
    actually moves — the CPU rail is present but reports exactly 0, even with every core
    pegged, which is precisely the part `powermetrics` unlocks with root. So the panel
    shows the rails it can read, labels them, and never invents a CPU/system number
    (see README「功耗」and the AGENTS.md note: don't fake the CPU figure).
  * `AppleSmartBattery` in the IORegistry: Voltage (mV) × InstantAmperage (mA) = watts.
    While discharging that IS the machine's whole draw; on AC it is the charging rate (or
    nothing at all), and the label says which of the two you are looking at. Laptops only.

Contract, because this runs inside the 0.25 s tick:
  * `PowerMeter.sample()` is throttled (default 1 Hz) and returns None when not due, so the
    caller just renders whatever comes back. A due sample costs ~0.4 ms (IOReport delta)
    plus ~40 us (registry read) — three orders of magnitude below the tick.
  * Nothing here raises or blocks: a missing library, a desktop Mac with no battery, or an
    Intel Mac without the energy group all degrade to a smaller reading, never to a crash.
  * The first due sample only establishes the IOReport baseline (a delta needs two), so on
    AC the label reads the placeholder for that first second instead of showing a fake
    number; a battery wattage needs no baseline and appears on the very first sample.

CLI: `uv run python src/power.py [秒数]` prints a reading a second, for eyeballing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import time

# Energy rails arrive with their own unit label; scale to joules.
_JOULES = {"nJ": 1e-9, "uJ": 1e-6, "mJ": 1e-3, "J": 1.0}
_UTF8 = 0x08000100
_SINT64 = 4
# InstantAmperage wobbles by a few mA around zero on AC; below this it is "neither
# charging nor discharging" rather than a real reading.
_IDLE_MA = 20

PLACEHOLDER = "功耗 —"


# ------------------------------------------------------------------ pure logic
def watts_from_delta(value: int | float, unit: str, dt: float) -> float | None:
    """Joules in this sampling window → watts. None for an unknown unit or a zero window."""
    scale = _JOULES.get((unit or "").strip())
    if scale is None or dt <= 0:
        return None
    return float(value) * scale / dt


def energy_watts(channels, dt: float) -> dict[str, float]:
    """Aggregate energy rails → {rail name: watts}, dropping rails that did not move.

    `channels` is [(name, unit, value)] as read from one IOReport delta. Only the aggregate
    rails (names ending in " Energy": CPU Energy, GPU Energy, PCIe Port 0 Energy …) are
    energy counters; the per-block entries beside them (GPU0, PCPU1DTL2a, ANE0, DRAM0) are
    residency-style counters in the same group and must NOT be summed in — including them
    would silently inflate the number.
    """
    out: dict[str, float] = {}
    for name, unit, value in channels:
        if not str(name).strip().endswith(" Energy"):
            continue
        watts = watts_from_delta(value, unit, dt)
        if watts is not None and watts > 0.0:
            out[str(name)] = watts
    return out


def battery_watts(voltage_mv: int | None, amperage_ma: int | None) -> float | None:
    """mV × mA → watts (µW/1e6), signed: negative current means discharging."""
    if not voltage_mv or amperage_ma is None:
        return None
    return voltage_mv * amperage_ma / 1e6


def describe(battery: dict | None, energy: dict[str, float]) -> tuple[str, str]:
    """(panel text, tooltip) for one reading — the only place the wording lives.

    Precedence is by usefulness: battery power is the machine's real draw and wins while
    discharging; a charge rate is next best but is NOT the system's draw, so it is labelled
    充电; with no battery number at all the GPU rail is the one honest thing left.
    """
    gpu = next((v for k, v in energy.items() if k.startswith("GPU")), None)
    cpu = next((v for k, v in energy.items() if k.startswith("CPU")), None)

    lines = []
    if battery:
        lines.append(f"电池 {battery['watts']:.1f} W"
                     + ("（放电 ≈ 整机功耗）" if not battery["charging"] else "（充电功率）"))
    if gpu is not None:
        lines.append(f"GPU {gpu:.1f} W")
    if cpu is not None:
        lines.append(f"CPU {cpu:.1f} W")
    lines.append("CPU/整机功耗需要 root（powermetrics），本面板不显示读不到的数值")
    tip = "\n".join(lines)

    if battery and not battery["charging"]:
        return f"电池 {battery['watts']:.1f} W", tip
    if battery and battery["charging"]:
        return f"充电 {battery['watts']:.1f} W", tip
    if gpu is not None:
        return f"GPU {gpu:.1f} W", tip
    return PLACEHOLDER, tip


# ------------------------------------------------------------------ CoreFoundation/IOKit
class _CF:
    """Minimal CoreFoundation surface, loaded once and shared by both readers."""

    def __init__(self):
        self.cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        self.cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        self.cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                      ctypes.c_uint32]
        self.cf.CFStringGetCString.restype = ctypes.c_bool
        self.cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                               ctypes.c_int64, ctypes.c_uint32]
        self.cf.CFArrayGetCount.restype = ctypes.c_int64
        self.cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        self.cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
        self.cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.cf.CFDictionaryGetValue.restype = ctypes.c_void_p
        self.cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.cf.CFNumberGetValue.restype = ctypes.c_bool
        self.cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                             ctypes.c_void_p]
        self.cf.CFRelease.restype = None
        self.cf.CFRelease.argtypes = [ctypes.c_void_p]
        self._strings: dict[str, int] = {}

    def string(self, text: str) -> int:
        """Interned CFString — created once, kept for the process lifetime."""
        ref = self._strings.get(text)
        if ref is None:
            ref = self.cf.CFStringCreateWithCString(None, text.encode(), _UTF8)
            self._strings[text] = ref
        return ref

    def text(self, ref) -> str:
        if not ref:
            return ""
        buf = ctypes.create_string_buffer(256)
        if not self.cf.CFStringGetCString(ref, buf, 256, _UTF8):
            return ""
        return buf.value.decode(errors="replace")

    def number(self, ref) -> int | None:
        if not ref:
            return None
        out = ctypes.c_int64()
        if not self.cf.CFNumberGetValue(ref, _SINT64, ctypes.byref(out)):
            return None
        return int(out.value)


class _EnergyReader:
    """IOReport "Energy Model" subscription. Lazily opened; unavailable is not fatal."""

    GROUP = "Energy Model"
    LIB = "/usr/lib/libIOReport.dylib"

    def __init__(self, cf: _CF):
        self.cf = cf
        self.ok = False
        self._prev = None
        self._prev_ts = 0.0
        try:
            self.io = ctypes.CDLL(self.LIB)
        except OSError:
            return
        io = self.io
        io.IOReportCopyChannelsInGroup.restype = ctypes.c_void_p
        io.IOReportCopyChannelsInGroup.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_uint64, ctypes.c_uint64,
                                                   ctypes.c_uint64]
        io.IOReportCreateSubscription.restype = ctypes.c_void_p
        io.IOReportCreateSubscription.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_void_p),
                                                  ctypes.c_uint64, ctypes.c_void_p]
        io.IOReportCreateSamples.restype = ctypes.c_void_p
        io.IOReportCreateSamples.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p]
        io.IOReportCreateSamplesDelta.restype = ctypes.c_void_p
        io.IOReportCreateSamplesDelta.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_void_p]
        for name in ("IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel"):
            fn = getattr(io, name)
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p]
        io.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
        io.IOReportSimpleGetIntegerValue.argtypes = [ctypes.c_void_p, ctypes.c_int32]

        try:
            # The channel dict and the subscription are kept for the process lifetime:
            # every sample we take is a delta against a sample we own, and releasing the
            # subscription would invalidate the retained previous sample.
            self.channels = io.IOReportCopyChannelsInGroup(cf.string(self.GROUP), None, 0, 0, 0)
            if not self.channels:
                return
            self.subbed = ctypes.c_void_p()
            self.sub = io.IOReportCreateSubscription(None, self.channels,
                                                     ctypes.byref(self.subbed), 0, None)
            if not self.sub:
                return
            self.ok = True
        except Exception:
            self.ok = False

    def _take(self):
        return self.io.IOReportCreateSamples(self.sub, self.subbed, None)

    def sample(self) -> tuple[dict[str, float], float] | None:
        """(rail → watts, dt) or None on the first call / any failure.

        Each sample dict owns its channels, so both are released before returning — this
        runs for hours at 1 Hz and a leak here would show up as slow growth.
        """
        if not self.ok:
            return None
        try:
            now = time.perf_counter()
            current = self._take()
            if not current:
                return None
            previous, prev_ts = self._prev, self._prev_ts
            self._prev, self._prev_ts = current, now
            if previous is None:
                return None                      # a delta needs two samples
            delta = self.io.IOReportCreateSamplesDelta(previous, current, None)
            self.cf.cf.CFRelease(previous)        # ours to free; `current` is now the baseline
            if not delta:
                return None
            try:
                rows = self._read(delta)
            finally:
                self.cf.cf.CFRelease(delta)
            return rows, max(1e-6, now - prev_ts)
        except Exception:
            return None

    def _read(self, delta) -> list[tuple[str, str, int]]:
        """Raw (name, unit, value) triples out of one delta — the pure fold happens later."""
        array = self.cf.cf.CFDictionaryGetValue(delta, self.cf.string("IOReportChannels"))
        if not array:
            return {}
        rows = []
        for i in range(self.cf.cf.CFArrayGetCount(array)):
            channel = self.cf.cf.CFArrayGetValueAtIndex(array, i)
            rows.append((self.cf.text(self.io.IOReportChannelGetChannelName(channel)),
                         self.cf.text(self.io.IOReportChannelGetUnitLabel(channel)),
                         int(self.io.IOReportSimpleGetIntegerValue(channel, 0))))
        return rows


_iokit_handle = None    # the resolved CDLL, False once the library proved unavailable


def _iokit():
    """IOKit handle + prototypes, resolved once.

    read_battery runs ~1 Hz on the app's main thread; redoing find_library + CDLL +
    the prototype setup on every call was pure overhead there — the registry read
    itself is ~40 µs. A machine without IOKit is remembered as False so the failed
    lookup is not retried every second either.
    """
    global _iokit_handle
    if _iokit_handle is None:
        try:
            iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
        except OSError:
            _iokit_handle = False
            return False
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceGetMatchingService.restype = ctypes.c_uint32
        iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
        iokit.IORegistryEntryCreateCFProperty.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
                                                          ctypes.c_void_p, ctypes.c_uint32]
        iokit.IOObjectRelease.restype = ctypes.c_uint32
        iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
        _iokit_handle = iokit
    return _iokit_handle


def read_battery(cf: _CF | None = None) -> dict | None:
    """{'watts', 'charging', 'on_ac'} from AppleSmartBattery, or None (desktop Mac)."""
    cf = cf or _CF()
    iokit = _iokit()
    if iokit is False:
        return None
    try:
        # IOServiceGetMatchingService consumes the matching dictionary on success.
        service = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AppleSmartBattery"))
        if not service:
            return None
        values = {}
        try:
            for key in ("Voltage", "InstantAmperage", "Amperage", "IsCharging",
                        "ExternalConnected"):
                ref = iokit.IORegistryEntryCreateCFProperty(service, cf.string(key), None, 0)
                if ref:
                    values[key] = cf.number(ref)
                    cf.cf.CFRelease(ref)
        finally:
            iokit.IOObjectRelease(service)
        mv = values.get("Voltage")
        ma = values.get("InstantAmperage")
        if ma is None:
            ma = values.get("Amperage")
        watts = battery_watts(mv, ma)
        if watts is None or abs(ma) < _IDLE_MA:
            # A battery sitting at ~0 mA (on AC and topped up) says nothing about the
            # machine's draw — better to fall through to the GPU rail than to print 0.0 W.
            return None
        charging = bool(values.get("IsCharging")) or (ma or 0) > 0
        return {"watts": abs(watts), "charging": charging}
    except Exception:
        return None


# ------------------------------------------------------------------ facade
class PowerMeter:
    """Throttled reader the tick calls; returns (text, tooltip) or None.

    `sample()` is deliberately total: it swallows every hardware surprise and keeps
    returning nothing, because a panel that dies while reading a wattage would be a worse
    bug than a missing wattage.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._cf: _CF | None = None
        self._energy: _EnergyReader | None = None
        self._last_ts = 0.0
        self.last_text = ""

    def _open(self):
        if self._cf is None:
            try:
                self._cf = _CF()
                self._energy = _EnergyReader(self._cf)
            except Exception:
                self._cf, self._energy = None, None

    def reading(self) -> dict:
        """One unthrottled reading — used by the CLI and the smoke test."""
        self._open()
        energy: dict[str, float] = {}
        if self._energy is not None and self._energy.ok:
            got = self._energy.sample()
            if got is not None:
                rows, dt = got
                energy = energy_watts(rows, dt)
        battery = read_battery(self._cf) if self._cf is not None else None
        return {"battery": battery, "energy": energy}

    def sample(self) -> tuple[str, str] | None:
        """(text, tooltip) when due, else None. Never raises."""
        try:
            now = time.perf_counter()
            if self._last_ts and now - self._last_ts < self.interval_s:
                return None
            self._last_ts = now
            data = self.reading()
            text, tip = describe(data["battery"], data["energy"])
            self.last_text = text
            return text, tip
        except Exception:
            return None


if __name__ == "__main__":
    import sys

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    meter = PowerMeter(interval_s=1.0)
    print("每行一次读数（首次只建立基线；Ctrl+C 退出）")
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        out = meter.sample()
        if out:
            print(f"  {out[0]}")
        time.sleep(0.25)
