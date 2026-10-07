"""Offline regression for the no-root power readout. Run: python -B -m unittest discover -s tests.

The hardware paths (IOReport energy rails, AppleSmartBattery) cannot run in a unit test —
that is what probe/hud_smoke.py and `python src/power.py` are for. What is tested here is
everything that decides the number and the wording, plus the promise that a missing
library or a desktop Mac degrades to a placeholder instead of raising.
"""
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import power


class WattsFromDelta(unittest.TestCase):
    def test_units(self):
        self.assertAlmostEqual(power.watts_from_delta(2_000_000_000, "nJ", 2.0), 1.0)
        self.assertAlmostEqual(power.watts_from_delta(1_500_000, "uJ", 1.5), 1.0)
        self.assertAlmostEqual(power.watts_from_delta(2_000, "mJ", 2.0), 1.0)
        self.assertAlmostEqual(power.watts_from_delta(3, "J", 2.0), 1.5)

    def test_unknown_unit_and_zero_window_are_refused(self):
        self.assertIsNone(power.watts_from_delta(10, "", 1.0))
        self.assertIsNone(power.watts_from_delta(10, "ticks", 1.0))
        self.assertIsNone(power.watts_from_delta(10, "nJ", 0.0))


class EnergyWatts(unittest.TestCase):
    # The real group mixes aggregate energy rails with per-block counters that carry the
    # same units; summing the block counters too would silently inflate the wattage.
    CHANNELS = [
        ("GPU Energy", "nJ", 2_000_000_000),
        ("CPU Energy", "mJ", 1_000),
        ("PCIe Port 0 Energy", "uJ", 0),
        ("GPU0", "mJ", 5_000),
        ("PCPU1DTL2a", "mJ", 4_000),
        ("ANE0", "mJ", 3_000),
    ]

    def test_only_aggregate_rails_count(self):
        got = power.energy_watts(self.CHANNELS, 2.0)
        self.assertAlmostEqual(got["GPU Energy"], 1.0)
        self.assertAlmostEqual(got["CPU Energy"], 0.5)
        self.assertEqual(set(got), {"GPU Energy", "CPU Energy"})

    def test_idle_rails_are_dropped(self):
        self.assertEqual(power.energy_watts([("CPU Energy", "mJ", 0)], 1.0), {})
        self.assertEqual(power.energy_watts([], 1.0), {})


class BatteryWatts(unittest.TestCase):
    def test_millivolt_milliamp_to_watts(self):
        self.assertAlmostEqual(power.battery_watts(12_000, -500), -6.0)   # discharging
        self.assertAlmostEqual(power.battery_watts(12_000, 2_500), 30.0)  # charging

    def test_missing_fields(self):
        self.assertIsNone(power.battery_watts(None, -500))
        self.assertIsNone(power.battery_watts(12_000, None))
        self.assertIsNone(power.battery_watts(0, -500))


class Describe(unittest.TestCase):
    def test_discharging_battery_wins_and_is_labelled_as_the_whole_machine(self):
        text, tip = power.describe({"watts": 8.4, "charging": False}, {"GPU Energy": 1.2})
        self.assertEqual(text, "电池 8.4 W")
        self.assertIn("整机功耗", tip)
        self.assertIn("GPU 1.2 W", tip)
        self.assertIn("root", tip)          # the CPU gap is stated, not faked

    def test_charging_is_not_passed_off_as_system_draw(self):
        text, tip = power.describe({"watts": 45.0, "charging": True}, {"GPU Energy": 1.2})
        self.assertEqual(text, "充电 45.0 W")
        self.assertIn("充电功率", tip)

    def test_gpu_rail_is_the_fallback_on_ac(self):
        text, _ = power.describe(None, {"GPU Energy": 0.3, "CPU Energy": 2.0})
        self.assertEqual(text, "GPU 0.3 W")

    def test_nothing_readable_still_renders_a_placeholder(self):
        text, tip = power.describe(None, {})
        self.assertEqual(text, power.PLACEHOLDER)
        self.assertIn("root", tip)


class Throttle(unittest.TestCase):
    def test_due_once_per_interval_and_never_raises(self):
        meter = power.PowerMeter(interval_s=60)
        meter.reading = lambda: {"battery": {"watts": 2.5, "charging": False},
                                "energy": {"GPU Energy": 1.0}}
        text, tip = meter.sample()
        self.assertEqual(text, "电池 2.5 W")
        self.assertIsNone(meter.sample())          # still inside the interval
        time.sleep(0.01)
        meter.interval_s = 0.0
        self.assertEqual(meter.sample()[0], "电池 2.5 W")

    def test_a_failing_reader_returns_none_instead_of_raising(self):
        meter = power.PowerMeter()
        def boom():
            raise RuntimeError("registry went away")
        meter.reading = boom
        self.assertIsNone(meter.sample())


class Degradation(unittest.TestCase):
    def test_missing_ioreport_library_is_not_fatal(self):
        cf = power._CF()
        saved = power._EnergyReader.LIB
        power._EnergyReader.LIB = "/nonexistent/libIOReport.dylib"
        try:
            reader = power._EnergyReader(cf)
            self.assertFalse(reader.ok)
            self.assertIsNone(reader.sample())
        finally:
            power._EnergyReader.LIB = saved

    def test_meter_without_corefoundation_still_answers(self):
        meter = power.PowerMeter(interval_s=0.0)
        meter._open = lambda: None          # pretend the frameworks are unavailable
        meter._cf = None
        out = meter.sample()
        self.assertIsNotNone(out)
        self.assertEqual(out[0], power.PLACEHOLDER)


if __name__ == '__main__':
    unittest.main()
