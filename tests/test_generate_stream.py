"""Offline regression for streaming over the keep-alive pool. Run: python -B -m unittest discover -s tests.

_stream_openai used to bypass _KeepAlivePool via urllib.urlopen — every streamed
regeneration / tone change paid a fresh DNS+TCP+TLS (~0.1–0.3 s). These tests pin the
pool contract against a local HTTP server (no network, no real credentials): SSE deltas
and full text agree, two sequential streams reuse one connection, a gateway that answers
`stream: true` with plain JSON still parses, and HTTP >= 300 keeps the
urllib.error.HTTPError shape the callers catch.
"""
import http.server
import json
import sys
import threading
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from generate import Generator


def _sse_event(obj) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


SSE_BODY = (
    _sse_event({"choices": [{"delta": {"content": "收到\n"}}]}) +
    _sse_event({"choices": [{"delta": {"content": "马上\n"}}]}) +
    b"data: [DONE]\n\n"
)


def _start_server(status: int, ctype: str, body: bytes):
    """A local endpoint that speaks HTTP/1.1 keep-alive and records client ports."""
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"   # without this every response closes the socket

        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            self.rfile.read(length)
            Handler.client_ports.append(self.client_address[1])
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    Handler.client_ports = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, Handler


class StreamOverKeepAlivePool(unittest.TestCase):
    def _url(self, server) -> str:
        return f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"

    def test_deltas_and_full_text_agree_and_connection_is_reused(self):
        server, handler = _start_server(200, "text/event-stream", SSE_BODY)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        gen = Generator(model="m", api="openai")

        frags_a: list[str] = []
        out_a = gen._stream_openai(self._url(server), {}, {"model": "m"},
                                   "m", "deepseek-chat", frags_a.append)
        frags_b: list[str] = []
        out_b = gen._stream_openai(self._url(server), {}, {"model": "m"},
                                   "m", "deepseek-chat", frags_b.append)

        self.assertEqual(out_a, "收到\n马上\n")
        self.assertEqual(frags_a, ["收到\n", "马上\n"])
        self.assertEqual(out_b, out_a)
        # the whole point of the pool: the second request rode the same connection
        self.assertEqual(len(handler.client_ports), 2)
        self.assertEqual(handler.client_ports[0], handler.client_ports[1])

    def test_gateway_answering_plain_json_falls_back_to_one_shot_parse(self):
        payload = json.dumps({"choices": [{"message": {"content": "行1\n行2"}}]}).encode()
        server, _ = _start_server(200, "application/json", payload)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        gen = Generator(model="m", api="openai")

        frags: list[str] = []
        out = gen._stream_openai(self._url(server), {}, {"model": "m"},
                                 "m", "deepseek-chat", frags.append)
        self.assertEqual(out, "行1\n行2")
        self.assertEqual(frags, [])   # nothing streamed, so nothing early-painted

    def test_http_error_keeps_the_httperror_shape(self):
        payload = json.dumps({"error": "boom"}).encode()
        server, _ = _start_server(500, "application/json", payload)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        gen = Generator(model="m", api="openai")

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            gen._stream_openai(self._url(server), {}, {"model": "m"},
                               "m", "deepseek-chat", lambda frag: None)
        self.assertEqual(ctx.exception.code, 500)
        self.assertEqual(ctx.exception.read(), payload)   # callers do e.read()[:160]


if __name__ == "__main__":
    unittest.main()
