"""UDP motion stream for a Blender listener.

This is a development bridge, not a production wearable renderer.
Start blender/aperture_signs_receiver.py inside Blender to consume frames.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005


class BlenderBridge:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = int(port)
        self.enabled = False
        self.frames_sent = 0
        self.last_error: str | None = None
        self.last_send_ms = 0.0
        self.last_ok_at: float | None = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    def configure(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)

    def start(self) -> dict[str, Any]:
        self.enabled = True
        self.last_error = None
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.enabled = False
        return self.status()

    def send(self, payload: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        started = time.perf_counter()
        try:
            packet = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._sock.sendto(packet, (self.host, self.port))
            self.frames_sent += 1
            self.last_error = None
            self.last_ok_at = time.time()
            self.last_send_ms = (time.perf_counter() - started) * 1000.0
            return True
        except OSError as exc:
            self.last_error = str(exc)
            self.last_send_ms = (time.perf_counter() - started) * 1000.0
            return False

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "frames_sent": self.frames_sent,
            "last_error": self.last_error,
            "last_send_ms": round(self.last_send_ms, 3),
            "last_ok_at": self.last_ok_at,
            "label": self._label(),
        }

    def _label(self) -> str:
        if not self.enabled:
            return "STOPPED"
        if self.last_error:
            return "ERROR"
        if self.frames_sent:
            return "STREAMING"
        return "WAITING"

    def close(self) -> None:
        self.enabled = False
        self._sock.close()
