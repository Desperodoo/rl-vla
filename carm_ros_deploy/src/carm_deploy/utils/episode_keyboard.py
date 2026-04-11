#!/usr/bin/env python3
"""Keyboard controls for inference episode lifecycle only."""

from __future__ import annotations

import threading
from typing import Callable, Optional

try:
    import rospy
    HAS_ROSPY = True
except ImportError:
    HAS_ROSPY = False


class EpisodeKeyboardHandler:
    """Listen for episode control keys without injecting action-space intervention."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._record_callback: Optional[Callable[[str], None]] = None
        self._quit_callback: Optional[Callable[[], None]] = None

    def set_record_callback(self, callback: Callable[[str], None]) -> None:
        self._record_callback = callback

    def set_quit_callback(self, callback: Callable[[], None]) -> None:
        self._quit_callback = callback

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._thread.start()
        self._log("Episode keyboard controls enabled: R=toggle, Y=save, N=discard, S=success, F=failure, Ctrl+C=quit")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _keyboard_loop(self) -> None:
        try:
            import select
            import sys
            import termios
            import tty

            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while self._running:
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        key = sys.stdin.read(1)
                        self._handle_key(key)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception as exc:
            self._log(f"Episode keyboard listener error: {exc}", warn=True)

    def _handle_key(self, key: str) -> None:
        key_lower = key.lower()
        if key_lower == "r" and self._record_callback:
            self._record_callback("toggle")
        elif key_lower == "y" and self._record_callback:
            self._record_callback("confirm")
        elif key_lower == "n" and self._record_callback:
            self._record_callback("discard")
        elif key_lower == "s" and self._record_callback:
            self._record_callback("mark_success")
        elif key_lower == "f" and self._record_callback:
            self._record_callback("mark_failure")
        elif key == "\x03" and self._quit_callback:
            self._quit_callback()

    def _log(self, msg: str, warn: bool = False) -> None:
        if HAS_ROSPY:
            (rospy.logwarn if warn else rospy.loginfo)(msg)
        else:
            print(f"[{'WARN' if warn else 'INFO'}] {msg}")
