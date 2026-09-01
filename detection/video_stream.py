"""
Threaded camera/RTSP reader that always serves the *latest* frame.

Why this exists
----------------
`cv2.VideoCapture.read()` called synchronously inside a processing loop is
fine for local video files, but it's the classic cause of growing lag on a
**live** source (phone IP-camera app, RTSP stream, webcam): the camera keeps
producing frames in real time regardless of how fast your pipeline consumes
them. If YOLO + tracking + OCR can't keep up frame-for-frame, unread frames
pile up in the OS/network socket buffer. `cap.read()` always returns the
*oldest* unread frame, so the pipeline falls further and further behind the
live moment — which looks exactly like "video in slow motion" / "frame
arrives 2-3 seconds late", and the gap keeps growing the longer it runs.

The fix is to decouple capture from processing: a background thread reads
frames from the source as fast as they arrive and only ever keeps the most
recent one. The processing loop asks for "whatever the current frame is"
and always gets something close to real time — older frames are dropped
instead of queued.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

import cv2


class ThreadedVideoStream:
    """Drop-in-ish replacement for cv2.VideoCapture for live sources.

    Supports the same `.isOpened()`, `.get(prop)`, `.read()`, `.release()`
    surface used by the rest of the pipeline, so switching to it is a
    one-line change in the caller.
    """

    def __init__(self, source, use_tcp_for_rtsp: bool = True):
        # RTSP over UDP drops packets more readily on WiFi (phone cameras are
        # almost always on WiFi), which shows up as corrupt/delayed frames.
        # Forcing TCP trades a little latency for far fewer dropped/garbled
        # frames.
        if use_tcp_for_rtsp and isinstance(source, str) and source.lower().startswith("rtsp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

        self._cap = cv2.VideoCapture(source)
        try:
            # Not all backends honor this, but when they do it further
            # limits how many frames OpenCV itself will buffer internally.
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ok = False
        self._frame_seq = 0          # increments every time a new frame arrives
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    # -- cv2.VideoCapture-compatible surface --------------------------------
    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def get(self, prop_id):
        return self._cap.get(prop_id)

    def start(self) -> "ThreadedVideoStream":
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        # Give the reader a brief head start so the very first .read() call
        # doesn't race an empty buffer.
        for _ in range(50):
            if self._frame_seq > 0 or not self._cap.isOpened():
                break
            time.sleep(0.02)
        return self

    def read(self) -> Tuple[bool, "Optional[object]"]:
        """Return the most recently captured frame (never blocks on I/O)."""
        with self._lock:
            if self._latest_frame is None:
                return self._latest_ok, None
            return self._latest_ok, self._latest_frame.copy()

    def release(self):
        self.stop()

    def stop(self):
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()

    # -- internal -------------------------------------------------------------
    def _reader_loop(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            with self._lock:
                self._latest_ok = ok
                if ok:
                    self._latest_frame = frame
                    self._frame_seq += 1
            if not ok:
                # Source hiccup (WiFi drop, phone app restarted, etc.) —
                # back off briefly instead of spinning, then keep retrying.
                time.sleep(0.2)
