import sys
import cv2
from app.core.types import FramePacket


class WebcamCapture:
    def __init__(self, camera=0, backend="auto", width=640, height=480):
        self.cap = None
        candidates = [camera] if camera >= 0 else list(range(5))

        for cam_idx in candidates:
            cap = self._open_candidate(cam_idx, backend, width, height)
            if cap is None:
                continue
            if self._looks_usable(cap):
                self.cap = cap
                print(f"[camera] using camera={cam_idx} backend={backend}")
                break
            cap.release()

        if self.cap is None:
            raise RuntimeError(f"Could not open usable webcam (camera={camera}, backend={backend})")

        self.frame_id = 0

    def read_packet(self) -> FramePacket:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from webcam")
        packet = FramePacket(frame_id=self.frame_id, bgr=frame)
        self.frame_id += 1
        return packet

    def close(self):
        self.cap.release()

    @staticmethod
    def _backend_code(backend):
        if backend == "avfoundation":
            return cv2.CAP_AVFOUNDATION
        if backend == "any":
            return cv2.CAP_ANY
        return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    def _open_candidate(self, camera, backend, width, height):
        be = self._backend_code(backend)
        cap = cv2.VideoCapture(camera, be)
        if not cap.isOpened() and backend == "auto":
            cap.release()
            cap = cv2.VideoCapture(camera, cv2.CAP_ANY)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    @staticmethod
    def _looks_usable(cap):
        # Some macOS camera devices open successfully but return black frames.
        for _ in range(8):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.mean() > 2.0:
                return True
        return False
