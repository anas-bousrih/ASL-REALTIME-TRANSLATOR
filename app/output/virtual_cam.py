import cv2


class VirtualCamOutput:
    def __init__(self, enabled: bool = False, fps: int = 20, mirror: bool = False):
        self.enabled = bool(enabled)
        self.fps = int(fps)
        self.mirror = bool(mirror)
        self._cam = None
        self._backend_ok = None

    def _ensure_cam(self, width: int, height: int) -> bool:
        if not self.enabled:
            return False
        if self._cam is not None:
            return True

        try:
            import pyvirtualcam
        except Exception:
            if self._backend_ok is not False:
                print("[virtual_cam] pyvirtualcam not installed; disabling virtual camera")
            self._backend_ok = False
            self.enabled = False
            return False

        try:
            self._cam = pyvirtualcam.Camera(width=width, height=height, fps=self.fps)
            self._backend_ok = True
            print(f"[virtual_cam] started: {self._cam.device}")
            return True
        except Exception as exc:
            self._backend_ok = False
            self.enabled = False
            print(f"[virtual_cam] failed to start: {exc}")
            return False

    def send_bgr(self, frame_bgr):
        if not self.enabled:
            return

        h, w = frame_bgr.shape[:2]
        if not self._ensure_cam(width=w, height=h):
            return

        if self.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._cam.send(rgb)
        self._cam.sleep_until_next_frame()

    def close(self):
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
