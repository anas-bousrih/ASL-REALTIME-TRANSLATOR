from collections import deque
import numpy as np
import cv2

def preprocess_frame(frame_bgr):
    img = frame_bgr
    h, w, _ = img.shape

    if h < 226 or w < 226:
        d = 226.0 - min(h, w)
        sc = 1 + d / min(h, w)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

    h, w, _ = img.shape
    if h > 256 or w > 256:
        img = cv2.resize(img, (256, 256))

    if img.shape[0] != 256 or img.shape[1] != 256:
        img = cv2.resize(img, (256, 256))

    img = (img / 255.0) * 2 - 1
    return img.astype(np.float32)

class ClipQueue:
    def __init__(self, clip_len=32):
        self.queue = deque(maxlen=clip_len) 

    def add(self, frame_packet):
        self.queue.append(frame_packet)

    def ready(self):
        return len(self.queue) == self.queue.maxlen

    def to_numpy(self):
        return np.array([p.bgr for p in self.queue], dtype=np.float32)
