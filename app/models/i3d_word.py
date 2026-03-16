import csv
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import videotransforms
from app.models.base import Predictor
from app.models.pytorch_i3d import InceptionI3d


class I3DWordPredictor(Predictor):
    def __init__(self, checkpoint, train_csv, device, clip_len=32, topk=5):
        self.checkpoint = checkpoint
        self.train_csv = train_csv
        self.device = device
        self.clip_len = clip_len
        self.topk = topk

        self.gloss2idx = {}
        self.idx2gloss = {}
        self.model = None
        self.center_crop = transforms.Compose([videotransforms.CenterCrop(224)])

    def build_gloss_dict(self):
        gloss_list = []
        if not self.train_csv:
            raise RuntimeError("train_csv path is empty. Set models.word.train_csv in configs/models.yaml")
        with open(self.train_csv, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                g = row[2].strip()
                if g and g not in gloss_list:
                    gloss_list.append(g)
        gloss_list.sort()
        self.gloss2idx = {g: i for i, g in enumerate(gloss_list)}
        self.idx2gloss = {i: g for g, i in self.gloss2idx.items()}

    def load(self):
        self.build_gloss_dict()
        n_classes = len(self.gloss2idx)

        model = InceptionI3d(400, in_channels=3)
        model.replace_logits(n_classes)
        state = torch.load(self.checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        self.model = model

    def reset(self):
        pass

    def _clip_to_tensor(self, clip_frames):
        # clip_frames: list or np.ndarray of shape (T,H,W,C)
        clip = np.asarray(clip_frames, dtype=np.float32)
        clip = self.center_crop(clip)
        clip = torch.from_numpy(clip.transpose([3, 0, 1, 2]))  # C,T,H,W
        return clip.unsqueeze(0)  # B,C,T,H,W

    def predict_raw(self, clip_frames):
        # Returns raw probability vector. Output shape: (num_classes,)
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            inp = self._clip_to_tensor(clip_frames).to(self.device)
            t = inp.size(2)
            per_frame_logits = self.model(inp, pretrained=False)
            per_frame_logits = F.interpolate(per_frame_logits, size=t, mode="linear", align_corners=False)
            predictions = torch.max(per_frame_logits, dim=2)[0]
            probs = torch.softmax(predictions, dim=1)[0]
        return probs

    def topk_from_probs(self, probs):
        # probs: tensor shape (num_classes,)
        values, indices = torch.topk(probs, k=min(self.topk, probs.numel()))
        return [(self.idx2gloss[int(i)], float(v)) for v, i in zip(values.tolist(), indices.tolist())]

    def predict(self, clip_frames):
        probs = self.predict_raw(clip_frames)
        topk = self.topk_from_probs(probs)
        label, conf = topk[0]
        return {
            "label": label,
            "confidence": conf,
            "topk": topk,
            "probs": probs,
        }
