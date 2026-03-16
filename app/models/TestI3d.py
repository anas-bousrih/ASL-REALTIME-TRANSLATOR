import numpy as np
import torch
from app.core.config import load_config


from app.models.i3d_word import I3DWordPredictor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cfg = load_config("configs/models.yaml").raw
checkpoint = cfg["models"]["word"]["checkpoint"]
train_csv = cfg["models"]["word"]["train_csv"]

model = I3DWordPredictor(
    checkpoint= checkpoint ,
    train_csv= train_csv,
    device=device,
    clip_len=32,
    topk=5,
)
model.load()

# fake clip: T,H,W,C in [-1,1]
clip = np.random.uniform(-1, 1, size=(32, 256, 256, 3)).astype(np.float32)

out = model.predict(clip)
print("label:", out["label"])
print("confidence:", out["confidence"])
print("topk:", out["topk"][:5])
print("probs shape:", tuple(out["probs"].shape))
