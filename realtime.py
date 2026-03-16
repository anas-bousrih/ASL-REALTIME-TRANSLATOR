# has no use in the current repo was used previously for testing the real-time webcam inference before the app was built.

import argparse
import csv
import math
import sys
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import videotransforms
from pytorch_i3d import InceptionI3d


def build_gloss_dict(train_csv):
    gloss_list = []
    with open(train_csv, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            g = row[2].strip()
            if g and g not in gloss_list:
                gloss_list.append(g)
    gloss_list.sort()
    gloss2idx = {g: i for i, g in enumerate(gloss_list)}
    idx2gloss = {i: g for g, i in gloss2idx.items()}
    return gloss2idx, idx2gloss


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


def clip_to_tensor(clip_frames, center_crop):
    clip = np.asarray(clip_frames, dtype=np.float32)  # T,H,W,C
    clip = center_crop(clip)
    clip = torch.from_numpy(clip.transpose([3, 0, 1, 2]))  # C,T,H,W
    return clip.unsqueeze(0)  # B,C,T,H,W


def select_device(device_pref):
    pref = device_pref.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Real-time webcam inference for I3D top-25 model")
    parser.add_argument(
        "--checkpoint",
        default="/Users/anasbousrih/dev/ASL-Citizen/ASL-citizen-code/I3D/saved_weights_may/_v174_0.741945.pt",
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument(
        "--train_csv",
        default="/Users/anasbousrih/Downloads/ASL_Citizen/splits/train.csv",
        help="Train CSV used to build gloss index mapping",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "any", "avfoundation"],
        help="Camera backend selection",
    )
    parser.add_argument("--clip_len", type=int, default=32, help="Number of frames per clip")
    parser.add_argument("--infer_every", type=int, default=2, help="Run inference every N frames")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions to display")
    parser.add_argument(
        "--ema",
        type=float,
        default=0.60,
        help="EMA smoothing factor for probabilities (0 disables, 0.4-0.8 recommended)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for Top-1 label (below this shows 'Uncertain')",
    )
    parser.add_argument("--cam_width", type=int, default=640, help="Requested camera width")
    parser.add_argument("--cam_height", type=int, default=480, help="Requested camera height")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Inference device")
    args = parser.parse_args()

    device = select_device(args.device)
    gloss2idx, idx2gloss = build_gloss_dict(args.train_csv)
    n_classes = len(gloss2idx)

    print(f"Using device: {device}")
    print(f"Num classes: {n_classes}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Realtime config: clip_len={args.clip_len}, infer_every={args.infer_every}, ema={args.ema}")

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(n_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    if args.backend == "avfoundation":
        backend = cv2.CAP_AVFOUNDATION
    elif args.backend == "any":
        backend = cv2.CAP_ANY
    else:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY

    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened() and args.backend == "auto":
        # Fallback to default backend before failing.
        cap.release()
        cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. "
            "On macOS, grant Camera permission to your terminal app "
            "(System Settings -> Privacy & Security -> Camera)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    # Warm up camera stream.
    warmup_ok = False
    warmup_brightness = []
    for _ in range(20):
        ok, f = cap.read()
        if ok:
            warmup_ok = True
            warmup_brightness.append(float(np.mean(f)))
            break
    if not warmup_ok:
        raise RuntimeError(
            "Webcam opened but frames could not be read. "
            "Try --camera 1 (or 2) and ensure no other app is using the camera."
        )
    if warmup_brightness and warmup_brightness[0] < 5.0:
        print(
            "Warning: camera frames look very dark/black. "
            "Try a different camera index (--camera 1/2) or backend (--backend any)."
        )

    center_crop = transforms.Compose([videotransforms.CenterCrop(224)])
    frame_buffer = deque(maxlen=args.clip_len)
    frame_count = 0
    pred_text = "Collecting frames..."
    topk_lines = []
    smoothed_probs = None

    print("Press 'q' to quit.")
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            proc = preprocess_frame(frame)
            frame_buffer.append(proc)

            if len(frame_buffer) == args.clip_len and frame_count % args.infer_every == 0:
                inp = clip_to_tensor(frame_buffer, center_crop).to(device)
                t = inp.size(2)
                per_frame_logits = model(inp, pretrained=False)
                per_frame_logits = F.interpolate(per_frame_logits, size=t, mode="linear", align_corners=False)
                predictions = torch.max(per_frame_logits, dim=2)[0]
                probs = torch.softmax(predictions, dim=1)[0]

                if args.ema > 0:
                    if smoothed_probs is None:
                        smoothed_probs = probs
                    else:
                        smoothed_probs = args.ema * probs + (1.0 - args.ema) * smoothed_probs
                    score_vec = smoothed_probs
                else:
                    score_vec = probs

                values, indices = torch.topk(score_vec, k=min(args.topk, score_vec.numel()))

                top_idx = int(indices[0].item())
                top_prob = float(values[0].item())
                if top_prob >= args.threshold:
                    pred_text = f"Top-1: {idx2gloss[top_idx]} ({top_prob:.3f})"
                else:
                    pred_text = f"Top-1: Uncertain ({top_prob:.3f} < {args.threshold:.2f})"

                topk_lines = []
                for rank, (v, i) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                    topk_lines.append(f"{rank}. {idx2gloss[int(i)]} ({v:.3f})")

            draw = frame.copy()
            cv2.putText(draw, pred_text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            y = 64
            for line in topk_lines:
                cv2.putText(draw, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                y += 26

            cv2.imshow("I3D Real-time ASL (Top-25)", draw)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
