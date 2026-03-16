import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch

from app.core.config import load_config
from app.core.types import FramePacket
from app.models.i3d_word import I3DWordPredictor
from app.pipeline.postprocess import DecisionFilter
from app.pipeline.preprocess import ClipQueue, preprocess_frame


def list_eval_files(root_dir):
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"eval_root not found: {root}")

    items = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        true_label = label_dir.name
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            for video_path in sorted(label_dir.glob(ext)):
                items.append((true_label, str(video_path)))
    return items


def _norm_col(name):
    return name.strip().lower().replace("_", " ")


def _pick_column(fieldnames, candidates):
    norm = {_norm_col(n): n for n in fieldnames}
    for c in candidates:
        if c in norm:
            return norm[c]
    return None


def _resolve_video_path(raw_path, true_label, eval_csv, videos_dir):
    raw = Path(str(raw_path).strip()).expanduser()
    csv_based_videos = eval_csv.parent.parent / "videos"

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        if videos_dir is not None:
            candidates.append(videos_dir / raw)
            candidates.append(videos_dir / raw.name)
            if true_label:
                candidates.append(videos_dir / true_label / raw.name)
        candidates.append(csv_based_videos / raw)
        candidates.append(csv_based_videos / raw.name)
        if true_label:
            candidates.append(csv_based_videos / true_label / raw.name)

    for c in candidates:
        if c.exists():
            return c

    return None


def list_eval_from_csv(eval_csv_path, videos_dir=None):
    eval_csv = Path(eval_csv_path).expanduser()
    if not eval_csv.exists():
        raise FileNotFoundError(f"eval_csv not found: {eval_csv}")

    videos_dir_path = Path(videos_dir).expanduser() if videos_dir else None

    items = []
    missing = 0

    with eval_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header row")

        video_col = _pick_column(reader.fieldnames, ["video file", "video", "video_path", "file", "path"])
        label_col = _pick_column(reader.fieldnames, ["gloss", "label", "class", "word"])

        if video_col is None or label_col is None:
            raise RuntimeError(
                f"Could not find required columns in CSV. Got columns: {reader.fieldnames}. "
                "Need a video column (e.g. 'Video file') and label column (e.g. 'Gloss')."
            )

        for row in reader:
            true_label = str(row.get(label_col, "")).strip()
            raw_video = str(row.get(video_col, "")).strip()
            if not true_label or not raw_video:
                continue

            resolved = _resolve_video_path(raw_video, true_label, eval_csv, videos_dir_path)
            if resolved is None:
                missing += 1
                continue

            items.append((true_label, str(resolved)))

    return items, missing


def build_decision_from_cfg(rt_cfg):
    return DecisionFilter(
        topk=int(rt_cfg.get("topk", 5)),
        threshold=float(rt_cfg.get("threshold", 0.07)),
        margin=float(rt_cfg.get("margin_threshold", 0.008)),
        ema=float(rt_cfg.get("ema", 0.6)),
        vote_window=int(rt_cfg.get("vote_window", 5)),
        min_votes=int(rt_cfg.get("min_votes", 3)),
        cooldown=int(rt_cfg.get("cooldown", 2)),
        stuck_window=int(rt_cfg.get("stuck_window", 10)),
        stuck_majority=int(rt_cfg.get("stuck_majority", 9)),
        stuck_conf_max=float(rt_cfg.get("stuck_conf_max", 0.09)),
    )


def finalize_clip_label(emitted_labels, frame_labels):
    if emitted_labels:
        return Counter(emitted_labels).most_common(1)[0][0]
    non_uncertain = [x for x in frame_labels if x != "Uncertain"]
    if non_uncertain:
        return Counter(non_uncertain).most_common(1)[0][0]
    return "Uncertain"


def evaluate_video(video_path, true_label, model, rt_cfg):
    clip_len = int(rt_cfg.get("clip_len", 32))
    infer_every = int(rt_cfg.get("infer_every", 2))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    clip = ClipQueue(clip_len=clip_len)
    decision = build_decision_from_cfg(rt_cfg)

    frame_idx = 0
    infer_steps = 0
    uncertain_steps = 0
    frame_pred_labels = []
    emitted_tokens = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        preprocessed = preprocess_frame(frame)
        clip.add(FramePacket(frame_id=frame_idx, bgr=preprocessed))

        if clip.ready() and (frame_idx % infer_every == 0):
            probs = model.predict(clip.to_numpy())["probs"]
            res = decision.update(probs, model.idx2gloss)

            infer_steps += 1
            frame_pred_labels.append(res["label"])

            if res["uncertain"]:
                uncertain_steps += 1
            if res["emit"] is not None:
                emitted_tokens.append(res["emit"])

    cap.release()

    pred_label = finalize_clip_label(emitted_tokens, frame_pred_labels)
    clip_uncertain_rate = (uncertain_steps / infer_steps) if infer_steps > 0 else 1.0

    return {
        "true": true_label,
        "pred": pred_label,
        "video": video_path,
        "total_frames": frame_idx,
        "infer_steps": infer_steps,
        "uncertain_steps": uncertain_steps,
        "uncertain_rate": clip_uncertain_rate,
    }


def compute_metrics(results):
    n = len(results)
    correct = sum(1 for r in results if r["true"] == r["pred"])
    overall_acc = (correct / n) if n > 0 else 0.0

    per_true_total = defaultdict(int)
    per_true_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    total_infer_steps = 0
    total_uncertain_steps = 0

    labels = set(["Uncertain"])
    for r in results:
        t = r["true"]
        p = r["pred"]
        labels.add(t)
        labels.add(p)

        per_true_total[t] += 1
        if t == p:
            per_true_correct[t] += 1

        confusion[t][p] += 1
        total_infer_steps += r["infer_steps"]
        total_uncertain_steps += r["uncertain_steps"]

    per_class_acc = {}
    for label in sorted(per_true_total.keys()):
        denom = per_true_total[label]
        per_class_acc[label] = (per_true_correct[label] / denom) if denom > 0 else 0.0

    uncertain_rate = (total_uncertain_steps / total_infer_steps) if total_infer_steps > 0 else 1.0

    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "confusion": confusion,
        "labels": sorted(labels),
        "n_videos": n,
        "uncertain_rate": uncertain_rate,
    }


def write_confusion_csv(path, labels, confusion):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label", "count"])
        for t in labels:
            for p in labels:
                count = confusion[t][p] if t in confusion and p in confusion[t] else 0
                writer.writerow([t, p, count])


def write_summary_txt(path, metrics):
    with open(path, "w") as f:
        f.write(f"Videos: {metrics['n_videos']}\n")
        f.write(f"Top-1 accuracy: {metrics['overall_acc']:.4f}\n")
        f.write(f"Uncertain rate: {metrics['uncertain_rate']:.4f}\n")
        f.write("\nPer-class accuracy:\n")
        for label, acc in sorted(metrics["per_class_acc"].items()):
            f.write(f"  {label}: {acc:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate webcam clips with app model + postprocess")
    parser.add_argument("--config", default="configs/models.yaml", help="Path to models.yaml")
    parser.add_argument("--eval_root", default="", help="Folder: <eval_root>/<TRUE_LABEL>/*.mp4")
    parser.add_argument("--eval_csv", default="", help="CSV with label+video columns (e.g. Gloss + Video file)")
    parser.add_argument("--videos_dir", default="", help="Base videos directory for relative paths in eval_csv")
    parser.add_argument("--out_dir", default="outputs", help="Where to save summary and confusion CSV")
    args = parser.parse_args()

    if not args.eval_root and not args.eval_csv:
        raise RuntimeError("Provide either --eval_root or --eval_csv")

    cfg = load_config(args.config).raw
    rt_cfg = cfg.get("runtime", {})
    word_cfg = cfg["models"]["word"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = I3DWordPredictor(
        checkpoint=word_cfg["checkpoint"],
        train_csv=word_cfg["train_csv"],
        device=device,
        clip_len=int(rt_cfg.get("clip_len", 32)),
        topk=int(rt_cfg.get("topk", 5)),
    )
    model.load()

    if args.eval_csv:
        files, missing = list_eval_from_csv(args.eval_csv, args.videos_dir or None)
        if missing:
            print(f"Warning: skipped {missing} rows with missing video files")
    else:
        files = list_eval_files(args.eval_root)

    if not files:
        raise RuntimeError("No evaluable videos found.")

    print(f"Found {len(files)} videos")
    results = []
    for i, (true_label, video_path) in enumerate(files, start=1):
        r = evaluate_video(video_path, true_label, model, rt_cfg)
        results.append(r)
        print(
            f"[{i}/{len(files)}] {Path(video_path).name}: "
            f"true={true_label} pred={r['pred']} uncertain_rate={r['uncertain_rate']:.3f}"
        )

    metrics = compute_metrics(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "eval_webcam_summary.txt"
    confusion_path = out_dir / "eval_webcam_confusion.csv"

    write_summary_txt(str(summary_path), metrics)
    write_confusion_csv(str(confusion_path), metrics["labels"], metrics["confusion"])

    print("\n=== Evaluation Summary ===")
    print(f"Videos: {metrics['n_videos']}")
    print(f"Top-1 accuracy: {metrics['overall_acc']:.4f}")
    print(f"Uncertain rate: {metrics['uncertain_rate']:.4f}")
    print(f"Summary saved: {summary_path}")
    print(f"Confusion saved: {confusion_path}")


if __name__ == "__main__":
    main()
