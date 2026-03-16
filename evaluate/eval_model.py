import argparse
import csv
import math
from operator import add
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.utils.data as data_utl
from torchvision import transforms
from tqdm import tqdm

import videotransforms
from app.models.pytorch_i3d import InceptionI3d


def eval_metrics(sorted_args, label):
    res, = np.where(sorted_args == label)
    dcg = 1 / math.log2(res[0] + 2)
    mrr = 1 / (res[0] + 1)
    if res < 1:
        return res[0], [dcg, 1, 1, 1, 1, mrr]
    if res < 5:
        return res[0], [dcg, 0, 1, 1, 1, mrr]
    if res < 10:
        return res[0], [dcg, 0, 0, 1, 1, mrr]
    if res < 20:
        return res[0], [dcg, 0, 0, 0, 1, mrr]
    return res[0], [dcg, 0, 0, 0, 0, mrr]


def load_rgb_frames_from_video(video_path, max_frames=64):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frameskip = 1
    if total_frames >= 96:
        frameskip = 2
    if total_frames >= 160:
        frameskip = 3

    if frameskip == 3:
        start = np.clip(int((total_frames - 192) // 2), 0, 160)
    elif frameskip == 2:
        start = np.clip(int((total_frames - 128) // 2), 0, 96)
    else:
        start = np.clip(int((total_frames - 64) // 2), 0, 64)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for offset in range(0, min(max_frames * frameskip, max(0, total_frames - start))):
        success, img = vidcap.read()
        if not success or img is None:
            break

        if offset % frameskip != 0:
            continue

        h, w, _ = img.shape
        if h < 226 or w < 226:
            d = 226.0 - min(h, w)
            sc = 1 + d / min(h, w)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        h, w, _ = img.shape
        if h > 256 or w > 256:
            img = cv2.resize(img, (256, 256))

        img = (img / 255.0) * 2 - 1
        frames.append(img)

    vidcap.release()
    return np.asarray(frames, dtype=np.float32)


def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class ASLCitizenDataset(data_utl.Dataset):
    def __init__(self, datadir, transforms_, video_file, gloss_dict=None):
        self.datadir = Path(datadir).expanduser()
        self.transforms = transforms_
        self.video_paths = []
        self.video_info = []
        self.labels = []

        if gloss_dict is None:
            gloss_list = []
            with open(video_file, "r") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    g = row[2].strip()
                    if g and g not in gloss_list:
                        gloss_list.append(g)
            gloss_list.sort()
            self.gloss_dict = {g: i for i, g in enumerate(gloss_list)}
        else:
            self.gloss_dict = gloss_dict

        with open(video_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                user = row[0].strip()
                fname = row[1].strip()
                gloss = row[2].strip()
                if not gloss:
                    continue

                p = Path(fname)
                full = p if p.is_absolute() else self.datadir / fname
                if not full.exists() or gloss not in self.gloss_dict:
                    continue

                self.video_paths.append(str(full))
                self.video_info.append({"user": user, "filename": fname, "gloss": gloss})
                self.labels.append(self.gloss_dict[gloss])

    def __len__(self):
        return len(self.video_paths)

    @staticmethod
    def pad(imgs, total_frames=64):
        if imgs.shape[0] == 0:
            raise RuntimeError("Decoded 0 frames")
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]
            if np.random.random_sample() > 0.5:
                pad_img = imgs[0]
            else:
                pad_img = imgs[-1]
            pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
            imgs = np.concatenate([imgs, pad], axis=0)
        return imgs

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label_idx = self.labels[index]

        label = np.zeros(len(self.gloss_dict), dtype=np.float32)
        label[label_idx] = 1
        label = np.tile(label, (64, 1))
        label = np.moveaxis(label, 1, 0)

        imgs = load_rgb_frames_from_video(video_path, 64)
        imgs = self.pad(imgs, 64)
        imgs = self.transforms(imgs)
        ret_img = video_to_tensor(imgs)

        return ret_img, self.video_info[index], torch.tensor(label, dtype=torch.float32)


def choose_device(device_pref):
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
    parser = argparse.ArgumentParser(description="I3D parity evaluator")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--tag", default="top25_eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--interop_threads", type=int, default=2)
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        sharing = mp.get_all_sharing_strategies()
        if "file_descriptor" in sharing:
            mp.set_sharing_strategy("file_descriptor")
        elif "file_system" in sharing:
            mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    device = choose_device(args.device)
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(max(1, args.interop_threads))

    print(f"Using device: {device}")
    print(f"CPU threads: {torch.get_num_threads()} | Interop threads: {torch.get_num_interop_threads()} | DataLoader workers: {args.dataloader_workers} | Batch size: {args.batch_size}")

    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(224),
    ])

    train_ds = ASLCitizenDataset(args.videos_dir, train_transforms, args.train_csv)
    test_ds = ASLCitizenDataset(args.videos_dir, test_transforms, args.eval_csv, gloss_dict=train_ds.gloss_dict)

    print(len(train_ds.gloss_dict))
    print(len(test_ds.gloss_dict))

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.dataloader_workers > 0),
    )

    gloss2idx = train_ds.gloss_dict
    idx2gloss = {v: k for k, v in gloss2idx.items()}

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(len(gloss2idx))
    print(f"Using checkpoint: {args.checkpoint}")
    i3d.load_state_dict(torch.load(args.checkpoint, map_location=device))
    i3d.to(device)
    i3d.eval()

    count_total = 0
    count_correct = [0, 0, 0, 0, 0, 0]
    conf_matrix = np.zeros((len(gloss2idx), len(gloss2idx)))
    user_stats = {}
    user_counts = {}

    with torch.no_grad():
        for inputs, name, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            t = inputs.size(2)
            users = name["user"]

            per_frame_logits = i3d(inputs, pretrained=False)
            ground_truth = torch.max(labels, dim=2)[0]
            per_frame_logits = F.interpolate(per_frame_logits, size=t, mode="linear", align_corners=False)

            predictions = torch.max(per_frame_logits, dim=2)[0]
            y_pred_tag = torch.softmax(predictions, dim=1)
            pred_args = torch.argsort(y_pred_tag, dim=1, descending=True)
            true_args = torch.argmax(ground_truth, dim=1)

            for i in range(len(pred_args)):
                pred = pred_args[i].cpu().numpy()
                gti = true_args[i].cpu().numpy()

                _, counts = eval_metrics(pred, gti)
                count_correct = list(map(add, counts, count_correct))
                count_total += 1

                conf_matrix[gti, pred[0]] += 1

                u = users[i]
                if u not in user_counts:
                    user_counts[u] = 1
                    user_stats[u] = counts
                else:
                    user_counts[u] += 1
                    user_stats[u] = list(map(add, counts, user_stats[u]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"output {args.tag}.txt", "w") as f:
        f.write(f"Total files in eval = {count_total}\n")
        f.write(f"Discounted Cumulative Gain is {count_correct[0]/count_total}\n")
        f.write(f"Mean Reciprocal Rank is {count_correct[5]/count_total}\n")
        f.write(f"Top-1 accuracy is {count_correct[1]/count_total}\n")
        f.write(f"Top-5 accuracy is {count_correct[2]/count_total}\n")
        f.write(f"Top-10 accuracy is {count_correct[3]/count_total}\n")
        f.write(f"Top-20 accuracy is {count_correct[4]/count_total}\n\n")

    with open(out_dir / f"user_stats {args.tag}.txt", "w") as f:
        for u in user_counts:
            f.write(f"User: {u}\n")
            f.write(f"Files: {user_counts[u]}\n")
            f.write(f"Discounted Cumulative Gain is {user_stats[u][0]/user_counts[u]}\n")
            f.write(f"Mean Reciprocal Rank is {user_stats[u][5]/user_counts[u]}\n")
            f.write(f"Top-1 accuracy is {user_stats[u][1]/user_counts[u]}\n")
            f.write(f"Top-5 accuracy is {user_stats[u][2]/user_counts[u]}\n")
            f.write(f"Top-10 accuracy is {user_stats[u][3]/user_counts[u]}\n")
            f.write(f"Top-20 accuracy is {user_stats[u][4]/user_counts[u]}\n\n")

    np.savetxt(out_dir / f"confusion matrix {args.tag}.txt", conf_matrix, fmt="%d")

    with open(out_dir / f"conf_mini_{args.tag}.csv", "w") as f:
        for i in range(len(idx2gloss)):
            g = idx2gloss[i]
            counts = conf_matrix[i]
            acc = conf_matrix[i, i] / np.sum(counts) if np.sum(counts) != 0 else 0
            sorted_args = counts.argsort()[::-1][:5]

            pred0 = idx2gloss[sorted_args[0]]
            count0 = conf_matrix[i, sorted_args[0]]
            pred1 = idx2gloss[sorted_args[1]]
            count1 = conf_matrix[i, sorted_args[1]]
            pred2 = idx2gloss[sorted_args[2]]
            count2 = conf_matrix[i, sorted_args[2]]
            pred3 = idx2gloss[sorted_args[3]]
            count3 = conf_matrix[i, sorted_args[3]]
            pred4 = idx2gloss[sorted_args[4]]
            count4 = conf_matrix[i, sorted_args[4]]

            f.write(
                g + "," + str(acc) + "," + pred0 + "," + str(count0) + "," + pred1 + "," + str(count1) + "," +
                pred2 + "," + str(count2) + "," + pred3 + "," + str(count3) + "," + pred4 + "," + str(count4) + "\n"
            )


if __name__ == "__main__":
    main()
