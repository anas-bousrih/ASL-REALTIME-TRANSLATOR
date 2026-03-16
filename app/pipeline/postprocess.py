from collections import deque
import torch


class DecisionFilter:
    def __init__(
        self,
        topk=5,
        threshold=0.08,
        margin=0.01,
        ema=0.6,
        vote_window=10,
        min_votes=9,
        cooldown=6,
        stuck_window=10,
        stuck_majority=9,
        stuck_conf_max=0.09,
    ):
        self.topk = topk
        self.threshold = threshold
        self.margin = margin
        self.ema = ema
        self.min_votes = min_votes
        self.cooldown = cooldown
        self.stuck_majority = stuck_majority
        self.stuck_conf_max = stuck_conf_max

        self.scores = None
        self.history = deque(maxlen=vote_window)
        self.recent_labels = deque(maxlen=stuck_window)
        self.cooldown_left = 0
        self.last_emit = None

    def update(self, probs, idx2label):
        if self.ema > 0:
            self.scores = probs if self.scores is None else self.ema * probs + (1.0 - self.ema) * self.scores
        else:
            self.scores = probs

        values, indices = torch.topk(self.scores, k=min(self.topk, self.scores.numel()))

        top_idx = int(indices[0].item())
        top_label = idx2label[top_idx]
        top_conf = float(values[0].item())
        second_conf = float(values[1].item()) if len(values) > 1 else 0.0
        margin = top_conf - second_conf

        self.recent_labels.append(top_label)
        dominant_count = self.recent_labels.count(top_label)
        stuck_spam = (
            len(self.recent_labels) == self.recent_labels.maxlen
            and dominant_count >= self.stuck_majority
            and top_conf < self.stuck_conf_max
        )

        low_conf = top_conf < self.threshold
        low_margin = margin < self.margin
        uncertain = low_conf or low_margin or stuck_spam

        if uncertain:
            self.history.append("Uncertain")
            stable = False
        else:
            self.history.append(top_label)
            stable = self.history.count(top_label) >= self.min_votes

        emit = None
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        elif (not uncertain) and stable and top_label != self.last_emit:
            emit = top_label
            self.last_emit = top_label
            self.cooldown_left = self.cooldown

        reasons = []
        if low_conf:
            reasons.append("low_conf")
        if low_margin:
            reasons.append("low_margin")
        if stuck_spam:
            reasons.append("stuck")

        return {
            "label": "Uncertain" if uncertain else top_label,
            "top_conf": top_conf,
            "top2_conf": second_conf,
            "margin": margin,
            "uncertain": uncertain,
            "reasons": reasons,
            "emit": emit,
            "topk": [
                (idx2label[int(i)], float(v))
                for v, i in zip(values.tolist(), indices.tolist())
            ],
        }
