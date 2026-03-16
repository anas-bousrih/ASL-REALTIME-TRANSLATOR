import time

import cv2
import torch

from app.capture.webcam import WebcamCapture
from app.core.types import FramePacket
from app.pipeline.preprocess import ClipQueue, preprocess_frame
from app.pipeline.postprocess import DecisionFilter
from app.models.i3d_word import I3DWordPredictor
from app.output.transcript import TranscriptBuffer
from app.output.llm_rewrite import LLMRewriter
from app.output.virtual_cam import VirtualCamOutput


def _wrap_caption(text: str, max_chars: int = 42):
    words = text.strip().split()
    if not words:
        return []

    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    if len(lines) > 2:
        lines = lines[-2:]
    return lines


def _draw_caption(frame, lines):
    if not lines:
        return

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    line_height = 34
    base_y = h - 70 if len(lines) == 2 else h - 40

    for i, line in enumerate(lines):
        (text_w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        x = max(16, (w - text_w) // 2)
        y = base_y + i * line_height
        cv2.putText(frame, line, (x, y), font, scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def run_session(
    checkpoint: str,
    train_csv: str,
    camera: int = 0,
    backend: str = "auto",
    cam_width: int = 640,
    cam_height: int = 480,
    clip_len: int = 32,
    topk: int = 5,
    threshold: float = 0.07,
    margin_threshold: float = 0.008,
    infer_every: int = 2,
    ema: float = 0.60,
    vote_window: int = 5,
    min_votes: int = 3,
    cooldown: int = 2,
    stuck_window: int = 10,
    stuck_majority: int = 9,
    stuck_conf_max: float = 0.09,
    debug_probs: bool = False,
    show_preview: bool = True,
    shared_state=None,
    virtual_cam_enabled: bool = False,
    virtual_cam_fps: int = 20,
    virtual_cam_mirror: bool = False,
    llm_enabled: bool = False,
    llm_model: str = "qwen3:4b",
    llm_interval_sec: float = 1.5,
    llm_min_tokens: int = 2,
):
    frame_count = 0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cam = WebcamCapture(camera=camera, backend=backend, width=cam_width, height=cam_height)
    clip = ClipQueue(clip_len=clip_len)
    transcript = TranscriptBuffer(max_tokens=24)

    model = I3DWordPredictor(
        checkpoint=checkpoint,
        train_csv=train_csv,
        device=device,
        clip_len=clip_len,
        topk=topk,
    )
    model.load()

    decision = DecisionFilter(
        topk=topk,
        threshold=threshold,
        margin=margin_threshold,
        ema=ema,
        vote_window=vote_window,
        min_votes=min_votes,
        cooldown=cooldown,
        stuck_window=stuck_window,
        stuck_majority=stuck_majority,
        stuck_conf_max=stuck_conf_max,
    )

    rewriter = LLMRewriter(
        enabled=llm_enabled,
        model=llm_model,
        interval_sec=llm_interval_sec,
        min_tokens=llm_min_tokens,
    )
    virtual_cam = VirtualCamOutput(
        enabled=virtual_cam_enabled,
        fps=virtual_cam_fps,
        mirror=virtual_cam_mirror,
    )

    pred_text = "Collecting frames..."
    topk_lines = []
    sentence_text = ""

    print("Press 'q' to quit.")
    while True:
        if shared_state is not None and hasattr(shared_state, "should_stop") and shared_state.should_stop():
            break

        pkt = cam.read_packet()
        if pkt is None:
            break

        if shared_state is not None and hasattr(shared_state, "consume_clear_transcript") and shared_state.consume_clear_transcript():
            transcript.clear()
            sentence_text = ""

        if shared_state is not None and hasattr(shared_state, "consume_toggle_debug") and shared_state.consume_toggle_debug():
            debug_probs = not debug_probs

        frame_count += 1
        raw_frame = pkt.bgr
        proc_frame = preprocess_frame(raw_frame)

        clip.add(
            FramePacket(
                frame_id=pkt.frame_id,
                bgr=proc_frame,
                timestamp=pkt.timestamp,
            )
        )

        if clip.ready() and (frame_count % infer_every == 0):
            frames = clip.to_numpy()
            probs = model.predict(frames)["probs"]
            res = decision.update(probs, model.idx2gloss)

            if debug_probs:
                print(
                    f"sum(probs)={float(probs.sum().item()):.6f} "
                    f"top1={res['top_conf']:.6f} top2={res['top2_conf']:.6f} "
                    f"margin={res['margin']:.6f} label={res['label']} "
                    f"uncertain={res['uncertain']}"
                )

            if res["uncertain"]:
                reason_str = ",".join(res["reasons"])
                pred_text = f"Top-1: Uncertain ({res['top_conf']:.3f}, m={res['margin']:.3f}, {reason_str})"
            else:
                pred_text = f"Top-1: {res['label']} ({res['top_conf']:.3f})"

            if res["emit"] is not None:
                transcript.add(res["emit"])

            topk_lines = [
                f"{i+1}. {label} ({score:.3f})"
                for i, (label, score) in enumerate(res["topk"])
            ]

        transcript_line = transcript.text()
        if llm_enabled:
            s = rewriter.maybe_rewrite(transcript_line, now_ts=time.time())
            if s:
                sentence_text = s

        if shared_state is not None and hasattr(shared_state, "update"):
            transcript_for_ui = transcript_line if not sentence_text else f"{transcript_line}\nSentence: {sentence_text}"
            shared_state.update(
                pred_text=pred_text,
                topk_lines=topk_lines,
                transcript=transcript_for_ui,
                debug=debug_probs,
            )

        draw = raw_frame.copy()
        caption_source = sentence_text.strip() if sentence_text.strip() else transcript_line.strip()
        caption_lines = _wrap_caption(caption_source, max_chars=42)
        _draw_caption(draw, caption_lines)

        virtual_cam.send_bgr(draw)

        if show_preview:
            cv2.imshow("ASL Realtime", draw)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cam.close()
    virtual_cam.close()
    if show_preview:
        cv2.destroyAllWindows()
