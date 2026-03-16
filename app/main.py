import argparse

from app.core.config import load_config
from app.runtime.session import run_session


def build_session_kwargs(cfg):
    word_cfg = cfg["models"]["word"]
    rt = cfg.get("runtime", {})

    return dict(
        checkpoint=word_cfg["checkpoint"],
        train_csv=word_cfg["train_csv"],
        camera=int(rt.get("camera", 0)),
        backend=str(rt.get("backend", "auto")),
        cam_width=int(rt.get("cam_width", 640)),
        cam_height=int(rt.get("cam_height", 480)),
        clip_len=int(rt.get("clip_len", 32)),
        infer_every=int(rt.get("infer_every", 2)),
        ema=float(rt.get("ema", 0.6)),
        topk=int(rt.get("topk", 5)),
        threshold=float(rt.get("threshold", 0.07)),
        margin_threshold=float(rt.get("margin_threshold", 0.008)),
        vote_window=int(rt.get("vote_window", 5)),
        min_votes=int(rt.get("min_votes", 3)),
        cooldown=int(rt.get("cooldown", 2)),
        stuck_window=int(rt.get("stuck_window", 10)),
        stuck_majority=int(rt.get("stuck_majority", 9)),
        stuck_conf_max=float(rt.get("stuck_conf_max", 0.09)),
        debug_probs=bool(rt.get("debug_probs", False)),
        show_preview=bool(rt.get("show_preview", False)),
        virtual_cam_enabled=bool(rt.get("virtual_cam_enabled", False)),
        virtual_cam_fps=int(rt.get("virtual_cam_fps", 20)),
        virtual_cam_mirror=bool(rt.get("virtual_cam_mirror", False)),
        llm_enabled=bool(rt.get("llm_enabled", False)),
        llm_model=str(rt.get("llm_model", "qwen3:4b")),
        llm_interval_sec=float(rt.get("llm_interval_sec", 1.5)),
        llm_min_tokens=int(rt.get("llm_min_tokens", 2)),
    )


def main():
    parser = argparse.ArgumentParser(description="ASL Realtime App")
    parser.add_argument("--config", default="configs/models.yaml", help="Path to config yaml")
    parser.add_argument("--ui", choices=["none", "webview"], default="none", help="UI mode")
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    session_kwargs = build_session_kwargs(cfg)

    if args.ui == "webview":
        from app.ui.dashboard import launch_dashboard

        launch_dashboard(session_kwargs)
    else:
        run_session(**session_kwargs)


if __name__ == "__main__":
    main()
