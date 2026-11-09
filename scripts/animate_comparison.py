"""Animate a saved PINN and reference side by side, without retraining.

Example: python scripts/animate_comparison.py --run runs/forward_full_adam_20261109
"""

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from sciml._artifacts import load_checkpoint, save_arrays, sha256
from sciml.config import Config
from sciml.data import coordinates_at, load_data
from sciml.evaluate import _predict
from sciml.model import PINN


def load_comparison(run, data_path, device_name):
    config = Config.load(run / "config.json")
    checkpoint = load_checkpoint(run / "checkpoint.pt")
    assert checkpoint["config_fingerprint"] == config.fingerprint
    assert checkpoint["data_sha256"] == sha256(data_path)
    with np.load(run / "samples.npz") as samples:
        lower, upper = samples["lower"], samples["upper"]
    data = load_data(data_path)
    frames = np.flatnonzero((data.times >= lower[2]) & (data.times <= upper[2] + 1e-10))
    indices = (np.arange(len(data.xy))[:, None] * len(data.times) + frames).ravel()
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    model = PINN(lower, upper, config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"Predicting {len(indices):,} space-time points on {device}...", flush=True)
    uv, p = _predict(model, coordinates_at(data, indices), config.eval_batch_size, device)
    uv = uv.reshape(len(data.xy), len(frames), 2)
    p = p.reshape(len(data.xy), len(frames))
    predicted = np.stack((uv[:, :, 0], uv[:, :, 1], p), axis=-1)
    reference = np.stack((data.velocity[:, 0, frames], data.velocity[:, 1, frames],
                          data.pressure[:, frames]), axis=-1)
    # The pressure offset is unidentifiable and may vary with time.
    predicted[:, :, 2] -= predicted[:, :, 2].mean(axis=0, keepdims=True)
    reference[:, :, 2] -= reference[:, :, 2].mean(axis=0, keepdims=True)
    save_arrays(run / "animation_fields.npz", xy=data.xy, times=data.times[frames],
                frame_indices=frames, predicted=predicted, reference=reference)
    return data.xy, data.times[frames], frames, predicted, reference


def make_player_html(player, frame_count):
    player = re.sub(r'<link\b[^>]*href="https://[^\"]*font-awesome[^\"]*"[^>]*>', "", player)
    icons = {"minus": "-", "fast-backward": "|<", "step-backward": "<",
             "play": "Play", "pause": "Pause", "step-forward": ">",
             "fast-forward": ">|", "plus": "+"}
    css = "\n".join(f'.fa-{name}::before {{content:"{symbol}";}}' for name, symbol in icons.items())
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PINN prediction vs reference</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1250px;margin:24px auto;padding:0 16px;color:#20252b}
.animation img{max-width:100%;height:auto}.anim-slider{width:95%}
button{cursor:pointer;min-width:38px;min-height:34px;margin:3px}
.fa{font-family:system-ui,sans-serif;font-style:normal;display:inline-block}
.fa-flip-horizontal{transform:scaleX(-1)}
""" + css + """</style><h1>PINN 预测与真实流场：同步对照</h1>
<p>左侧：训练完成的 PINN 预测。右侧：官方参考数据。三行依次为 u、v、p。</p>
<p>点击 Play 播放；拖动滑块或逐帧查看。每行左右及全部时刻共用固定色标。
压力在每个时刻分别减去全空间均值。时间无插值；空间按原始规则网格显示，无平滑。</p>
<p>这是同一个最终模型随物理时间变化的预测，不是训练过程动画。
全空间画面包含初始/边界训练位置，独立测试误差另见 metrics.json。</p>
""" + f"<p>共 {frame_count} 帧。</p>" + player + "</html>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/raw/cylinder_nektar_wake.mat")
    parser.add_argument("--output", type=Path, default=PROJECT / "output/forward_comparison.html")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.fps <= 0 or args.stride <= 0:
        parser.error("fps and stride must be positive")
    xy, times, frames, predicted, reference = load_comparison(args.run, args.data, args.device)
    # Sort original samples into their native y-by-x grid, without interpolation.
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    assert len(xs) * len(ys) == len(xy), "Expected the rectangular reference grid"
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    xgrid, ygrid = np.meshgrid(xs, ys)
    np.testing.assert_allclose(xy[order], np.column_stack((xgrid.ravel(), ygrid.ravel())))
    grids = [value[order].reshape(len(ys), len(xs), len(times), 3)
             for value in (predicted, reference)]
    limits = []
    for field in range(3):
        low = min(predicted[:, :, field].min(), reference[:, :, field].min())
        high = max(predicted[:, :, field].max(), reference[:, :, field].max())
        if field > 0:
            high = max(abs(low), abs(high))
            low = -high
        limits.append((float(low), float(high)))
    # Pixel edges enclose native sample centers exactly.
    dx, dy = np.diff(xs).mean(), np.diff(ys).mean()
    extent = (xs[0]-dx/2, xs[-1]+dx/2, ys[0]-dy/2, ys[-1]+dy/2)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    artists = []
    for row, (label, cmap) in enumerate(zip(("u", "v", "p (mean removed)"), ("viridis", "RdBu_r", "RdBu_r"))):
        row_artists = []
        for col, title in enumerate(("PINN prediction", "Reference")):
            ax = axes[row, col]
            artist = ax.imshow(grids[col][:, :, 0, row], origin="lower", extent=extent,
                               cmap=cmap, norm=Normalize(*limits[row]), interpolation="nearest")
            ax.set(title=f"{label} | {title}", xlabel="x", ylabel="y",
                   xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]))
            row_artists.append(artist)
        fig.colorbar(row_artists[0], ax=list(axes[row]), label="nondimensional", shrink=.85)
        artists.append(row_artists)
    title = fig.suptitle("")

    def update(index):
        for row in range(3):
            for col in range(2):
                artists[row][col].set_data(grids[col][:, :, index, row])
        title.set_text(f"Final forward PINN vs reference | t = {times[index]:.2f} | frame {frames[index]}")

    selected = np.arange(0, len(times), args.stride)
    update(0)
    animation = FuncAnimation(fig, update, frames=selected, interval=1000/args.fps, cache_frame_data=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(selected)} synchronized frames...", flush=True)
    with matplotlib.rc_context({"animation.embed_limit": 250, "savefig.dpi": 85}):
        player = animation.to_jshtml(fps=args.fps, embed_frames=True, default_mode="loop")
    assert len(re.findall(r'frames\[\d+\]\s*=', player)) == len(selected), "Incomplete HTML frames"
    args.output.write_text(make_player_html(player, len(selected)), encoding="utf-8")
    update(len(times)//2)
    fig.savefig(args.output.with_suffix(".png"), dpi=120, facecolor="white")
    plt.close(fig)
    args.output.with_suffix(".source.json").write_text(json.dumps({
        "run": str(args.run.resolve()), "checkpoint_sha256": sha256(args.run / "checkpoint.pt"),
        "dataset": str(args.data.resolve()), "dataset_sha256": sha256(args.data),
        "frames": frames[selected].tolist(), "times": times[selected].tolist(), "fps": args.fps,
        "color_limits_shared_prediction_reference_all_times": limits,
        "pressure_transform": "separate spatial mean removal at every time",
        "rendering": "native rectangular grid, nearest pixels; no temporal interpolation",
        "layout": "rows u/v/p; left prediction, right reference",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
