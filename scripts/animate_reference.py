"""Animate reference u/v/p fields from the author's cylinder-wake MAT dataset.

Run from anywhere. Defaults resolve relative to this script's project directory.
HTML is self-contained and needs no browser plugins. Optional GIF uses Pillow;
MP4 requires a separately installed FFmpeg executable.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
import numpy as np
from scipy.io import loadmat


PROJECT = Path(__file__).resolve().parents[1]


def load_fields(path):
    raw = loadmat(path)
    xy = np.asarray(raw["X_star"], dtype=np.float64)
    times = np.asarray(raw["t"], dtype=np.float64).ravel()
    velocity = np.asarray(raw["U_star"], dtype=np.float64)
    pressure = np.asarray(raw["p_star"], dtype=np.float64)
    n, nt = len(xy), len(times)
    if xy.shape != (n, 2) or velocity.shape != (n, 2, nt) or pressure.shape != (n, nt):
        raise ValueError("Expected X_star=(S,2), U_star=(S,2,T), p_star=(S,T)")
    if nt < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Expected multiple strictly increasing time samples")
    if not all(np.isfinite(a).all() for a in (xy, times, velocity, pressure)):
        raise ValueError("Nonfinite source data")
    # A separate spatial offset is removed for EVERY time, never across time.
    pressure = pressure - pressure.mean(axis=0, keepdims=True)
    return xy, times, (velocity[:, 0, :], velocity[:, 1, :], pressure)


def build_animation(path, fps=10, stride=1):
    if fps <= 0 or stride <= 0:
        raise ValueError("fps and stride must be positive")
    xy, times, values = load_fields(path)
    indices = np.arange(0, len(times), stride)
    triangulation = Triangulation(xy[:, 0], xy[:, 1])
    # Compute fixed limits from ALL reference times, not separately per frame.
    u_lo, u_hi = float(values[0].min()), float(values[0].max())
    if u_lo == u_hi:
        u_lo, u_hi = u_lo - 1, u_hi + 1
    limits = [(u_lo, u_hi)]
    for value in values[1:]:
        bound = float(np.abs(value).max()) or 1.0
        limits.append((-bound, bound))
    labels = ("u: horizontal velocity", "v: vertical velocity", "p: spatial mean removed")
    cmaps = ("viridis", "RdBu_r", "RdBu_r")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    artists = []
    for ax, value, label, cmap, (low, high) in zip(axes, values, labels, cmaps, limits):
        artist = ax.tripcolor(triangulation, value[:, 0], shading="gouraud",
                              cmap=cmap, norm=Normalize(low, high))
        fig.colorbar(artist, ax=ax, label="nondimensional")
        ax.set(xlabel="x", ylabel="y", title=label,
               xlim=(xy[:, 0].min(), xy[:, 0].max()),
               ylim=(xy[:, 1].min(), xy[:, 1].max()))
        ax.set_aspect("equal")
        artists.append(artist)
    title = fig.suptitle("")

    def update(index):
        for artist, value in zip(artists, values):
            artist.set_array(value[:, index])
        title.set_text(f"Reference cylinder wake | t={times[index]:.2f} | frame {index}/{len(times)-1}")
        return (*artists, title)

    update(0)
    animation = FuncAnimation(fig, update, frames=indices, interval=1000 / fps,
                              blit=False, repeat=True, cache_frame_data=False)
    metadata = {
        "dataset": str(Path(path).resolve()),
        "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "source": "https://github.com/maziarraissi/PINNs/blob/master/main/Data/cylinder_nektar_wake.mat",
        "field_order": ["u", "v", "p"],
        "velocity_axes": ["space", "component", "time"],
        "pressure_transform": "subtract spatial arithmetic mean independently at each time",
        "spatial_rendering": "native-point triangulation with Gouraud shading (spatial interpolation)",
        "time_interpolation": "none; original reference frames only",
        "frame_indices": indices.tolist(),
        "times": times[indices].tolist(),
        "fps": fps,
        "stride": stride,
        "fixed_color_limits_all_times": dict(zip(("u", "v", "p"), limits)),
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
    }
    return fig, animation, update, metadata


def offline_html(player, metadata):
    # Matplotlib's only external dependency is the optional icon font. Replace
    # it with local text icons so controls also work without internet access.
    player = re.sub(r'<link\b[^>]*href="https://[^\"]*font-awesome[^\"]*"[^>]*>', "", player)
    icons = {"minus": "−", "fast-backward": "|◀", "step-backward": "‹", "play": "▶",
             "pause": "Ⅱ", "step-forward": "›", "fast-forward": "▶|", "plus": "+"}
    css_icons = "\n".join(f'.fa-{name}::before {{content:"{symbol}";}}' for name, symbol in icons.items())
    styles = """
body {font:16px/1.5 system-ui,sans-serif; margin:24px auto; max-width:1100px; padding:0 16px; color:#20252b;}
.animation img {max-width:100%; height:auto;}
.anim-controls {max-width:100%; margin:auto;}
.anim-slider {width:95%;}
button {cursor:pointer; min-width:36px; min-height:34px; margin:3px;}
.fa {font-family:system-ui,sans-serif; font-style:normal; display:inline-block;}
.fa-flip-horizontal {transform:scaleX(-1);}
"""
    return ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Cylinder wake: u / v / p</title><style>" + styles + css_icons + "</style></head><body>"
            "<h1>圆柱尾流：u / v / p 同步动画</h1>"
            "<p>点击 ▶ 播放，Ⅱ 暂停；拖动滑块或使用 ‹ / › 逐帧查看，− / + 调整播放速度。</p>"
            f"<p>共 {len(metadata['frame_indices'])} 帧；初始速度 {metadata['fps']} fps。"
            "三幅图使用整段时间固定的各自色标。压力逐帧去空间均值；时间不插值。</p>"
            + player + "<p>来源：PINNs 原作者参考数据；空间显示使用三角剖分插值。"
            "这里展示参考流场，不是模型预测。</p></body></html>")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/raw/cylinder_nektar_wake.mat")
    parser.add_argument("--output", type=Path, default=PROJECT / "output/reference_animation.html")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth original time frame")
    parser.add_argument("--preview", type=Path, help="Optional PNG of middle selected frame")
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.stride <= 0:
        parser.error("--fps and --stride must be positive")
    suffix = args.output.suffix.lower()
    if suffix not in {".html", ".gif", ".mp4"}:
        parser.error("--output must end with .html, .gif, or .mp4")
    if suffix == ".mp4" and not writers.is_available("ffmpeg"):
        parser.error("MP4 needs FFmpeg. Use .html or .gif without extra installation.")
    fig, animation, update, metadata = build_animation(args.data, args.fps, args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if suffix == ".html":
            with matplotlib.rc_context({"animation.embed_limit": 150, "savefig.dpi": 100}):
                player = animation.to_jshtml(fps=args.fps, embed_frames=True, default_mode="loop")
            embedded = len(re.findall(r'frames\[\d+\]\s*=', player))
            if embedded != len(metadata["frame_indices"]):
                raise RuntimeError(f"Incomplete HTML: {embedded} / {len(metadata['frame_indices'])} frames")
            args.output.write_text(offline_html(player, metadata), encoding="utf-8")
        elif suffix == ".gif":
            animation.save(args.output, writer=PillowWriter(fps=args.fps), dpi=90)
        else:
            animation.save(args.output, writer=FFMpegWriter(fps=args.fps, codec="libx264",
                           extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
        if args.preview:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            update(metadata["frame_indices"][len(metadata["frame_indices"]) // 2])
            fig.savefig(args.preview, dpi=120, facecolor="white")
        args.output.with_suffix(".source.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved {len(metadata['frame_indices'])} frames: {args.output}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
