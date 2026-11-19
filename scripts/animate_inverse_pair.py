"""Export a synchronized, offline inverse-PINN comparison with shared color scales."""
import base64
import json
from io import BytesIO

import matplotlib as mpl
import numpy as np
from PIL import Image

from animate_comparison import PROJECT, load_comparison, sha256

RUNS = [PROJECT / "runs" / name for name in
        ("inverse_adam5k_lbfgs50k_seed0", "inverse_n100_noise05_seed0")]
OUTPUT = PROJECT / "output/inverse_wake_comparison.html"


def render():
    data = PROJECT / "data/raw/cylinder_nektar_wake.mat"
    loaded = []
    for run in RUNS:
        if json.loads((run / "status.json").read_text())["state"] != "completed":
            raise RuntimeError(f"Training still incomplete: {run.name}")
        # Regenerate from the final checkpoint so cached predictions cannot be stale.
        loaded.append(load_comparison(run, data, "auto"))
    xy, times, frame_ids, dense, truth = loaded[0]
    for a, b in zip(loaded[0][:3], loaded[1][:3]):
        np.testing.assert_array_equal(a, b)
    export_player(xy, times, [truth, dense, loaded[1][3]], OUTPUT,
                  ["真值", "5,000 点 · 无噪声", "100 点 · 5% 噪声"], "反问题尾流对照")
    OUTPUT.with_suffix(".source.json").write_text(json.dumps(dict(
        runs=[dict(path=str(r), checkpoint_sha256=sha256(r/"checkpoint.pt")) for r in RUNS],
        dataset_sha256=sha256(data), frame_indices=frame_ids.tolist(),
        pressure="Separate spatial mean removed per frame and model",
        rendering="Native grid, nearest pixels, no temporal interpolation",
        out_of_range_colors=dict(below="#ff00aa", above="#ffbb00"),
    ), indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT} ({len(times)} frames)", flush=True)



def export_player(xy, times, values, output, columns, title, fps=10):
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    order = np.lexsort((xy[:, 0], -xy[:, 1]))
    gx, gy = np.meshgrid(xs, ys[::-1])
    np.testing.assert_allclose(xy[order], np.column_stack((gx.ravel(), gy.ravel())))
    fields = [a[order].reshape(len(ys), len(xs), len(times), 3)
              for a in values]
    panels = []
    for k, name in enumerate(("u", "v", "p")):
        cmap = mpl.colormaps["RdBu_r" if k else "viridis"]
        palette = [mpl.colors.to_hex(cmap(x)) for x in np.linspace(0, 1, 33)]
        modes = {}
        for mode in ("reference", "full"):
            source = [fields[0]] if mode == "reference" else fields
            lo = float(min(a[..., k].min() for a in source))
            hi = float(max(a[..., k].max() for a in source))
            if k:
                hi = max(abs(lo), abs(hi))
                lo = -hi
            images, outside, ranges = [], [], []
            for t in range(len(times)):
                values = [a[:, :, t, k] for a in fields]
                row = np.concatenate(values, axis=1)
                colors = cmap(np.clip((row-lo)/(hi-lo), 0, 1), bytes=True)[..., :3].copy()
                # Distinct colors expose clipped data instead of hiding extreme predictions.
                colors[row < lo] = [255, 0, 170]
                colors[row > hi] = [255, 187, 0]
                buffer = BytesIO()
                Image.fromarray(colors).save(buffer, format="PNG")
                images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
                outside.append([float(np.mean((a < lo) | (a > hi))*100) for a in values])
                ranges.append([[float(a.min()), float(a.max())] for a in values])
            modes[mode] = dict(lo=lo, hi=hi, images=images, outside=outside, ranges=ranges)
        panels.append(dict(name=name, palette=palette, modes=modes))
    payload = dict(times=times.tolist(), width=len(xs), height=len(ys), panels=panels, columns=columns, title=title, fps=fps)
    template = (PROJECT / "scripts/inverse_comparison_player.html").read_text(encoding="utf-8")
    output.write_text(template.replace("/*PAYLOAD*/", json.dumps(payload)), encoding="utf-8")


if __name__ == "__main__":
    render()
