"""Run the two overlapping forward windows and compare their overlap."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciml.config import Config
from sciml.data import load_data, DEFAULT_DATA
from sciml.train import train
from sciml.model import PINN
from sciml.evaluate import _predict
from sciml._artifacts import load_checkpoint, write_json

base = Config.load(ROOT / "runs/forward_baseline/config.json")
runs = ROOT / "runs"
previous = None
models = []
for start in (0, 2):
    cfg = replace(base, name=f"forward_window_{start}_{start+4}_seed0", window_start=start,
                  time_end=start+4, init_from=previous, adam_steps=5000, purpose="pilot")
    run = runs / cfg.name
    status = run / "status.json"
    if not status.exists() or json.loads(status.read_text())["state"] != "completed":
        train(cfg, runs_dir=runs, resume=(run/"checkpoint.pt").exists())
    state = load_checkpoint(run/"checkpoint.pt")["model"]
    model = PINN(state["lower"], state["upper"], cfg)
    model.load_state_dict(state)
    models.append(model.requires_grad_(False).to("cuda"))
    previous = cfg.name

data = load_data(DEFAULT_DATA)
def velocity(model, coords):
    return _predict(model, coords, 4096, torch.device("cuda"))[0]

with np.load(run/"samples.npz") as samples:
    coords = samples["initial_coordinates"]
    assert np.allclose(coords[:, 2], 2)
    np.testing.assert_allclose(samples["initial_velocity"], velocity(models[0], coords), atol=1e-12)

point = np.argmin(np.linalg.norm(data.xy-[4,0], axis=1))
times = data.times[data.times<=6+1e-10]
fig, ax = plt.subplots()
ax.plot(times, data.velocity[point,1,:len(times)], color="black", label="Reference")
for start, model in zip((0,2), models):
    t = times[(times>=start-1e-10)&(times<=start+4+1e-10)]
    coords = np.column_stack((np.tile(data.xy[point],(len(t),1)), t))
    ax.plot(t,velocity(model,coords)[:,1],label=f"Window [{start},{start+4}]")
ax.set(xlabel="t",ylabel="v",title=f"Wake probe at {data.xy[point].round(2)}")
ax.legend()
fig.savefig(run/"handoff_probe.png", dpi=150,bbox_inches="tight")
rms = {}
for t in (2,4):
    coords = np.column_stack((data.xy,np.full(len(data.xy),t)))
    rms[str(t)] = np.sqrt(np.mean((velocity(models[0],coords)-velocity(models[1],coords))**2,axis=0)).tolist()
write_json(run/"handoff.json",dict(teacher_verified=True, rms_difference_uv=rms))
