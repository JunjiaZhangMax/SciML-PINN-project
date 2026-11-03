import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

path = r"C:\Users\Retina\Documents\ChatGPT\SciML project\data\raw\cylinder_nektar_wake.mat"
data = loadmat(path)

x, y = data["X_star"].T
times = data["t"].ravel()

# 选择最接近目标时间的一帧
target_time = 10.0  # 可修改，范围为 0～19.9
if not times.min() <= target_time <= times.max():
    raise ValueError("The target time is out of the range!")

# 估计最近的离散采样时间点
k = int(np.argmin(np.abs(times - target_time)))
print(f"Index time:{k}, Actual time:{times[k]:.2f}")

# U_star 的轴：空间点、速度分量、时间
u = data["U_star"][:, 0, k]
v = data["U_star"][:, 1, k]
p = data["p_star"][:, k]
p = p - p.mean()  # 去掉该时刻的空间平均压力

fig, axes = plt.subplots(
    3, 1, figsize=(10, 8), layout="constrained"
)

for ax, values, label in zip(
    axes, [u, v, p], ["u", "v", "p (mean removed)"]
):
    contour = ax.tricontourf(
        x, y, values, levels=40, cmap="viridis"
    )
    fig.colorbar(contour, ax=ax, label=label)
    ax.set(xlabel="x", ylabel="y", title=label)
    ax.set_aspect("equal")

fig.suptitle(f"Reference flow at t={times[k]:.2f}")
plt.show()