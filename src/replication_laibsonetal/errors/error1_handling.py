from pprint import pprint
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import h5py


from config import SRC, BLD, DEBUG


snapshots = list(DEBUG.glob("solve_snapshot_*"))

latest_snapshot = max(
    snapshots,
    key=lambda p: int(p.name.split("_")[-1])
)

print(f"Using snapshot: {latest_snapshot.name}")


with h5py.File(latest_snapshot / "arrays.h5", "r") as f:
    V = f["70/retirement/V_arr"][:]


fig = go.Figure(data=go.Heatmap(z=V[0, 0]))
fig.update_layout(title="Value function slice [0,0]")
fig.show()


fig, ax = plt.subplots(figsize=(9, 5.8))

im = ax.imshow(
    V[0, 0],
    aspect="auto",
    origin="lower",
    cmap="viridis"
)

plt.colorbar(im, ax=ax, label="Value")

ax.set_xlabel("Illiquid wealth grid")
ax.set_ylabel("Liquid wealth grid")

plt.tight_layout()

output_path = SRC / BLD / "figures" / "error1_value_function.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figure saved to: {output_path}")
