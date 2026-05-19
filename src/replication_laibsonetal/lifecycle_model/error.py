import h5py
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from config import BLD, SRC, SEED

with h5py.File("/home/brto/brenda/master-thesis/src/replication_laibsonetal/debug/solve_snapshot_019/arrays.h5", "r") as f:
    V = f["70/retirement/V_arr"][:]

fig = go.Figure(data=go.Heatmap(z=V[0,0]))
fig.update_layout(title="Value function slice [0,0]")
fig.show()

#borrar aca luego
fig, ax = plt.subplots(figsize=(5.9, 3.7))

im = ax.imshow(V[0, 0], aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(im, ax=ax, label="Value")

ax.set_xlabel("Liquid wealth grid")
ax.set_ylabel("Illiquid wealth grid")
ax.set_title("Value function slice [0, 0] — age 70, retirement")

plt.tight_layout()

output_path = SRC / BLD / "figures" / "value_function.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()