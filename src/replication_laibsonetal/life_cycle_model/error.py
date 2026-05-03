import h5py
import plotly.graph_objects as go
import numpy as np

with h5py.File("/home/brto/brenda/master-thesis/src/replication_laibsonetal/life_cycle_model/debug/solve_snapshot_053/arrays.h5", "r") as f:
    V = f["70/retirement/V_arr"][:]

fig = go.Figure(data=go.Heatmap(z=V[0,0]))
fig.update_layout(title="Value function slice [0,0]")
fig.show()