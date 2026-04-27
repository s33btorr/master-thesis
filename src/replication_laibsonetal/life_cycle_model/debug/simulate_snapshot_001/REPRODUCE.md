# Reproducing this run

1. Copy `pixi.lock` and `pyproject.toml` from this directory
2. Run `pixi install --frozen` to recreate the exact environment
3. Load the snapshot:

```python
from lcm import load_snapshot
snapshot = load_snapshot("simulate_snapshot_001")
# Re-run: snapshot.model.solve(snapshot.params)
```

Platform: x86_64-Linux
