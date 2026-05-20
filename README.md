# Master Thesis -> Lifecycle Consumption under Budget Constraints: A Replication of “Estimating Discount Functions with Consumption Choices over the Lifecycle”

This code tries to replicate the lifecycle simulation done by Laibson, David, Sean Chanwook Lee, Peter Maxted, Andrea Repetto, and Jeremy Tobacman published in 2024, using the package done by Tim Mensinger, Maximilian Jahn, Janos Gabler, Hans-Martin von Gaudecker called [PyLCM](https://github.com/OpenSourceEconomics/pylcm). 

For the original code, the link is [*here*](https://doi.org/10.7910/DVN/ZVAZVN).

## How to obtain the results?

The first commands needed to run the Lifecycle simulation are:

```bash
pixi install
pixi shell -e cuda12
cd src/replication_laibsonetal
```
To run the simulation one time and obtain the trends for consumption, accumulation of liquid and illiquid wealth, and income, you need to run this command:

```bash
pixi run python -m lifecycle_model.lcm
```

In order to obtain the MC simulation, and obtain the comparative tables, you should run:

```bash
pixi run python -m analysis.mc_results
```

To run my extension, run:

```bash
pixi run python -m extension.my_lcm
```

And, finally, to obtain the errors, you need to run these commands.

For the errors related to the low grid point in liquid wealth grid, it is important to run it in this order to obtain the right graph:
```bash
pixi run python -m errors.error1_lcm
pixi run python -m errors.error1_handling
```
For the errors of the lowest grid point in the illiquid wealth grid and the error that appears if I do not add extra constraints, run these commands respectively:
```bash
pixi run python -m errors.error2_lcm
pixi run python -m errors.error1_lcm
```


