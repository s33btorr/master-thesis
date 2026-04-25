Hi! Here an explanation of what is working and what is not.

## Problem:
The problem is with the grids of total wealth (liquid and illiquid) and with the actions of investment (liquid and iliquid)

### Grids that work:
The grids that work and the simulation runs without any problem (apparently) are, for the states:
```python
"wealth": LinSpacedGrid(start=-1500, stop=400000, n_points=50), 
"wealth_illiquid": LinSpacedGrid(start=0, stop=3500000, n_points=50),
```
And for the actions:
```python
"investment_x": LinSpacedGrid(start=-1500, stop=1500, n_points=100),
"investment_z": LinSpacedGrid(start=-1500, stop=1500, n_points=50),
```
As you can see, the problem with these grids are several:
1. The individuals are not able to get very high levels of investment (from -150 until 150) and we want that they are able to take any level that is allowed given the restrictions.
1. The state `wealth` cannot be negative, and we want that individuals are able to be on debt.

### Grids that does not work:

### First problem: 
Leaving the second problem apart, we first focus on the first problem where we would like individuals to have possibilities of higher levels of investments.

We tried a grid like this:
```python
"investment_x": LinSpacedGrid(start=-15_000, stop=15_000, n_points=100),
"investment_z": LinSpacedGrid(start=-15_000, stop=15_000, n_points=50),
```
We can notice that is should be still inside the possible numbers of the total wealth grid in both cases. My theory is that it does not respect the restriction and then the model explodes.
We receive for every period (in both, backward ind and forward iter) the message: `NaN/Inf in V_arr for regime 'retirement' at age 90`.

### Second problem:
Now we leave the first problem apart
sirve con -150 en vez de 0.1. probar con numeros mayores...
### How to run the project? 
```bash
pixi run python task_lcm.py
```