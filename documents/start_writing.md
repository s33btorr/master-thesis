# Paper Summary

The paper we are replicating is called Estimating Discount Functions with Consumption Choices over the Lifecycle by Laibson et al. On this paper, what they do is generate a lifecycle model to understand which parameters better explain the behaviour houdeholds making consumption, savings and borrowing decisions during their lifetime. The study proposes that a beta-delta model will explain better the data than an exponential clasical model.

They started by collecting the data from different datasets:
1. Aca explicar IPMUS
2. PSID
3. SCF
y los otros detalles. Tambien explicar de que anos proviene la data y que variables se forman con esta y como.

To compare which model a better job explaining the data and with which specific parameters this happened, they used a econometrical method known as Method of Simulated Moments.

Explicar un poco este metodo y como funciona. Escogieron 16 momentos, explicarlos.

The bootsrap was done xxx times where inside of it a lifecycle model was also being run.

The lifecycle model was also build with specific characteristics.
1.
1.
1.
1.
1.
...
Todos los detalles del modelo (edades, AR1, restricciones, funciones, TODO)

Programas que utilizaron: Stata (para recoleccion de data) y Matlab (para la programacion del lifecycle model y del MSM).

Resultados que obtuvieron.

Limitaciones. Entre ellas lo complejo que es el codigo y poco flexible maybe????

# My strategy for replication

Given the complexity that a lifecycle model code can reach by being done manually, we decided to use the package that is being developed by Tim and HMVG called PyLCM.

This package allows not only to do the lifecycle simulations with a simpler (forma de verlo y leerlo y entenderlo) but also allows different shocks that do not require manual coding. For example AR1 process or la muerte random con prob de survival.

While programming this, we did have some changes and struggles that where harder to replicate:
1.
1.
1.
1. Por ejemplo, el tema de las grillas.

quiza aca un apartado explicando el tema de biased.

Poner que todo el codigo se puede mirar en el github y replicar tan solo con un comando.

Quizas poner tambien si me da tiempo lo de la data... si no poner que se utilizaron esos numeros yo que se.

# Results

1. Criticas al paper
1. Criticas al paquete (tema de grillas, dificil de calibrar...)
