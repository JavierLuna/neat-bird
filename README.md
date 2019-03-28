# NEAT flappybird

This project is part of the "Skynet también juega a Pacman".
You can find the slides [here](https://slides.com/javierlunamolina/skynet-also-plays-pacman) 

## How the NEAT algorithm works
NEAT or *NeuroEvolution of Augmenting Topologies* is basically genetic algorithms applied to neural network construction.

Let me explain:

When you encounter an AI problem and you decide to apply machine/deep learning to it, 
you have to decide how your neural network architecture will be.

How many hidden layers? How will we connect the layers? Which activation functions will we use?

Unless you are a pro and hit the nail in the first try, 
you will be spending a lot of time tuning all the hyper-parameters of your shiny network.

Little do them data-scientists know, there's a kind of AI algorithm 
(who would have thought that there was life in AI beyond ML uh) which is very very good at optimizing things.

Enter, Genetic algorithms.

### Genetic algorithms
GAs are a kind of algorithm which is very very good at optimization problems, mimicking how natural evolution works.
Remember about Darwin and the survival of the fittest?

Genetic Algorithms go through this (roughly explained) phases:

1. Initialization: A set of random solutions to the problem is generated. We will call this set a `population`
2. Selection: Each and every solution is scored against a *fitness_function*. 
The fitness_function will give us a score that indicates how good the solution was to the problem we want to tackle.
After the scoring, a few solutions will be picked to "breed" the new generation of solutions.
3. Genetic operations: Here, the previously generated solutions will combine one with another and create a new solution.
This process is called *genetic crossover*. 
Also, there's a possibility in which something "new" is introduced in the produced solution, something the parents didn't have.
This is called *mutation*.
4. Termination: Was the problem solved? If not, do the whole process again with the new generation of solutions.

In NEAT's case, the solutions are neural networks.

And in our case, the fitness function is how many pipes we have survived.

## Installation

`pip install -r requirements.txt`


## Training

With your dependencies already installed, do:

`python flappybird.py train --genome_path winner.pk`

This will train your birdie and save it in `winner.pk` when the training is done.

Also, you can specify a custom neat config file as:

`python flappybird.py train --genome_path winner.pk --config config-file`

## Watching your birdie go woooo

`python flappybird.py play --genome_path winner.pk`