from bitcoin_collector import *

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
from tqdm import trange

N_ITER = 500
N_TARGETS = 5
N_GEN = 100
POP_SIZE = 50

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_real", np.random.normal, 0.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, 18)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    nn = NeuralNetwork(4, 2, 3, ind)
    game = Game(Agent(nn), N_TARGETS, render=False)
    return (game.rollout(N_ITER), )

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.2)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=2)

hof = tools.HallOfFame(10)

pop = toolbox.population(POP_SIZE)
for g in trange(N_GEN, desc="Evolving"):
    selected = toolbox.select(pop, POP_SIZE)
    offsprings = algorithms.varAnd(selected, toolbox, cxpb=0.5, mutpb=0.5)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(offsprings, fitnesses):
        ind.fitness.valuse = fit
    hof.update(offsprings)
    pop[:] = offsprings
input("Evolution complete. Press ENTER to continue:")

# Test the best agent.
nn = NeuralNetwork(4, 2, 3, hof[0])
game = Game(Agent(nn), N_TARGETS, render=True)
game.rollout(10000)