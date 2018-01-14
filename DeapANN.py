import random
import math
import game
import pygame

import config
from ANN import ANN
from copy import deepcopy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

num_inputs = config.nnet['n_inputs']
num_hidden_nodes = config.nnet['n_h_neurons']
num_outputs = config.nnet['n_outputs']

ms = game.Game()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_real", random.uniform, -10, 10)

#inputs->hidden_layer->outputs

toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_real, num_hidden_nodes*(num_inputs+1)+ (num_hidden_nodes+1)*num_outputs)


toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalANN(individual):
    return ms.get_ind_fitness(individual) ,

toolbox.register("evaluate", evalANN) 

toolbox.register("mate", tools.cxBlend, alpha = .2)
toolbox.register("mutate", tools.mutGaussian, mu = 0, sigma = .1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=2)

def main():

    # new game
    #ms = game.Game()

    random.seed(1)
    NGEN = config.ga['n_gen']
    NPOP = config.game['n_agents']  

    pop = toolbox.population(NPOP)

    for ind in pop:
        ann = ANN(num_inputs, num_hidden_nodes, num_outputs,ind)
        ms.add_agent(ann)

    ms.game_loop(True)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(1, NGEN):
	ms.generation += 1
        ms.reset()

        offspring = toolbox.select(pop, NPOP)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=.5, mutpb=.5)

        for ind in offspring:
            ann = ANN(num_inputs, num_hidden_nodes, num_outputs,ind)
            ms.add_agent(ann)

        ms.game_loop(False)

        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        pop[:] = pop + offspring
	
    
    raw_input("Training is over!")
    while True:
    	ms.game_loop(True)

    
    pygame.quit()

if __name__ == "__main__":
    main()
