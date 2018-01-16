from bitcoin_collector import *

from deap import base
from deap import creator
from deap import tools

import numpy as np

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.normal, 0.0, 0.1)
toolbox.register("individual", tools.initRepeat, creator.Individual, )