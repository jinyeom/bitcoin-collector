#!/usr/bin/env python

import pygame
import math
import time
import util
import config as c
import random as r
from agent import Agent as A
from target import Target as T

class Dummy:
    def __init__(self):
        self.outputs = [r.uniform(2., 4.), r.uniform(2., 4.)]

    def evaluate(self, meh=0):
        return self.outputs

class Game:
    def __init__(self):
        # pygame setup
        pygame.init()
        pygame.display.set_caption(c.game['g_name'])

        self.clock      = pygame.time.Clock()
        self.display    = pygame.display.set_mode(
                            (c.game['width'], c.game['height']))

        self.agents     = []
        self.targets    = [T() for _ in range(c.game['n_targets'])]
        self.generation = 0

        # save terminal
        print "\033[?47h"

    # add an agent to the game with a brain
    def add_agent(self, brain):
        self.agents.append(A(len(self.agents), brain))

	# kill all the agents
    def reset(self):
        self.agents = []

    # game_loop(False) runs the game without graphics
    def game_loop(self, display=True):
        for i in range(c.game['g_time']):
            done = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            if done:
                break

            self.game_logic()

            # for GA, comment this out for better performance
            if i % c.game['delay'] == 0: self.update_terminal()
            if display: self.process_graphic()

        return [a.fitness for a in self.agents]

    # game logic
    def game_logic(self):
        for a in self.agents:

            a.update(self.targets)

            if a.check_collision(self.targets) != -1:
                self.targets[a.t_closest].reset()
                a.fitness += 1

        self.agents = util.quicksort(self.agents)

	# shows graphics of the game using pygame
    def process_graphic(self):
        self.display.fill((0xff, 0xff, 0xff))

        for t in self.targets:
            t_img = pygame.image.load(c.image['target']).convert_alpha()
            self.display.blit(t_img, (t.position[0], t.position[1]))

        if len(self.agents) == c.game['n_agents']:
            for i in range(c.game['n_best']):
                a_img = pygame.transform.rotate(
                    pygame.image.load(c.image['best']).convert_alpha(),
                    self.agents[i].rotation * -180 / math.pi)
                self.display.blit(a_img, (self.agents[i].position[0],
                                        self.agents[i].position[1]))

            for i in range(c.game['n_best'], c.game['n_agents']):
                a_img = pygame.transform.rotate(
                    pygame.image.load(c.image['agent']).convert_alpha(),
                    self.agents[i].rotation * -180 / math.pi)
                self.display.blit(a_img, (self.agents[i].position[0],
                                        self.agents[i].position[1]))
        else:
            for a in self.agents:
                a_img = pygame.transform.rotate(
                    pygame.image.load(c.image['best']).convert_alpha(),
                                    a.rotation * -180 / math.pi)
                self.display.blit(a_img, (a.position[0], a.position[1]))

        pygame.display.update()
        self.clock.tick(c.game['fps'])

	# moniter agents' performances on the terminal
    def update_terminal(self):
        print "\033[2J\033[H",
        print c.game['g_name'],
        print "\tGEN.: " + str(self.generation),
        print "\tTIME: " + str(time.clock()) + '\n'

        for a in self.agents:
            print "AGENT " + repr(a.number).rjust(2) + ": ",
            print "FITN.:" + repr(a.fitness).rjust(5)

# run the game without GA and ANN
if __name__ == '__main__':
    g = Game()

    for _ in range(c.game['n_agents']):
        g.add_agent(Dummy())

    g.game_loop()
    pygame.quit()
