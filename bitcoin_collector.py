""" Bitcoin Collector

A very hardworking UT student wants to make some money. But it turns out, the job market isn't
very good. As a result, sadly, she couldn't find an internship for the summer. Now, all she
has is nothing but descent knowledge in evolutionary algorithms and neural networks! So,
this summer, she decided to use her knowledge to make some money on her own, by LITERALLY 
collecting Bitcoins with a robot (because, you know, that's how you mine Bitcoins, right?). Lucky 
for her, there are some Bitcoins lying around in the FRI lab. Even luckier for her, someone already
built a toy car with a Raspberry Pi, so she doesn't have to build one herself! All she has to do 
now is to use Genetic Algorithm to evolve a neural network that controls the car to collect those
Bitcoins.

Here's a twist. Turns out, she wasn't so hardworking as you thought she was. So, as a great
friend that you are, you'll have to do it for her! Come on, your dear friend needs her money.

"""

import math
import numpy as np
import pygame

FPS = 60
SCREEN_SIZE = 600
TILE_SIZE = 50
TARGET_RADIUS = 10
POLICY_INPUT_DIM = 4
POLICY_HIDDEN_DIM = 3
POLICY_OUTPUT_DIM = 2
AGENT_RADIUS = 15
AGENT_MIN_SPEED = 2.0
AGENT_MAX_SPEED = 3.0

class Entity(object):
    """ Any entity that has coordinates for its position. 

    Attributes:
        position: a numpy.ndarray for the entity's coordinates.
        radius: a float that defines the entity's collision barrier.
    """
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def check_collision(self, ent2):
        """ Check collision betwene two entities and return True if there is; False otherwise. """
        dist = math.sqrt(np.sum((self.position - ent2.position) ** 2))
        return dist < (self.radius + ent2.radius)

    def find_closest(self, others):
        """ Return the normalized direction vector towards the closest entity.

        Args:
            ent: an Entity at the starting point.
            others: a list of Entities from which the closest one must be found.

        Returns:
            A numpy.ndarray for the normalized direction towards the closest Entity.
        """
        diffs = [ent.position - self.position for ent in others]
        dists = [math.sqrt(np.sum(diff ** 2)) for diff in diffs]
        i = dists.index(min(dists))
        return others[i].position, (diffs[i] / dists[i] if dists[i] != 0.0 else diffs[i])

class Target(Entity):
    """ Target that an agent must collect to increase fitness. 

    Attributes:
        position: a tuple for the target's position coordinates (x, y).
    """

    def __init__(self):
        super().__init__(np.random.randint(SCREEN_SIZE, size=2), TARGET_RADIUS)

    def reset(self):
        self.position = np.random.randint(SCREEN_SIZE, size=2)

class Policy(object):
    """ A function that defines an agent's behavior. 

    A policy is a function that defines an agent's behavior. In this game's case, 
    an agent's policy decides the speed of its wheels (left and right), given the position
    of the closest target, i.e., policy((target_x, target_y)) = (speed_left, speed_right).
    And the speed of each wheel determines how fast it moves and how much it turns in
    what direction. 

    Attributes:
        input_dim: an int that represents the number of inputs.
        output_dim: an int that represents the number of outputs.
    """

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, inputs):
        return self.get_wheel_speeds(inputs)

    def get_wheel_speeds(self, inputs):
        """ Get random speeds for the left wheel and the right wheel. 
        Args:
            inputs: a numpy.ndarray for the closest target's coordinates.
        Returns:
            A random numpy.ndarray for the speed of each wheel (left, right). 
        """
        return np.random.uniform(0.0, 1.0, 2)

class NeuralNetwork(Policy):
    """ Neural network for an agent's policy.

    Currently, a Policy object returns a random tuple of speed. For an agent to work
    properly, an evolved neural network must be used as a policy to navigate the agent.

    **** Design such a neural network class that extends Policy. ****

    Attributes:
        input_dim: an int that represents the number of inputs.
        output_dim: an int that represents the number of outputs.
        hidden: an int that represents the number of hidden neurons.

        TODO: Add more necessary attributes.

    """

    def __init__(self, input_dim, output_dim, hidden, weights):
        super().__init__(input_dim, output_dim)
        if len(weights) != input_dim * hidden + hidden * output_dim:
            raise ValueError("invalid number of weights")

        # TODO: Add more necessary attributes.

    def __call__(self, inputs):
        return self.get_wheel_speeds(inputs)

    def get_wheel_speeds(self, inputs):
        """ Feed forward this neural network and get wheel speed. 

        This function overrides Policy's get_wheel_speeds so that it no longer returns
        a random tuple for the speed of the agent's wheels.

        Args:
            inputs: a numpy.ndarray for the closest target's coordinates.

        Returns:
            A numpy.ndarray for the neural network's output, which will be used as the speed
            of the agent's wheels.
        """
        return self.forward(inputs)

    def forward(self, inputs):
        """ Feed forward the argument inputs to get the outputs. """
        if len(inputs) != self.input_dim:
            raise ValueError("invalid number of inputs")

        # TODO: Implement this function.

class Agent(Entity):
    def __init__(self, neural_net=None):
        super().__init__(np.random.randint(SCREEN_SIZE, size=2), AGENT_RADIUS)
        self.speed = np.array([AGENT_MIN_SPEED, AGENT_MIN_SPEED])
        self.rotation = np.random.random()
        self.target_dir = np.array([0.0, 0.0])
        self.target_pos = np.array([0.0, 0.0])

        if neural_net is None:
            # If the given neural network is None, use a dummy policy instead.
            self.policy = Policy(POLICY_INPUT_DIM, POLICY_OUTPUT_DIM)
        else:
            self.policy = neural_net

    @property
    def direction(self):
        return np.array([-math.sin(self.rotation), math.cos(self.rotation)])

    def update(self, targets):
        """ Update the agent's states. """
        self.target_pos, self.target_dir = self.find_closest(targets)
        outputs = self.policy(np.hstack([self.direction, self.target_dir]))
        self.speed = outputs * (AGENT_MAX_SPEED - AGENT_MIN_SPEED) + AGENT_MIN_SPEED
        self.rotation += self.speed[0] - self.speed[1]
        self.position += (self.direction * np.sum(self.speed)).astype(int)
        self.position = np.clip(self.position, 0, SCREEN_SIZE)

class Game(object):
    def __init__(self, agent, n_targets, render=False):
        self.agent = agent
        self.targets = [Target() for _ in range(n_targets)]
        self._render = render
        if self._render:
            # Add Pygame related attributes if the game renders.
            pygame.init()
            pygame.display.set_caption("BTC Collector")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font("asset/flipps.otf", 12)
            self.display = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))

            # Add sprite images for targets, agent, and floor.
            self.target_sprite = pygame.image.load("asset/target.png").convert_alpha()
            self.agent_sprite = pygame.image.load("asset/agent.png").convert_alpha()
            self.floor_tile = pygame.image.load("asset/tile.png")

    def rollout(self, n_iter):
        """ Let the agent play the game. 
        
        When the game starts, the agent will start moving around and try to collect targets.
        The game will keep track of the score and return it at the end of the game.

        Args:
            n_iter: an int for the number of iterations of the game loop.

        Returns:
            An int that counts the number of targets that was collected by the agent.
        """
        score = 0
        for i in range(n_iter):
            self.agent.update(self.targets)
            for t in self.targets:
                if self.agent.check_collision(t):
                    score += 1
                    t.reset()
            if self._render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return -1

                # Render floor.
                for r in range(int(SCREEN_SIZE / TILE_SIZE)):
                    for c in range(int(SCREEN_SIZE / TILE_SIZE)):
                        if abs(r - c) % 2:
                            self.display.blit(self.floor_tile, (r * TILE_SIZE, c * TILE_SIZE))
                        else:
                            self.display.blit(pygame.transform.rotate(self.floor_tile, 90), 
                                              (r * TILE_SIZE, c * TILE_SIZE))

                # Render all targets.
                for t in self.targets:
                    t_corner = t.position - np.array([TARGET_RADIUS, TARGET_RADIUS])
                    self.display.blit(self.target_sprite, t_corner)

                # Draw a circle around the closest target.
                pygame.draw.circle(self.display, (0, 255, 0), self.agent.target_pos, 12, 2)

                # Draw a point for the agent's vision
                dot_pos = (self.agent.position + self.agent.direction * 60).astype(int)
                pygame.draw.circle(self.display, (255, 0, 0), dot_pos, 1)

                # Render the agent.
                a_angle = self.agent.rotation * -180 / math.pi
                a_corner = self.agent.position - np.array([AGENT_RADIUS, AGENT_RADIUS])
                self.display.blit(pygame.transform.rotate(self.agent_sprite, a_angle), a_corner)

                # Show the agent's outputs.
                l_str = self.font.render("L_TRACK = %.3f" % self.agent.speed[0], False, (0, 0, 0))
                r_str = self.font.render("R_TRACK = %.3f" % self.agent.speed[1], False, (0, 0, 0))
                self.display.blit(l_str, (2, 0))
                self.display.blit(r_str, (2, 20))

                # Show current score.
                score_str = self.font.render("SCORE = %d" % score, False, (0, 0, 0))
                self.display.blit(score_str, (2, SCREEN_SIZE - 24))

                # Update display.
                pygame.display.update()
                self.clock.tick(FPS)
                
        return score

if __name__ == "__main__":
    np.random.seed(0)
    score = Game(Agent(), 10, render=True).rollout(10000)
    print("Score =", score)
