""" Mine Collector """

import math
import numpy as np

SCREEN_SIZE = 800

POLICY_INPUT_DIM = 4
POLICY_OUTPUT_DIM = 2

AGENT_MIN_SPEED = 2.0
AGENT_MAX_SPEED = 4.0

ROTATION_MIN = -0.1
ROTATION_MAX = 0.1

class Entity(object):
    """ Any entity that has coordinates for its position. 

    Attributes:
        _position: a numpy.ndarray for the entity's coordinates.
    """
    def __init__(self, position):
        self.position = position

class Target(Entity):
    """ Target that an agent must collect to increase fitness. 

    Attributes:
        position: a tuple for the target's position coordinates (x, y).
    """

    def __init__(self):
        super().__init__(np.random.randint(SCREEN_SIZE, size=2))

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
        return np.random.uniform(AGENT_MIN_SPEED, AGENT_MAX_SPEED, 2)

class NeuralNetwork(Policy):
    """ Neural network for an agent's policy.

    Currently, a Policy object returns a random tuple of speed. For an agent to work
    properly, an evolved neural network must be used as a policy to navigate the agent.

    **** Design such a neural network class that extends Policy. ****

    Attributes:
        input_dim: an int that represents the number of inputs.
        output_dim: an int that represents the number of outputs.
        hidden: an int that represents the number of hidden neurons.
    """

    def __init__(self, input_dim, output_dim, hidden):
        super().__init__(input_dim, output_dim)
        self.hidden = hidden

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

        # TODO: Implement this function.

        return

class Agent(Entity):
    def __init__(self, agent_id, neural_net=None):
        super().__init__(np.random.randint(SCREEN_SIZE, size=2))
        self.id = agent_id
        self.speed = np.array([AGENT_MIN_SPEED, AGENT_MIN_SPEED])
        self.rotation = np.random.random()

        if neural_net is None:
            self.policy = Policy(POLICY_INPUT_DIM, POLICY_OUTPUT_DIM)
        else:
            self.policy = neural_net

    @property
    def direction(self):
        return np.array([-math.sin(self.rotation), math.cos(self.rotation)])

    def dir_closest(self, targets):
        """ Return the normalized direction vector towards the closest target.
        Args:
            targets: a list that consists of all Targets.
        Returns:
            A numpy.ndarray for the normalized direction towards the closest target.
        """
        diffs = [self.position - t.position for t in targets]
        dists = [math.sqrt(np.sum(diff ** 2)) for diff in diffs]
        min_dist = min(dists)
        if min_dist == 0.0:
            return diffs[dists.index(min_dist)]
        return diffs[dists.index(min_dist)] / min_dist

    def update(self, targets):
        """ Update the agent's states. """
        outputs = self.policy(np.hstack([self.direction, self.dir_closest(targets)]))
        self.rotation += np.clip(outputs[0] - outputs[1], ROTATION_MIN, ROTATION_MAX)
        self.position += (self.direction * np.sum(outputs)).astype(int)
        self.position = np.clip(self.position, 0, SCREEN_SIZE)

class Game(object):
    def __init__(self, agent, n_targets):
        self.agent = agent
        self.targets = [Target() for _ in range(n_targets)]

    def loop():
        return

if __name__ == "__main__":
    np.random.seed(0)
    agent = Agent(0)
    targets = [Target() for _ in range(10)]
    while True:
        agent.update(targets)