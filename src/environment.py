import numpy as np
from agent import Agent

GRID_LENGTH = 4
START_POINT = [[0, 0]]
WUMPUS_POINT = [[2, 0]]
GOLD_POINT = [[2, 1]]
PIT_POINTS = [[0, 2], [2, 2], [3, 3]]
BREEZE_POINTS = [[0, 1], [0, 3], [1, 2], [2, 1], [2, 3], [3, 2]]
STENCH_POINTS = [[1, 0], [2, 1], [3, 0]]


class Environment:
    def __init__(self):
        self.grid_length = GRID_LENGTH
        self.start_point = np.array(START_POINT)
        self.wumpus_point = np.array(WUMPUS_POINT)
        self.gold_point = np.array(GOLD_POINT)
        self.pit_points = np.array(PIT_POINTS)
        self.breeze_points = np.array(BREEZE_POINTS)
        self.stench_points = np.array(STENCH_POINTS)
        self.reward_grid = np.zeros((GRID_LENGTH, GRID_LENGTH))
        # rewards_initialization
        self.reward_grid.fill(-1)
        for point in np.concatenate((self.wumpus_point, self.pit_points), axis=0):
            row, col = point[0], point[1]
            self.reward_grid[row, col] = -1000

    def get_status(self, agent: Agent):
        position = [agent.row, agent.col]
        agent.is_breeze = False
        agent.is_stench = False
        agent.is_glitter = False
        if position in self.breeze_points.tolist():
            agent.is_breeze = True
        if position in self.stench_points.tolist():
            agent.is_stench = True
        if position in self.gold_point.tolist():
            agent.is_glitter = True
        if (
            position
            in np.concatenate((self.wumpus_point, self.pit_points), axis=0).tolist()
        ):
            agent.is_terminated = True

    def get_move_rewards(self, agent: Agent):
        self.get_status(agent)
        return self.reward_grid[agent.row, agent.col]

    def get_grab_rewards(self, agent: Agent):
        if agent.has_reward:
            return -1
        if [agent.row, agent.col] in self.gold_point.tolist():
            agent.has_reward = True
            return 1000
        else:
            return -1

    def check_climb(self, agent: Agent):
        if [agent.row, agent.col] in self.start_point.tolist():
            agent.is_terminated = True
            return 0
        else:
            return -1


if __name__ == "__main__":
    env = Environment()
    print(env.reward_grid)
