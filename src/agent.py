import random

STARTING_POS = [0, 0]
ACTIONS = ["up", "down", "left", "right", "grab", "climb"]
ITER_LIMIT = 100


class QTable:
    def __init__(self):
        self.table = {}

    def get_Q(self, state, action):
        return self.table.get((state, action), 0)

    def get_location_Q(self, row, col):
        return {
            k: v for k, v in self.table.items() if k[0][0] == row and k[0][1] == col
        }

    def set_Q(self, state, action, value):
        self.table[(state, action)] = value

    def get_best_action(self, state):
        q_values = [self.table.get((state, action), 0) for action in ACTIONS]
        # find the best action(s)
        max_q = max(q_values)
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]

        # break ties randomly
        return random.choice(best_actions)


class Agent:
    def __init__(self):
        self.row = STARTING_POS[0]
        self.col = STARTING_POS[1]
        self.is_breeze = False
        self.is_stench = False
        self.is_glitter = False
        self.has_reward = False
        self.is_terminated = False
        self.q_table = QTable()
        self.is_win = False
        self.has_won_before = False
        self.path = []

    def reset(self):
        self.row = STARTING_POS[0]
        self.col = STARTING_POS[1]
        self.is_breeze = False
        self.is_stench = False
        self.is_glitter = False
        self.has_reward = False
        self.is_terminated = False
        self.is_win = False
        self.path = []

    def best_path(self, env):
        self.reset()
        total_rewards = 0
        state = (
            self.row,
            self.col,
            self.is_breeze,
            self.is_stench,
            self.is_glitter,
            self.has_reward,
        )
        iter = 0
        while not self.is_terminated:
            iter += 1
            if iter > 100:
                return [], "Error:Infinite Loop on Best Path Finding, Try re-training"
            action = self.q_table.get_best_action(state)
            next_state, reward = self.take_action(env, action)
            self.path.append(action)
            total_rewards += reward
            state = next_state
        return self.path, total_rewards

    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(ACTIONS)
        else:
            state = (
                self.row,
                self.col,
                self.is_breeze,
                self.is_stench,
                self.is_glitter,
                self.has_reward,
            )

            return self.q_table.get_best_action(state)

    def take_action(self, env, action):

        if action == "up":
            self.row = max(0, self.row - 1)
            reward = env.get_move_rewards(self)
        elif action == "down":
            self.row = min(env.grid_length - 1, self.row + 1)
            reward = env.get_move_rewards(self)
        elif action == "left":
            self.col = max(0, self.col - 1)
            reward = env.get_move_rewards(self)
        elif action == "right":
            self.col = min(env.grid_length - 1, self.col + 1)
            reward = env.get_move_rewards(self)
        elif action == "grab":
            reward = env.get_grab_rewards(self)
        elif action == "climb":
            reward = env.check_climb(self)
        next_state = (
            self.row,
            self.col,
            self.is_breeze,
            self.is_stench,
            self.is_glitter,
            self.has_reward,
        )
        return next_state, reward


class QLearning(Agent):
    def __init__(self):
        super().__init__()

    def train(self, env, episodes=100, alpha=0.3, gamma=0.90, epsilon=0.3):
        self.reset()
        self.q_table = QTable()
        self.epsilon = epsilon
        total_rewards = 0

        for episode in range(episodes):
            # Q-learning steps
            # Q(S,A) <- Q(S,A) + alpha(R + gamma * maxQ(S',A') - Q(S,A))
            self.reset()
            # epsilon decay
            self.epsilon *= 0.99
            # 1. Start a state S
            state = (
                self.row,
                self.col,
                self.is_breeze,
                self.is_stench,
                self.is_glitter,
                self.has_reward,
            )
            iter = 0
            while not self.is_terminated and iter < ITER_LIMIT:
                iter += 1
                # 2. Select an action A
                action = self.choose_action()
                self.path.append(action)
                # 3. Take the action Q(S,A)
                next_state, reward = self.take_action(env, action)
                total_rewards += reward
                best_next_action = self.q_table.get_best_action(next_state)
                if self.is_terminated:
                    best_next_action = "done"
                    if self.is_win and not self.has_won_before:
                        self.win_episode = episode
                        self.has_won_before = True
                # 4. R + gamma * maxQ(S',A')
                td_target = reward + gamma * self.q_table.get_Q(
                    next_state, best_next_action
                )
                td_error = td_target - self.q_table.get_Q(state, action)
                new_q = self.q_table.get_Q(state, action) + alpha * td_error
                self.q_table.set_Q(state, action, new_q)
                state = next_state
        return total_rewards


class SARSA(Agent):
    def __init__(self):
        super().__init__()

    def train(self, env, episodes=100, alpha=0.3, gamma=0.90, epsilon=0.3):
        self.reset()
        self.q_table = QTable()
        self.epsilon = epsilon
        total_rewards = 0

        for episode in range(episodes):
            # SARSA steps
            # Q(S,A) <- Q(S,A) + alpha(R + gamma * Q(S',A') - Q(S,A))
            self.reset()
            # epsilon decay
            self.epsilon *= 0.99
            # 1. Start a state S
            state = (
                self.row,
                self.col,
                self.is_breeze,
                self.is_stench,
                self.is_glitter,
                self.has_reward,
            )
            action = self.choose_action()
            iter = 0
            while not self.is_terminated and iter < ITER_LIMIT:
                iter += 1
                # 2. Select an action A
                self.path.append(action)
                # 3. Take the action Q(S,A)
                next_state, reward = self.take_action(env, action)
                total_rewards += reward

                if self.is_terminated:
                    next_action = "done"
                    td_target = reward
                    if self.is_win and not self.has_won_before:
                        self.win_episode = episode
                        self.has_won_before = True
                else:
                    # 4. R + gamma * Q(S',A')
                    next_action = self.choose_action()
                    td_target = reward + gamma * self.q_table.get_Q(
                        next_state, next_action
                    )

                td_error = td_target - self.q_table.get_Q(state, action)
                new_q = self.q_table.get_Q(state, action) + alpha * td_error
                self.q_table.set_Q(state, action, new_q)
                state = next_state
                action = next_action
        return total_rewards
