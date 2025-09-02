from agent import QLearning, ACTIONS
from environment import Environment

qlearning = QLearning()
env = Environment()

print(qlearning.train(env=env, episodes=1000))
print("QLearning QTable:")
print(qlearning.q_table.table)
print("\nQLearning Best Path:")
print(qlearning.best_path(env=env))
print(qlearning.q_table.table.get((0, 0, False, False, False, False)), 404)
