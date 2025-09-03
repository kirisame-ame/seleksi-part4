from agent import QLearning, SARSA
from environment import Environment

qlearning = QLearning()
sarsa = SARSA()
env = Environment()

print(qlearning.train(env=env, episodes=1000))
print(sarsa.train(env=env, episodes=1000))

print("QLearning QTable:")
print(qlearning.q_table.table)
print("\nSarsa QTable:")
print(sarsa.q_table.table)
print("\nQLearning Best Path:")
print(qlearning.best_path(env=env))
print("\nSarsa Best Path:")
print(sarsa.best_path(env=env))
print("\nQLearning First Win:")
print(qlearning.win_episode)
print("\nSarsa First Win:")
print(sarsa.win_episode)
