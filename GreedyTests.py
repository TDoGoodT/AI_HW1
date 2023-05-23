MAP = [
    "SFFFFFFF",
    "FFFFFTAL",
    "TFFHFFTF",
    "FPFFFHTF",
    "FAFHFPFF",
    "FHHFFFHF",
    "FHTFHFTL",
    "FLFHFFFG",
]
BOARD_SIZE = 8
from queue import PriorityQueue
import time
from IPython.display import clear_output
import numpy as np
from Algorithms import GreedyAgent
from FrozenLakeEnv import FrozenLakeEnv

env = FrozenLakeEnv(MAP)


def print_solution(actions, env: FrozenLakeEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
        state, cost, terminated = env.step(action)
        total_cost += cost
        clear_output(wait=True)

        print(env.render())
        print(f"Timestep: {i + 2}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Cost: {cost}")
        print(f"Total cost: {total_cost}")

        time.sleep(1)

        if terminated is True:
            break


def test_greedy_agent():
    env.reset()
    agent = GreedyAgent()
    path, cost, explored = agent.search(env)
    print(f"Path: {path}, cost: {cost}, explored: {explored}")


def test_pq():
    pq = PriorityQueue()
    pq.put((0, 0))
    pq.put((1, 1))
    assert pq.get() == (0, 0), f"pq.get() should be (0, 0)"


def main():
    test_greedy_agent()


if __name__ == "__main__":
    main()
