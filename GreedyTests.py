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
from Algorithms import GreedyAgent, h_manhattan, h_sap
from FrozenLakeEnv import FrozenLakeEnv
env = FrozenLakeEnv(MAP)

def print_solution(actions,env: FrozenLakeEnv) -> None:
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
    print_solution(path, env)
    
    
def test_pq():
    pq = PriorityQueue()
    pq.put((0, 0))
    pq.put((1, 1))
    assert pq.get() == (0, 0), f"pq.get() should be (0, 0)"

def test_h_manhattan():
    # Test distance between two adjacent cells]
    expected = 1
    assert h_manhattan(0, 1) == expected, f"h_manhattan(0, 1) should be {expected}"
    assert h_manhattan(1, 0) == expected, f"h_manhattan(1, 0) should be {expected}"
    # Test distance between two cells in the same row
    expected = 4
    assert h_manhattan(7, 3) == expected, f"h_manhattan(7, 3) should be {expected}"
    assert h_manhattan(3, 7) == expected, f"h_manhattan(3, 7) should be {expected}"
    # Test distance between two cells in the same column
    expected = 1
    assert h_manhattan(4, 12) == expected, f"h_manhattan(4, 12) should be {expected}"
    assert h_manhattan(12, 4) == expected, f"h_manhattan(12, 4) should be {expected}"
    # Test distance between two cells in a diagonal
    expected = 3
    assert h_manhattan(0, 10) == expected, f"h_manhattan(0, 10) should be {expected}"
    assert h_manhattan(10, 0) == expected, f"h_manhattan(10, 0) should be {expected}"

def test_h_sap():
    # Test when accumulated cost is 0
    expected = 0
    assert h_sap(0, 1, 0) == expected, f"h_sap(0, 1, 0) should be {expected}"
    assert h_sap(7, 3, 0) == expected, f"h_sap(7, 3, 0) should be {expected}"
    assert h_sap(4, 12, 0) == expected, f"h_sap(4, 12, 0) should be {expected}"
    assert h_sap(0, 9, 0) == expected, f"h_sap(0, 9, 0) should be {expected}"
    # Test when accumulated cost is non-zero
    expected = 1
    assert h_sap(0, 1, 9) == expected, f"h_sap(0, 1, 9) should be {expected}"
    expected = 3
    assert h_sap(7, 4, 5) == expected, f"h_sap(7, 4, 5) should be {expected}"
    expected = 5
    assert h_sap(16, 56, 20) == expected, f"h_sap(16, 56, 20) should be {expected}"
    expected = 2
    assert h_sap(0, 9, 15) == expected, f"h_sap(0, 9, 15) should be {expected}"

def main():
    test_greedy_agent()
    test_h_manhattan()
    test_h_sap()
    
if __name__ == "__main__":
    main()