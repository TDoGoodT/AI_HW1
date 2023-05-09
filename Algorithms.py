import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict
BOARD_SIZE = 8

class BFSAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError


class DFSAgent():
    def __init__(self):
        return

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, set[int]]:
        def dfs(state, path, cost, explored):
            if env.is_final_state(state):
                return path, cost, len(explored)
            for action in range(env.action_space.n):
                env.set_state(state)
                next_state, action_cost, terminated = env.step(action)
                if terminated and action_cost == np.inf:
                    explored.add(next_state)
                    continue
                if next_state not in explored:
                    result = dfs(next_state, path + [action], cost + action_cost, explored | {next_state})
                    if result[0]:
                        return result
            return [], np.inf, set()
        return dfs(env.reset(), [env.reset()], 0, set())



class UCSAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError

def h_manhattan(state: int, goal: int):
    return np.abs(state // BOARD_SIZE - goal // BOARD_SIZE) + np.abs(state % BOARD_SIZE - goal % BOARD_SIZE)

def h_sap(state: int, goal: int, acc_cost: int):
    return min(h_manhattan(state, goal), acc_cost)

class GreedyAgent():
    def __init__(self):
        self.h = h_sap

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, set[int]]:
        frontier = [(env.reset(), 0)]
        explored = set()
        while frontier:
            state, cost = frontier.pop(0)
            if state in explored:
                continue
            explored.add(state)
            for action in range(env.action_space.n):
                env.set_state(state)
                next_state, action_cost, terminated = env.step(action)
                if terminated and action_cost == np.inf:
                    explored.add(next_state)
                    continue
                if next_state not in explored:
                    smallest_h = min([self.h(next_state, goal, cost + action_cost) for goal in env.get_goal_states()])
                    frontier.append((next_state, cost + action_cost + smallest_h))
            frontier.sort(key=lambda x: x[1])
        return [], np.inf, []

class WeightedAStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
        raise NotImplementedError   


class IDAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError