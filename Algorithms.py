import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict


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
                if action_cost == np.inf:
                    explored.add(next_state)
                    continue
                if next_state not in explored:
                    result = dfs(next_state, path + [action], cost + action_cost, explored | {next_state})
                    if result[0]:
                        return result
            return [], np.inf, 0.
        return dfs(env.reset(), [env.reset()], 0, set())



class UCSAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError



class GreedyAgent():
    def __init__(self, h):
        self.h = h

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:


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