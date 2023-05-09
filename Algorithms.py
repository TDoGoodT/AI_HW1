from queue import PriorityQueue
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
    y_distance = np.abs(state // BOARD_SIZE - goal // BOARD_SIZE)
    x_distance = np.abs(state % BOARD_SIZE - goal % BOARD_SIZE)
    return y_distance + x_distance

def h_sap(state: int, goal: int, acc_cost: int):
    return min(h_manhattan(state, goal), acc_cost)


def step(state, action, env):
    env.set_state(state)
    return env.step(action)
class GreedyAgent():
    def __init__(self):
        self.heuristic_fn = h_sap

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, set[int]]:
        def greedy_search(graph, start, goal):
            # Create an empty priority queue and insert the start node
            pq = PriorityQueue()
            pq.put((0, start))

            # Initialize the visited set and parent dictionary
            visited = set()
            parent = {start: None}

            # Continue searching until the priority queue is empty
            while not pq.empty():
                # Get the node with the lowest priority
                current_cost, current = pq.get()

                # Check if we've reached the goal
                if current == goal:
                    path = []
                    while current is not None:
                        path.append(current)
                        current = parent[current]
                    path.reverse()
                    return path, current_cost, set()

                # Mark the current node as visited
                visited.add(current)

                # Check the neighbors of the current node
                for neighbor, cost, terminated in [step(current, action, env) for action in range(env.action_space.n)]:
                    # If the neighbor has not been visited, add it to the priority queue
                    if neighbor not in visited:
                        priority = self.heuristic_fn(neighbor, goal, cost) # Calculate the priority using a heuristic function
                        pq.put((current_cost + cost + priority, neighbor))
                        visited.add(neighbor)
                        parent[neighbor] = current

            # If we couldn't find a path to the goal, return None
            return [], np.inf, set()

        return greedy_search({env.reset(): []}, 0, env.get_goal_states()[0])
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