from queue import PriorityQueue
import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict

BOARD_SIZE = 8

class BFSAgent():
    # def __init__(self) -> None:
    #     raise NotImplementedError

    def __init__(self):
        self.env = None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        cost = 0
        expanded = 0
        OPEN = []
        history = []
        states = []

        state = self.env.get_initial_state()
        dic_state = {'state': state, 'actions': [], 'total_cost': cost}
        states.append(dic_state)        # List of nodes for different paths that were tested
        history.append(state)           # List of history of which state has been tested
        OPEN.append(dic_state)          # List of open nodes to be open in the loop
        if self.env.is_final_state(states[-1]['state']):  # Check whether the initial condition is hte goal
            actions = states[-1]['actions']
            total_cost = states[-1]['total_cost']
            return actions, total_cost, expanded  # Return the goal's data and path

        while len(OPEN) > 0:
            cur = OPEN[0]
            expanded += 1
            for action, successor in env.succ(cur['state']).items():        # Expand each state via its SUCCESSOR
                if cur['total_cost'] == np.inf:                             # If a hole has approached, stay with the father
                    dic_succ = cur
                else:                                                       # If not a hole, define it as a state_dic and a path
                    dic_succ = {'state': successor[0], 'actions': cur['actions'] + [action],
                                'total_cost': cur['total_cost'] + successor[1]}
                if dic_succ['state'] in history:                            # If such state is repeated, ignore it
                    continue
                history.append(dic_succ['state'])                           # Add the state to history
                states.append(dic_succ)                                     # Add the state_dic and data to the states
                OPEN.append(dic_succ)                                       # Add the state_dic to OPEN list to be open in the future
                if self.env.is_final_state(states[-1]['state']):            # Check whether the goal ahs been reached
                    actions = states[-1]['actions']
                    total_cost = states[-1]['total_cost']
                    return actions, total_cost, expanded                    # Return the goal's data and path
            OPEN = OPEN[1:]                                                 # FIFO on each opened node

        return 'NO SOLUTION HAS BEEN FOUND'


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

    # def __init__(self) -> None:
    #     raise NotImplementedError

    def __init__(self):
        self.env = None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        cost = 0
        expanded = 0
        OPEN = heapdict.heapdict()
        OPEN_full = {}
        CLOSE = {}
        states = []

        state = self.env.get_initial_state()
        dic_state = {'state': state, 'actions': [], 'total_cost': cost}
        states.append(dic_state)        # List of nodes for different paths that were tested
        OPEN[state] = dic_state         # Dict of open nodes to be open in the loop
        OPEN[state] = (dic_state['total_cost'], state)    # Heapdict of open nodes to be open in the loop
        OPEN_full[state] = dic_state                      # Dict of open nodes to be open in the loop (full)

        while len(OPEN) > 0:
            cur_snc = OPEN.popitem()                                # Getting the state with the minimum cost
            cur = OPEN_full.pop(cur_snc[0])                         # Get its whole data
            CLOSE[cur['state']] = cur                               # Close it

            if self.env.is_final_state(cur['state']):                # Check whether the goal ahs been reached
                actions = cur['actions']
                total_cost = cur['total_cost']
                return actions, total_cost, expanded                 # Return the goal's data and path

            expanded += 1
            for action, successor in env.succ(cur['state']).items():  # Expand each state via its SUCCESSOR
                if successor[1] == np.inf or successor[0] in CLOSE:     # If a hole has approached, stay with the father
                    continue
                else:                                                       # If not a hole, define it as a state_dic and a path
                    dic_succ = {'state': successor[0], 'actions': cur['actions'] + [action],
                                'total_cost': cur['total_cost'] + successor[1]}
                states.append(dic_succ)                                     # Add the state_dic and data to the states
                if dic_succ['state'] not in CLOSE and dic_succ['state'] not in OPEN:
                    OPEN[dic_succ['state']] = (dic_succ['total_cost'], dic_succ['state'])  # Add state_dic to open if not seen before
                    OPEN_full[dic_succ['state']] = dic_succ
                elif dic_succ['state'] in OPEN and OPEN[dic_succ['state']][0] > dic_succ['total_cost']:
                    OPEN[dic_succ['state']] = (dic_succ['total_cost'], dic_succ['state'])  # Update state_dic to open if f is lower
                    OPEN_full[dic_succ['state']] = dic_succ
                    CLOSE.pop(dic_succ['state'])  # get state_dic out of CLOSE if cost is lower

        return 'NO SOLUTION HAS BEEN FOUND'


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
                        current_action = parent[current]
                        current = None
                        if current_action is not None:
                            current, action = current_action
                            path.append(action)
                    path.reverse()
                    return path, current_cost, set()

                # Mark the current node as visited
                visited.add(current)

                # Check the neighbors of the current node
                for action, neighbor, cost, terminated in [step(current, action, env) for action in range(env.action_space.n)]:
                    # If the neighbor has not been visited, add it to the priority queue
                    if neighbor not in visited:
                        priority = self.heuristic_fn(neighbor, goal, cost) # Calculate the priority using a heuristic function
                        pq.put((current_cost + cost + priority, neighbor))
                        visited.add(neighbor)
                        parent[neighbor] = (current, action)

            # If we couldn't find a path to the goal, return None
            return [], np.inf, set()

        return greedy_search({env.reset(): []}, 0, env.get_goal_states()[0])
class WeightedAStarAgent():

    # def __init__(self):
    #     raise NotImplementedError

    def __init__(self):
        self.env = None

    def h_manhattan(state: int, goal: int, env):
        y_distance = np.abs(state // env.nrow - goal // env.nrow)
        x_distance = np.abs(state % env.ncol - goal % env.ncol)
        return y_distance + x_distance

    def h_msap(state: int, goals, acc_cost: int, env):
        # Checking if goals is single or many
        if type(goals) == int:
            # define as an iterable list
            goals = [goals]
        # Iterate and calculate manhattan distance
        h_man_goal = []
        for goal in goals:
            h_man_goal.append(WeightedAStarAgent.h_manhattan(state, goal, env))
        # Add the portal's cost
        h_man_goal.append(acc_cost)
        return min(h_man_goal)

    def search(self, env: FrozenLakeEnv, h_weight=0.5) -> Tuple[List[int], int, float]:

        self.env = env
        self.env.reset()
        cost = 0
        expanded = 0
        p_cost = 100
        goal_state = env.goals
        OPEN = heapdict.heapdict()
        OPEN_full = {}
        CLOSE = {}
        states = []
        state = self.env.get_initial_state()
        dic_state = {'state': state, 'actions': [], 'total_cost': cost,
                     'h':WeightedAStarAgent.h_msap(state,goal_state,p_cost, env), 'g': 0}
        dic_state['f'] = h_weight*dic_state['h'] + (1-h_weight)*dic_state['g']
        states.append(dic_state)        # List of nodes for different paths that were tested
        OPEN[state] = (dic_state['f'], state)    # Heapdict of open nodes to be open in the loop
        OPEN_full[state] = dic_state             # Dict of open nodes to be open in the loop (full)

        while len(OPEN) > 0:
            cur_snf = OPEN.popitem()                                # Getting the state with the minimum f
            cur = OPEN_full.pop(cur_snf[0])                         # Get its whole data
            CLOSE[cur['state']] = cur                               # Close it

            if self.env.is_final_state(cur['state']):                # Check whether the goal ahs been reached
                actions = cur['actions']
                total_cost = cur['total_cost']
                return actions, total_cost, expanded                 # Return the goal's data and path

            expanded += 1
            for action, successor in env.succ(cur['state']).items():  # Expand each state via its SUCCESSOR
                if successor[1] == np.inf or successor[0] in CLOSE:     # If a hole has approached, stay with the father
                    continue
                else:                                                       # If not a hole, define it as a state_dic and a path
                    dic_succ = {'state': successor[0], 'actions': cur['actions'] + [action],
                                'total_cost': cur['total_cost'] + successor[1]}
                    dic_succ['h'] = WeightedAStarAgent.h_msap(dic_succ['state'], goal_state, p_cost, env)
                    dic_succ['g'] = dic_succ['total_cost']
                    dic_succ['f'] = h_weight*dic_succ['h'] + (1-h_weight)*dic_succ['g']
                states.append(dic_succ)                                                 # Add the state_dic and data to the states
                if dic_succ['state'] not in CLOSE and dic_succ['state'] not in OPEN:
                    OPEN[dic_succ['state']] = (dic_succ['f'], dic_succ['state'])        # Add state_dic to open if not seen before
                    OPEN_full[dic_succ['state']] = dic_succ
                elif dic_succ['state'] in OPEN:
                    if dic_succ['f'] < OPEN[dic_succ['state']][0]:
                        OPEN[dic_succ['state']] = (dic_succ['f'], dic_succ['state'])    # Update state_dic to open if f is lower
                        OPEN_full[dic_succ['state']] = dic_succ
                else:
                    if dic_succ['f'] < CLOSE[dic_succ['state']]['f']:
                        OPEN[dic_succ['state']] = (dic_succ['f'], dic_succ['state'])    # Add state_dic to open if not seen before
                        OPEN_full[dic_succ['state']] = dic_succ                         # Do that in cases that node was closed already
                        CLOSE.pop(dic_succ['state'])                                    # get state_dic out of CLOSE if f is lower

        return 'NO SOLUTION HAS BEEN FOUND'


class IDAStarAgent():
    # def __init__(self) -> None:
    #     raise NotImplementedError

    def __init__(self):
        self.env = None

    def h_manhattan(state: int, goal: int, env):
        y_distance = np.abs(state // env.nrow - goal // env.nrow)
        x_distance = np.abs(state % env.ncol - goal % env.ncol)
        return y_distance + x_distance

    def h_msap(state: int, goals, acc_cost: int, env):
        # Checking if goals is single or many
        if type(goals) == int:
            # define as an iterable list
            goals = [goals]
        # Iterate and calculate manhattan distance
        h_man_goal = []
        for goal in goals:
            h_man_goal.append(WeightedAStarAgent.h_manhattan(state, goal, env))
        # Add the portal's cost
        h_man_goal.append(acc_cost)
        return min(h_man_goal)

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:

        self.env = env
        self.env.reset()
        global p_cost
        global new_limit
        global expand_iter
        global expand_tot
        p_cost = 100
        goal_state = env.goals
        state = self.env.get_initial_state()

        new_limit = IDAStarAgent.h_msap(state, goal_state, p_cost, env)
        expand_tot = 0
        while 1:
            expand_iter = 0

            f_limit = new_limit
            new_limit = np.inf
            result = IDAStarAgent.DFS_f(self.env.get_initial_state(), 0, [], f_limit, env, [])
            if result != False:
                # expand_iter
                # expand_tot
                return result[1], result[2], expand_iter
        return 'NO SOLUTION HAS BEEN FOUND'

    def DFS_f(state, g, path, f_limit, problem, actions):
        global p_cost
        global new_limit
        global expand_iter
        global expand_tot
        new_f = g + IDAStarAgent.h_msap(state, problem.goals, p_cost, problem)
        if new_f > f_limit:
            new_limit = min(new_limit, new_f)
            return False
        if problem.is_final_state(state):  # Check whether the goal ahs been reached
            return path, actions, g
        expand_iter += 1
        expand_tot += 1
        for action, successor in problem.succ(state).items():  # Expand each state via its SUCCESSOR
            if successor[1] == np.inf or successor[0] in path:  # If a hole has approached, stay with the father
                continue
            # The recursive DFS-f Function
            result = IDAStarAgent.DFS_f(successor[0], g+successor[1], path+[successor[0]], f_limit, problem, actions+[action])
            if result != False:
                return result
        return False
