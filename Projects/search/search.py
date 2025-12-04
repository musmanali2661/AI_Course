# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # --- 1. Initialization ---

    # Use a set for O(1) average time complexity lookup of visited states.
    # Stores states that have already been expanded (successors generated).
    expanded_states = set()

    # The stack will store tuples: (state, path_to_state_as_list_of_directions)
    dfs_stack = util.Stack()

    # Initial state and an empty path
    start_state = problem.getStartState()
    dfs_stack.push((start_state, []))

    # --- 2. Main Search Loop ---

    while not dfs_stack.isEmpty():

        # Pop the current state and its accumulated path
        current_state, path_so_far = dfs_stack.pop()

        # Check for Goal (should be done immediately upon popping a node)
        if problem.isGoalState(current_state):
            return path_so_far

        # Graph Search Check: Skip if we've already expanded this state
        if current_state in expanded_states:
            continue

        # Mark the current state as expanded
        expanded_states.add(current_state)
        # nodes_expanded += 1 # Uncomment if you need to track this metric

        # Get successors: (successor_state, direction, cost)
        for successor_state, direction, _ in problem.getSuccessors(current_state):

            # 1. Create a NEW path by copying the old path
            new_path = path_so_far + [direction]  # Using list concatenation to create a new list

            # 2. Check visited status of the successor state (Optimization)
            if successor_state not in expanded_states:
                # Push the successor state and the NEW path onto the stack
                dfs_stack.push((successor_state, new_path))

    # If the stack is empty and the goal hasn't been found
    return []
    #util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    # --- 1. Initialization ---

    # Use a set for O(1) average time complexity lookup of visited states.
    # Stores states that have already been expanded (successors generated).
    expanded_states = set()

    # The stack will store tuples: (state, path_to_state_as_list_of_directions)
    dfs_stack = util.Queue()

    # Initial state and an empty path
    start_state = problem.getStartState()
    dfs_stack.push((start_state, []))

    # --- 2. Main Search Loop ---

    while not dfs_stack.isEmpty():

        # Pop the current state and its accumulated path
        current_state, path_so_far = dfs_stack.pop()

        # Check for Goal (should be done immediately upon popping a node)
        if problem.isGoalState(current_state):
            return path_so_far

        # Graph Search Check: Skip if we've already expanded this state
        if current_state in expanded_states:
            print("i am stuck here in continue")
            continue

        # Mark the current state as expanded
        expanded_states.add(current_state)
        # nodes_expanded += 1 # Uncomment if you need to track this metric

        # Get successors: (successor_state, direction, cost)
        for successor_state, direction, _ in problem.getSuccessors(current_state):

            # 1. Create a NEW path by copying the old path
            new_path = path_so_far + [direction]  # Using list concatenation to create a new list

            # 2. Check visited status of the successor state (Optimization)
            if successor_state not in expanded_states:
                # Push the successor state and the NEW path onto the stack
                dfs_stack.push((successor_state, new_path))
            print("i am stuck here in for loop")

    # If the stack is empty and the goal hasn't been found
    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    """Search the shallowest nodes in the search tree first."""
    # --- 1. Initialization ---

    # Use a set for O(1) average time complexity lookup of visited states.
    # Stores states that have already been expanded (successors generated).
    expanded_states = set()

    # The stack will store tuples: (state, path_to_state_as_list_of_directions)
    dfs_stack = util.PriorityQueue()

    # Initial state and an empty path
    start_state = problem.getStartState()
    dfs_stack.push((start_state, []),util.manhattanDistance(start_state,))

    # --- 2. Main Search Loop ---

    while not dfs_stack.isEmpty():

        # Pop the current state and its accumulated path
        current_state, path_so_far = dfs_stack.pop()

        # Check for Goal (should be done immediately upon popping a node)
        if problem.isGoalState(current_state):
            return path_so_far

        # Graph Search Check: Skip if we've already expanded this state
        if current_state in expanded_states:
            print("i am stuck here in continue")
            continue

        # Mark the current state as expanded
        expanded_states.add(current_state)
        # nodes_expanded += 1 # Uncomment if you need to track this metric

        # Get successors: (successor_state, direction, cost)
        for successor_state, direction, _ in problem.getSuccessors(current_state):

            # 1. Create a NEW path by copying the old path
            new_path = path_so_far + [direction]  # Using list concatenation to create a new list

            # 2. Check visited status of the successor state (Optimization)
            if successor_state not in expanded_states:
                # Push the successor state and the NEW path onto the stack
                dfs_stack.push((successor_state, new_path))
            print("i am stuck here in for loop")

    # If the stack is empty and the goal hasn't been found
    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
