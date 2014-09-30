'''
Created on Sep 23, 2014

TODO:

1. Modify the Queue class so that it defines:
    append and pop methods. Remove extend.
    DONE.

2. Modify the FIFOQueue so that it does not keep track of an index.
    DONE.

3. Create a LIFOQueue class.
    DONE.

4. Remove the update method.
    DONE.

5. See if we need the "some" method.
    DONE.

6. See if we need the if_ method.
    DONE.

7. For tree search and graph search, change the extend part.


'''

import sys, bisect


def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')

#______________________________________________________________________________
# Queues: FIFOQueue, LIFOQueue, PriorityQueue

class Queue:
    """Queue is an abstract class/interface. There are three types:        
        FIFOQueue(): A First In First Out Queue.
        LIFOQueue(): A Last In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue        
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    """
   
    def append(self, item):
        abstract()
    
    def pop(self):
        abstract()

class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""
    def __init__(self):
        self.A = []
    
    def append(self, item):
        """Add to the end"""
        self.A.append(item)

    def pop(self):
        """Remove the first item"""
        return self.A.pop(0)
        
    def __contains__(self, item):
        return item in self.A
    
    def __len__(self):
        return len(self.A)
    
    def __repr__(self):
        """Return [A[0], A[1], ...]"""
        rep = "[" + str(self.A[0])
        for i in range(1, len(self.A)):
            rep += ", " + str(self.A[i])
        rep += "]"
        return rep

class LIFOQueue(Queue):
    """A Last-In-First-Out Queue."""
    def __init__(self):
        self.A = []
    
    def append(self, item):
        """Add to the end"""
        self.A.append(item)

    def pop(self):
        """Remove the last item"""
        return self.A.pop()
        
    def __contains__(self, item):
        return item in self.A
    
    def __len__(self):
        return len(self.A)
    
    def __repr__(self):
        """Return [A[0], A[1], ...]"""
        rep = "[" + str(self.A[0])
        for i in range(1, len(self.A)):
            rep += ", " + str(self.A[i])
        rep += "]"
        return rep

class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min_first, the item with minimum f(x) is
    returned first; otherwise, the item with the maximum f(x) is returned.
    Also supports dict-like lookup.
    """
    def __init__(self, order="min_first", f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f
    
    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))
    
    def pop(self):
        if self.order == "min_first":
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]
        
    def __contains__(self, item):        
        for _, x in self.A:
            if item == x:
                return True                    
        return False
        
    
    def __len__(self):
        return len(self.A)
    
    def __repr__(self):
        """Return [A[0], A[1], ...]"""
        
        rep = "["
        
        if self.order == "min_first":
            rep = "[" + str(self.A[0][1]) + ":" + str(self.A[0][0]) 
            for i in range(1, len(self.A)):
                rep += ", " + str(self.A[i][1]) + ":" + str(self.A[i][0])
        else:
            rep = "[" + str(self.A[-1][1]) + ":" + str(self.A[-1][0])
            for i in range(len(self.A)-2, -1, -1):
                rep += ", " + str(self.A[i][1]) + ":" + str(self.A[i][0])        
        
        rep += "]"
        
        return rep
    
    # For dict-like operations
    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item
            
    def __delitem__(self, key):
        for i, (_, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)
                return

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        
        self.state=state
        self.parent=parent, 
        self.action=action,
        self.path_cost=path_cost
        
        self.depth=0        
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "Fig. 3.10"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state, action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

def tree_search(problem, frontier):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Don't worry about repeated paths to a state. [Fig. 3.7]"""
    frontier.append(Node(problem.initial))
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None

def graph_search(problem, frontier):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    If two paths reach a state, only use the first one. [Fig. 3.7]"""
    frontier.append(Node(problem.initial))
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored
                        and child not in frontier)
    return None

def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first."
    return tree_search(problem, FIFOQueue())

def depth_first_tree_search(problem):
    "Search the deepest nodes in the search tree first."
    return tree_search(problem, LIFOQueue())

def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first."
    return graph_search(problem, LIFOQueue())

def breadth_first_search(problem):
    "[Fig. 3.11]"
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = FIFOQueue()
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search."""

    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None

def uniform_cost_search(problem):
    "[Fig. 3.14]"
    return best_first_graph_search(problem, lambda node: node.path_cost)

def depth_limited_search(problem, limit=50):
    "[Fig. 3.17]"
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            
            if cutoff_occurred:
                return 'cutoff'
            else:
                return None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)

def iterative_deepening_search(problem):
    "[Fig. 3.18]"
    for depth in xrange(sys.maxint):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result

#______________________________________________________________________________
# Informed (Heuristic) Search

greedy_best_first_graph_search = best_first_graph_search
    # Greedy best-first search is accomplished by specifying f(n) = h(n).

def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = h or problem.h
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))
