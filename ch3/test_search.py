'''
Created on Sep 23, 2014

@author: mbilgic
'''

from problem import NQueensProblem, EightPuzzleProblem, misplaced_tiles_heuristic
from search import depth_first_tree_search, breadth_first_tree_search, best_first_graph_search

if __name__ == '__main__':
    #nq = NQueensProblem(4)
    #df = depth_first_tree_search(nq, interactive=True)
    #print df
    #bf = breadth_first_tree_search(nq, interactive=True)
    #print bf.path()
    ep = EightPuzzleProblem([[1, 2, 5], [3, 4, 0], [6, 7, 8]])
    #depth_first_tree_search(ep, interactive=True)
    #breadth_first_tree_search(ep, interactive=True)
    bfgs = best_first_graph_search(ep, misplaced_tiles_heuristic)
    print bfgs.path()
    
