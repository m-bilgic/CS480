'''
Created on Sep 23, 2014

@author: mbilgic
'''

from problem import NQueensProblem
from search import depth_first_tree_search, breadth_first_tree_search

if __name__ == '__main__':
    nq = NQueensProblem(4)
    #df = depth_first_tree_search(nq, interactive=True)
    #print df
    bf = breadth_first_tree_search(nq, interactive=True)
    #print bf.path()
    
