import random
import numpy as np
from ast import literal_eval

infinity = np.Inf


def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')

def todo():
    raise NotImplementedError('You must complete the implementation.')

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

#______________________________________________________________________________
# Minimax Search

def minimax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Fig. 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minimax_decision:
    return argmax(game.actions(state),
                  lambda a: min_value(game.result(state, a)))

#______________________________________________________________________________

def alphabeta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Fig. 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)

    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_search:
    return argmax(game.actions(state),
                  lambda a: min_value(game.result(state, a),
                                      -infinity, infinity))

#______________________________________________________________________________
# Players for Games

def query_tictactoe_player(game, state):
    "Make a move by querying standard input."
    game.display(state)
    move = raw_input('Your move? Enter as a tuple (x,y)')
    return literal_eval(move)
    
def random_player(game, state):
    "A player that chooses a legal move at random."
    return random.choice(game.actions(state))

def alphabeta_player(game, state):
    return alphabeta_search(state, game)

def play_game(game, *players):
    """Play an n-person, move-alternating game.
    """
    state = game.initial
    while True:
        for player in players:
            move = player(game, state)
            state = game.result(state, move)
            if game.terminal_test(state):
                return game.utility(state, game.to_move(game.initial))

#______________________________________________________________________________
# Games

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        "Return a list of the allowable moves at this point."
        abstract()

    def result(self, state, move):
        "Return the state that results from making a move from a state."
        abstract()

    def utility(self, state, player):
        "Return the value of this final state to player."
        abstract()

    def terminal_test(self, state):
        "Return True if this is a final state for the game."
        return not self.actions(state)

    def to_move(self, state):
        "Return the player whose move it is in this state."
        return state.to_move

    def display(self, state):
        "Print or otherwise display the state."
        print state

    def __repr__(self):
        return '<%s>' % self.__class__.__name__


class TicTacToe(Game):
    """Play TicTacToe on an 3 x 3 board, with Max (first player) playing 'X'.
    A move is a tuple (x,y).
    Each game state needs to know who's turn it is.
    """
    def __init__(self):
        "Create an empty TicTacToe board and assign it to the self.initial variable."
        todo()

    def actions(self, state):
        "Legal moves are any square not yet taken. Return as a list of tuples."
        todo()

    def result(self, state, move):
        """"Return the state that would result if the move is taken.
        If the move is illegal, ignore it.
        """ 
        todo()

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        todo()

    def terminal_test(self, state):
        "A state is terminal if it is won or there are no empty squares."
        todo()

    def display(self, state):
        "Display the state in a human-readable format."
        todo()

    def compute_utility(self, board, move, player):
        "If X wins with this move, return 1; if O wins return -1; else return 0."
        todo()

if __name__ == '__main__':
    
    # A random player playing against a random player
    play_game(TicTacToe(), random_player, random_player)
    
    # A random player playing against an alpha-beta player
    play_game(TicTacToe(), random_player, alphabeta_player)
    
    # A random player playing against a human player
    play_game(TicTacToe(), random_player, query_tictactoe_player)
    
    # An alpha-beta player playing against a human player
    play_game(TicTacToe(), alphabeta_player, query_tictactoe_player)
    
