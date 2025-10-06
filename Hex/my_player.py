from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import math as math

class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)
    
    def halpha_beta_strat(self, currentState: GameState, heuristic):
        
        def max_value(state: GameState, alpha, beta, depth):
            if state.is_done():
                return (state.scores[state.players[0].id], state.scores[state.players[1].id])
            elif (not state.is_done()) and depth == 0: # Si l'on a atteint la profondeur max à un état non terminal
                return (heuristic(state), -heuristic(state)) # Jeu à somme nulle : le score d'un joueur est l'opposé de l'autre
            else:
                v_star = -1.0*math.inf
                m_star = "osef"
                for a in state.generate_possible_light_actions():
                    s_next = state.apply_action(a)
                    v, _ = min_value(s_next, alpha, beta, depth-1)
                    if v > v_star:
                        v_star = v
                        m_star = a
                        alpha = max(alpha, v_star)
                        if v_star >= beta:
                            return (v_star, m_star)
                return (v_star, m_star)

        def min_value(state: GameState, alpha, beta, depth):
            if state.is_done():
                return (state.scores[state.players[0].id], state.scores[state.players[1].id])
            elif (not state.is_done()) and depth == 0: # Si l'on a atteint la profondeur max à un état non terminal
                return (heuristic(state), -heuristic(state)) # Jeu à somme nulle : le score d'un joueur est l'opposé de l'autre
            else:
                v_star = math.inf
                m_star = "osef"
                for a in state.generate_possible_light_actions():
                    s_next = state.apply_action(a)
                    v, _ = max_value(s_next, alpha, beta, depth-1)
                    if v < v_star:
                        v_star = v
                        m_star = a
                        beta = min(beta, v_star)
                        if v_star <= alpha:
                            return (v_star, m_star)
                return (v_star, m_star)        
        return max_value(currentState, -math.inf, math.inf, 3)[1]

    def naive_heuristic(self, state: GameState):
        return 0 if not state.is_done() else state.get_player_score(self)

    def heuristic_occupy_space(self, state: GameState):
        # Pour toute pièce sous notre contrôle, on calcule le nombre de voisins libres
        current_rep = state.get_rep()
        current_env = current_rep.get_env()
        return 0

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        return self.halpha_beta_strat(current_state, heuristic=self.heuristic_occupy_space)


        
        