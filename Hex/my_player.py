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
                return (state.scores[state.players[1].id], state.scores[state.players[0].id])
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
                return (state.scores[state.players[1].id], state.scores[state.players[0].id])
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

    def heuristic_territory(self, state: GameState):
        # Pour toute pièce sous notre contrôle, on calcule le nombre de voisins libres = calcul du territoire
        current_rep = state.get_rep()
        territory = 0
        if self.piece_type == "R": #i.e. we are the first player to play
            # Get all the red cases already played
            nb_red_cases, red_cases = current_rep.get_pieces_player(state.players[0])
            visited_empty_tiles = set()
            for red_case in red_cases:
                # Get neighbours list for each red tile
                neighbours = state.get_neighbours(red_case[0], red_case[1])
                for n_type, (ni, nj) in neighbours.values():
                    if n_type == "EMPTY" and (ni, nj) not in visited_empty_tiles:
                        visited_empty_tiles.add((ni, nj))
                        territory += 1
        else:
            # Get all the blue cases already played
            nb_blue_cases, blue_cases = current_rep.get_pieces_player(state.players[1])
            visited_empty_tiles = set()
            for blue_case in blue_cases:
                # Get neighbours list for each blue tile
                neighbours = state.get_neighbours(blue_case[0], blue_case[1])
                for n_type, (ni, nj) in neighbours.values():
                    if n_type == "EMPTY" and (ni, nj) not in visited_empty_tiles:
                        visited_empty_tiles.add((ni, nj))
                        territory += 1
        #print(territory)
        return territory

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action possible.
        """
        return self.halpha_beta_strat(current_state, heuristic=self.heuristic_territory)


        
        