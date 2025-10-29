from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from board_hex import BoardHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from seahorse.game.light_action import LightAction
import numpy as np
import heapq
import time

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
    
    def heuristic_parameters(self, state: GameState, nb_maillons: int):
        #TODO : update selon l'avancée de la game
        '''if nb_maillons > 7 and state.step > 8:
            return [state.step//2, 0.0, 0.0]
        elif nb_maillons > 7 and state.step <= 8:
            return [state.step//2, 0.0, 0.1]
        elif nb_maillons <= 7 and state.step <= 8:
            return [state.step//2, 0.5, 0.1]
        else:
            return [state.step//2, 0.5, 0.0]'''
        if nb_maillons > 7 and state.step > 8:
            return [1.0, 0.0, 0.0]
        elif nb_maillons > 7 and state.step <= 8:
            return [1.0, 0.0, 0.1]
        elif nb_maillons <= 7 and state.step <= 8:
            return [1.0, 0.5, 0.1]
        else:
            return [1.0, 0.5, 0.0]
    
    def halpha_beta_strat(self, currentState: GameState, heuristic, remaining_time: int, depth_allowed: int):
        
        start = time.time()
        time_to_play = remaining_time/(max(abs(13 - depth_allowed - currentState.step//2), 1e-4))
        
        # Identify if we are player1 or player2
        player1, player2 = currentState.players[0].id, currentState.players[1].id
        opponent_piece_type = currentState.next_player.get_piece_type()
        if opponent_piece_type == "R":
            id_me = player2
            id_opponent = player1
        else:
            id_me = player1
            id_opponent = player2
        
        def max_value(state: GameState, alpha, beta, depth):
            elapsed_time = time.time() - start
            if state.is_done():
                return (state.scores[id_me], state.scores[id_opponent])
            elif (not state.is_done()) and (depth == 0 or elapsed_time > time_to_play): # Si l'on a atteint la profondeur max à un état non terminal ou si le temps accordé pour jouer est épuisé
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
            elapsed_time = time.time() - start
            if state.is_done():
                return (state.scores[id_me], state.scores[id_opponent])
            elif (not state.is_done()) and (depth == 0 or elapsed_time > time_to_play): # Si l'on a atteint la profondeur max à un état non terminal
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
        return max_value(currentState, -math.inf, math.inf, depth_allowed)


    def heuristic_territory(self, state: GameState):
        # Pour toute pièce sous notre contrôle, on calcule le nombre de voisins libres = calcul du territoire
        turn_number = state.step
        current_rep = state.get_rep()
        territory = 0
        
        # Counter of empty cases in the current representation
        nb_empty_cases = 0
        for empty in current_rep.get_empty():
            nb_empty_cases += 1
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
                        influence = 1/turn_number*(1 + abs(nj - ni))
                        # computes the total influence of the occupied territory relative to the number of 
                        territory += influence/nb_empty_cases
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
                        influence = 1/turn_number*(1 + abs(nj - ni)) # The territory influence decreases in the long term 
                        territory += influence/nb_empty_cases # computes the total influence of the occupied territory
        #print(territory)
        return territory
    
    def heuristic_maillon(self, state: GameState):
        """
        Heuristique qui compte le nombre de maillons contrôlés par un joueur pour un état donné
        """
        current_rep = state.get_rep()
        env = current_rep.env
        
        nb_maillons = 0
        
        if self.piece_type == "R": #i.e. we are the first player to play
            # Get all the red cases already played
            nb_red_cases, red_cases = current_rep.get_pieces_player(state.players[0])
            for red_case in red_cases:
                i, j = red_case
                ## ------------- DETECTION DE MAILLONS INTERNES --------------------- ##
                if i > 1 and i < 12:
                    # Check au Sud Ouest si on a un maillon
                    if i+1 <= current_rep.dimensions[0] and j-2 >= 0*current_rep.dimensions[1] and env.get((i+1,j-2)) != None and env.get((i,j-1)) == None and env.get((i+1,j-1)) == None and env.get((i+1,j-2)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Nord Ouest si on a un maillon
                    elif i-1 >= 0*current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i-1,j-1)) != None and env.get((i,j-1)) == None and env.get((i-1,j)) == None and env.get((i-1,j-1)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Nord si on a un maillon
                    elif i-2 >= 0*current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i-2,j+1)) != None and env.get((i-1,j)) == None and env.get((i-1,j+1)) == None and env.get((i-2,j+1)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Nord Est si on a un maillon
                    elif i-1 >= 0*current_rep.dimensions[0] and j+2 <= current_rep.dimensions[1] and env.get((i-1,j+2)) != None and env.get((i,j+1)) == None and env.get((i-1,j+1)) == None and env.get((i-1,j+2)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Sud Est si on a un maillon
                    elif i+1 <= current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i+1,j+1)) != None and env.get((i,j+1)) == None and env.get((i+1,j)) == None and env.get((i+1,j+1)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Sud si on a un maillon
                    elif i+2 <= current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i+2,j-1)) != None and env.get((i+1,j-1)) == None and env.get((i+1,j)) == None and env.get((i+2,j-1)).piece_type == self.piece_type:
                        nb_maillons += 1
                        
                ## ------------- DETECTION DE MAILLONS AUX BORDS --------------------- ##
                else:
                    # Check si on a des maillons à la frontière nord
                    if i == 1:
                        if j >= 1 and j<= 12 and env.get((i-1,j)) == None and env.get((i-1,j+1)) == None:
                            nb_maillons += 1
                    # Check si on a des maillons à la frontière sud
                    elif i == 12:
                        if j >= 1 and j<= 12 and env.get((i+1,j)) == None and env.get((i+1,j-1)) == None:
                            nb_maillons += 1
        
        else: #the color is blue
            # Get all the red cases already played
            nb_blue_cases, blue_cases = current_rep.get_pieces_player(state.players[1])
            for blue_case in blue_cases:
                i, j = blue_case
                ## ------------- DETECTION DE MAILLONS INTERNES --------------------- ##
                if j > 1 and j < 12:
                    # Check au Sud Ouest si on a un maillon
                    if i+1 <= current_rep.dimensions[0] and j-2 >= 0*current_rep.dimensions[1] and env.get((i+1,j-2)) != None and env.get((i,j-1)) == None and env.get((i+1,j-1)) == None and env.get((i+1,j-2)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Nord Ouest si on a un maillon
                    elif i-1 >= 0*current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i-1,j-1)) != None and env.get((i,j-1)) == None and env.get((i-1,j)) == None and env.get((i-1,j-1)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Nord si on a un maillon
                    elif i-2 >= 0*current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i-2,j+1)) != None and env.get((i-1,j)) == None and env.get((i-1,j+1)) == None and env.get((i-2,j+1)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Nord Est si on a un maillon
                    elif i-1 >= 0*current_rep.dimensions[0] and j+2 <= current_rep.dimensions[1] and env.get((i-1,j+2)) != None and env.get((i,j+1)) == None and env.get((i-1,j+1)) == None and env.get((i-1,j+2)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Sud Est si on a un maillon
                    elif i+1 <= current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i+1,j+1)) != None and env.get((i,j+1)) == None and env.get((i+1,j)) == None and env.get((i+1,j+1)).piece_type == self.piece_type:
                        nb_maillons += 1
                    # Check au Sud si on a un maillon
                    elif i+2 <= current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i+2,j-1)) != None and env.get((i+1,j-1)) == None and env.get((i+1,j)) == None and env.get((i+2,j-1)).piece_type == self.piece_type:
                        nb_maillons += 1
                
                ## ------------- DETECTION DE MAILLONS AUX BORDS --------------------- ##
                else:
                    if j == 1:
                        if i >= 1 and i <= 12 and env.get((i,j-1)) == None and env.get((i+1,j-1)) == None:
                            nb_maillons += 1
                    # Check si on a des maillons à la frontière est
                    elif j == 12:
                        if i >= 1 and i <= 12 and env.get((i,j+1)) == None and env.get((i-1,j+1)) == None:
                            nb_maillons += 1
                
        return nb_maillons
    
    def shortest_path(self, state: GameState):
        """
        Function to implement the logic of the player (here greedy selection of a feasible solution).

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: Greedily selected feasible action
        """
        possible_actions = state.get_possible_light_actions()

        # Greedily find a shortest path connecting the 2 sides, and play closest to the center on it.
        env = state.rep.env 
        dist = np.full((state.rep.dimensions[0], state.rep.dimensions[1]), np.inf)
        preds = np.full((state.rep.dimensions[0], state.rep.dimensions[1]), None, dtype=tuple)
        objectives = []
        pq = []
        if self.piece_type == "R":
            for j in range(state.rep.dimensions[1]):
                objectives.append((state.rep.dimensions[0]-1, j))
                if env.get((0,j)) is None:
                    dist[0, j] = 1
                elif env.get((0,j)).piece_type == "R":
                    dist[0, j] = 0
                else:
                    continue
                heapq.heappush(pq, (dist[0, j], (0, j), None))

        else:
            for i in range(state.rep.dimensions[0]):
                objectives.append((i, state.rep.dimensions[1]-1))
                if env.get((i,0)) is None:
                    dist[i, 0] = 1
                elif env.get((i,0)).piece_type == "B":
                    dist[i, 0] = 0
                else:
                    continue
                heapq.heappush(pq, (dist[i, 0], (i, 0), None))

        while len(pq) != 0:
            d, (i, j), pred = heapq.heappop(pq)
            if d > dist[i, j]:
                continue
            preds[i,j] = pred
            if (i,j) in objectives:
                path = retrace_path(preds, (i,j))
                break
            for n_type, (ni, nj) in state.rep.get_neighbours(i, j).values():
                if n_type == "EMPTY":
                    new_dist = d + 1
                elif n_type == self.piece_type:
                    new_dist = d
                else:
                    continue
                if new_dist < dist[ni, nj]:
                    dist[ni, nj] = new_dist
                    heapq.heappush(pq, (new_dist, (ni, nj), (i, j)))
        if len(pq) > 0:
            dist = heapq.heappop(pq)[0]
            return dist
        else:
            return 0
    
    def heuristic_shortest_path(self, state: GameState):
        # Reconstruit le même état mais pour l'autre joueur (adversaire)
        '''current_rep = state.get_rep()
        env = current_rep.get_env()
        d = current_rep.get_dimensions()
        board = BoardHex(env=env, dim=d)
        state_opponent = GameStateHex(
            state.scores,
            state.compute_next_player(),
            state.players,
            board,
            step=state.step,
        )'''
        
        my_shortest_path = self.shortest_path(state)
        #opponent_shortest_path = self.shortest_path(state_opponent)
        return 13 - my_shortest_path # 13 est la longueur du chemin en ligne droite 
        
    
    def master_heuristic(self, state:GameState):
        nb_maillons = self.heuristic_maillon(state)
        params = self.heuristic_parameters(state, nb_maillons)
        pcc = self.heuristic_shortest_path(state)
        if params[0]*pcc - params[1]*nb_maillons > 3: # heuristique de PCC est trop forte
            params[0] = (2 + params[1]*nb_maillons)/pcc # ajuste le paramètre de l'heuristique pour repasser sous une valeur seuil
        elif params[0]*pcc - params[1]*nb_maillons < -3: # heuristique du nmobre de maillons est trop forte
            params[1] = (params[0]*pcc - 2)/nb_maillons # ajuste le paramètre de l'heuristique pour repasser sous une valeur seuil
        if params[2] == 0:
            return (params[0]*pcc + params[1]*nb_maillons)/10
        else:
            return (params[0]*pcc + params[1]*nb_maillons + params[2]*self.heuristic_territory(state))/10
            

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action possible.
        """
        possible_actions = current_state.get_possible_light_actions()
        current_rep = current_state.get_rep()
        env = current_rep.env
        print(current_state.step)
        
        # Opening moves
        if current_state.step < 2:
            if self.piece_type == "R":
                if current_state.step == 0: #premier tour à jouer pour les rouge (toujours possible de jouer)
                    opening = LightAction({"piece": self.piece_type, "position": (1, 2)})
                    return opening
            else:
                opening = LightAction({"piece": self.piece_type, "position": (4, 8)})
                if opening in possible_actions:
                    return opening
                else: # If our opening is already taken, try to counter it
                    return LightAction({"piece": self.piece_type, "position": (8, 4)})
        
        # First Answer moves
        elif current_state.step < 4:
            if self.piece_type == "R":
                if current_state.step == 2: #second tour à jouer pour les rouge (toujours possible de jouer)
                    nb_blue_cases, blue_cases = current_rep.get_pieces_player(current_state.players[1])
                    if blue_cases[0] == (4, 8):
                        first_answer = LightAction({"piece": self.piece_type, "position": (9, 8)})
                        return first_answer
                    else:
                        return LightAction({"piece": self.piece_type, "position": (4, 8)})
            else:
                nb_red_cases, red_cases = current_rep.get_pieces_player(current_state.players[0])
                if red_cases[-1] == (9, 8):
                    first_answer = LightAction({"piece": self.piece_type, "position": (8, 4)})
                    if first_answer in possible_actions:
                        return opening
                    elif LightAction({"piece": self.piece_type, "position": (4, 8)}) in possible_actions: # If our opening is already taken, try to counter it
                        return LightAction({"piece": self.piece_type, "position": (4, 8)})
                    else:
                        return LightAction({"piece": self.piece_type, "position": (3, 4)})
                else:
                    first_answer = LightAction({"piece": self.piece_type, "position": (10, 3)})
                    if first_answer in possible_actions:
                        return first_answer
        
        # General strategy : alpha-beta pruning with heuristic
        #(val, move) = self.halpha_beta_strat(current_state, heuristic=self.master_heuristic)
        """if val ==1 or val == -1: # Si le alpha-beta nous renvoie une situation de victoire ou défaite immédiate, jouer le move indiqué
            return move
        else:"""
        # Jouer IMMEDIATEMENT si un de nos maillons est menacé
        if self.piece_type == "R": #i.e. we are the first player to play
            # Get all the red cases already played
            nb_red_cases, red_cases = current_rep.get_pieces_player(current_state.players[0])
            piece_type_opponent = "B"
            for red_case in red_cases:
                i, j = red_case
                ## ------------- DETECTION DE MAILLONS INTERNES --------------------- ##
                if i > 1 and i < 12:
                    # Maillon Sud Ouest menacé (cas 1)
                    if i+1 <= current_rep.dimensions[0] and j-2 >= 0*current_rep.dimensions[1] and env.get((i+1,j-2)) != None and env.get((i,j-1)) != None and env.get((i+1,j-1)) == None and env.get((i,j-1)).piece_type == piece_type_opponent and env.get((i+1,j-2)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i+1, j-1)})
                    # Maillon Sud Ouest menacé (cas 2)
                    if i+1 <= current_rep.dimensions[0] and j-2 >= 0*current_rep.dimensions[1] and env.get((i+1,j-2)) != None and env.get((i,j-1)) == None and env.get((i+1,j-1)) != None and env.get((i+1,j-1)).piece_type == piece_type_opponent and env.get((i+1,j-2)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i, j-1)})
                    # Maillon Nord Ouest menacé (cas 1)
                    elif i-1 >= 0*current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i-1,j-1)) != None and env.get((i,j-1)) != None and env.get((i-1,j)) == None and env.get((i,j-1)).piece_type == piece_type_opponent and env.get((i-1,j-1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i-1, j)})
                    # Maillon Nord Ouest menacé (cas 2)
                    elif i-1 >= 0*current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i-1,j-1)) != None and env.get((i,j-1)) == None and env.get((i-1,j)) != None and env.get((i-1,j)).piece_type == piece_type_opponent and env.get((i-1,j-1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i, j-1)})
                    # Maillon Nord menacé (cas 1)
                    elif i-2 >= 0*current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i-2,j+1)) != None and env.get((i-1,j)) != None and env.get((i-1,j+1)) == None and env.get((i-1,j)).piece_type == piece_type_opponent and env.get((i-2,j+1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i-1, j+1)})
                    # Maillon Nord menacé (cas 2)
                    elif i-2 >= 0*current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i-2,j+1)) != None and env.get((i-1,j)) == None and env.get((i-1,j+1)) != None and env.get((i-1,j+1)).piece_type == piece_type_opponent and env.get((i-2,j+1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i-1, j)})
                    # Maillon Nord Est menacé (cas 1)
                    elif i-1 >= 0*current_rep.dimensions[0] and j+2 <= current_rep.dimensions[1] and env.get((i-1,j+2)) != None and env.get((i,j+1)) != None and env.get((i-1,j+1)) == None and env.get((i,j+1)).piece_type == piece_type_opponent and env.get((i-1,j+2)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i-1, j+1)})
                    # Maillon Nord Est menacé (cas 2)
                    elif i-1 >= 0*current_rep.dimensions[0] and j+2 <= current_rep.dimensions[1] and env.get((i-1,j+2)) != None and env.get((i,j+1)) == None and env.get((i-1,j+1)) != None and env.get((i-1,j+1)).piece_type == piece_type_opponent and env.get((i-1,j+2)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i, j+1)})
                    # Maillon Sud Est menacé (cas 1)
                    elif i+1 <= current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i+1,j+1)) != None and env.get((i,j+1)) != None and env.get((i+1,j)) == None and env.get((i,j+1)).piece_type == piece_type_opponent and env.get((i+1,j+1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i+1, j)})
                    # Maillon Sud Est menacé (cas 2)
                    elif i+1 <= current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i+1,j+1)) != None and env.get((i,j+1)) == None and env.get((i+1,j)) != None and env.get((i+1,j)).piece_type == piece_type_opponent and env.get((i+1,j+1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i, j+1)})
                    # Maillon Sud menacé (cas 1)
                    elif i+2 <= current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i+2,j-1)) != None and env.get((i+1,j)) != None and env.get((i+1,j-1)) == None and env.get((i+1,j)).piece_type == piece_type_opponent and env.get((i+2,j-1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i+1, j-1)})
                    # Maillon Sud menacé (cas 2)
                    elif i+2 <= current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i+2,j-1)) != None and env.get((i+1,j)) == None and env.get((i+1,j-1)) != None and env.get((i+1,j-1)).piece_type == piece_type_opponent and env.get((i+2,j-1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i+1, j)})
                
                ## ------------- DETECTION DE MAILLONS AUX BORDS --------------------- ##
                else:
                    # Check les maillons à la frontière nord
                    if i == 1:
                        # Maillon nord menacé (cas 1)
                        if j >= 1 and j<= 12 and env.get((i-1,j)) != None and env.get((i-1,j+1)) == None:
                            return LightAction({"piece": self.piece_type, "position": (i-1, j+1)})
                        # Maillon nord menacé (cas 2)
                        elif j >= 1 and j<= 12 and env.get((i-1,j)) == None and env.get((i-1,j+1)) != None:
                            return LightAction({"piece": self.piece_type, "position": (i-1, j)})
                    # Check les maillons à la frontière sud
                    elif i == 12:
                        # Maillon sud menacé (cas 1)
                        if j >= 1 and j<= 12 and env.get((i+1,j)) != None and env.get((i+1,j-1)) == None:
                            return LightAction({"piece": self.piece_type, "position": (i+1, j-1)})
                        # Maillon sud menacé (cas 2)
                        if j >= 1 and j<= 12 and env.get((i+1,j)) == None and env.get((i+1,j-1)) != None:
                            return LightAction({"piece": self.piece_type, "position": (i+1, j)})

        else: #the color is blue
            # Get all the red cases already played
            nb_blue_cases, blue_cases = current_rep.get_pieces_player(current_state.players[1])
            piece_type_opponent = "R"
            for blue_case in blue_cases:
                i, j = blue_case
                ## ------------- DETECTION DE MAILLONS INTERNES --------------------- ##
                if j > 1 and j < 12:
                    # Maillon Sud Ouest menacé (cas 1)
                    if i+1 <= current_rep.dimensions[0] and j-2 >= 0*current_rep.dimensions[1] and env.get((i+1,j-2)) != None and env.get((i,j-1)) != None and env.get((i+1,j-1)) == None and env.get((i,j-1)).piece_type == piece_type_opponent and env.get((i+1,j-2)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i+1, j-1)})
                    # Maillon Sud Ouest menacé (cas 2)
                    if i+1 <= current_rep.dimensions[0] and j-2 >= 0*current_rep.dimensions[1] and env.get((i+1,j-2)) != None and env.get((i,j-1)) == None and env.get((i+1,j-1)) != None and env.get((i+1,j-1)).piece_type == piece_type_opponent and env.get((i+1,j-2)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i, j-1)})
                    # Maillon Nord Ouest menacé (cas 1)
                    elif i-1 >= 0*current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i-1,j-1)) != None and env.get((i,j-1)) != None and env.get((i-1,j)) == None and env.get((i,j-1)).piece_type == piece_type_opponent and env.get((i-1,j-1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i-1, j)})
                    # Maillon Nord Ouest menacé (cas 2)
                    elif i-1 >= 0*current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i-1,j-1)) != None and env.get((i,j-1)) == None and env.get((i-1,j)) != None and env.get((i-1,j)).piece_type == piece_type_opponent and env.get((i-1,j-1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i, j-1)})
                    # Maillon Nord menacé (cas 1)
                    elif i-2 >= 0*current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i-2,j+1)) != None and env.get((i-1,j)) != None and env.get((i-1,j+1)) == None and env.get((i-1,j)).piece_type == piece_type_opponent and env.get((i-2,j+1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i-1, j+1)})
                    # Maillon Nord menacé (cas 2)
                    elif i-2 >= 0*current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i-2,j+1)) != None and env.get((i-1,j)) == None and env.get((i-1,j+1)) != None and env.get((i-1,j+1)).piece_type == piece_type_opponent and env.get((i-2,j+1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i-1, j)})
                    # Maillon Nord Est menacé (cas 1)
                    elif i-1 >= 0*current_rep.dimensions[0] and j+2 <= current_rep.dimensions[1] and env.get((i-1,j+2)) != None and env.get((i,j+1)) != None and env.get((i-1,j+1)) == None and env.get((i,j+1)).piece_type == piece_type_opponent and env.get((i-1,j+2)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i-1, j+1)})
                    # Maillon Nord Est menacé (cas 2)
                    elif i-1 >= 0*current_rep.dimensions[0] and j+2 <= current_rep.dimensions[1] and env.get((i-1,j+2)) != None and env.get((i,j+1)) == None and env.get((i-1,j+1)) != None and env.get((i-1,j+1)).piece_type == piece_type_opponent and env.get((i-1,j+2)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i, j+1)})
                    # Maillon Sud Est menacé (cas 1)
                    elif i+1 <= current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i+1,j+1)) != None and env.get((i,j+1)) != None and env.get((i+1,j)) == None and env.get((i,j+1)).piece_type == piece_type_opponent and env.get((i+1,j+1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i+1, j)})
                    # Maillon Sud Est menacé (cas 2)
                    elif i+1 <= current_rep.dimensions[0] and j+1 <= current_rep.dimensions[1] and env.get((i+1,j+1)) != None and env.get((i,j+1)) == None and env.get((i+1,j)) != None and env.get((i+1,j)).piece_type == piece_type_opponent and env.get((i+1,j+1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i, j+1)})
                    # Maillon Sud menacé (cas 1)
                    elif i+2 <= current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i+2,j-1)) != None and env.get((i+1,j)) != None and env.get((i+1,j-1)) == None and env.get((i+1,j)).piece_type == piece_type_opponent and env.get((i+2,j-1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i+1, j-1)})
                    # Maillon Sud menacé (cas 2)
                    elif i+2 <= current_rep.dimensions[0] and j-1 >= 0*current_rep.dimensions[1] and env.get((i+2,j-1)) != None and env.get((i+1,j)) == None and env.get((i+1,j-1)) != None and env.get((i+1,j-1)).piece_type == piece_type_opponent and env.get((i+2,j-1)).piece_type == self.piece_type:
                        return LightAction({"piece": self.piece_type, "position": (i+1, j)})
                
                ## ------------- DETECTION DE MAILLONS AUX BORDS --------------------- ##
                else:
                    # Check les maillons à la frontière ouest
                    if j == 1:
                        # Maillon ouest menacé (cas 1)
                        if i >= 1 and i <= 12 and env.get((i,j-1)) != None and env.get((i+1,j-1)) == None:
                            return LightAction({"piece": self.piece_type, "position": (i+1, j-1)})
                        # Maillon ouest menacé (cas 2)
                        elif i >= 1 and i <= 12 and env.get((i,j-1)) == None and env.get((i+1,j-1)) != None:
                            return LightAction({"piece": self.piece_type, "position": (i, j-1)})
                    # Check les maillons à la frontière est
                    elif j == 12:
                        # Maillon est menacé (cas 1)
                        if i >= 1 and i <= 12 and env.get((i,j+1)) != None and env.get((i-1,j+1)) == None:
                            return LightAction({"piece": self.piece_type, "position": (i-1, j+1)})
                        # Maillon est menacé (cas 2)
                        if i >= 1 and i <= 12 and env.get((i,j+1)) == None and env.get((i-1,j+1)) != None:
                            return LightAction({"piece": self.piece_type, "position": (i, j+1)})
        
        # Si aucun maillon n'est menacé, on prend le move donné par alpha-beta
        print("-------------------------------------")
        print("------ AUCUN MAILLON DE MENACE ------")
        print("-------------------------------------")
        (val, move) = self.halpha_beta_strat(current_state, self.master_heuristic, remaining_time, 2)
        return move


def retrace_path(preds, end):
    """
    Recreate the path from the start to the end position using the predecessors.
    """
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = preds[current]
    return path