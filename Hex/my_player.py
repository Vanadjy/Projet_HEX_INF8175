# -*- coding: utf-8 -*-
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
#import networkx as nx
#from operator import itemgetter

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
    
    def heuristic_parameters(self, state: GameState, nb_maillons: int, pcc):
        #TODO : update selon l'avancée de la game
        '''if nb_maillons > 8 and state.step > 8:
            return [1.0, 0.0, 0.0]
        elif nb_maillons > 7 and state.step <= 8:
            return [1.0, 0.0, 0.1]
        elif nb_maillons <= 7 and state.step <= 8:
            return [1.0, 0.5, 0.1]
        else:
            return [1.0, 0.5, 0.0]'''
        weight = (state.step+1)/(65)
        return [weight, 0.5*(1-weight), 0.1*(1 - weight), (1 - weight)]
    
    def bridges_creation(self):
        possible_inner_bridges = [(+1, -2), (-1, -1), (-2, +1), (-1, +2), (+1, +1), (+2, -1)]
        pairs_completing_bridges = [((0, -1),(+1, -1)), ((0, -1), (-1, 0)), ((-1, 0), (-1, +1)), ((0, +1), (-1, +1)), ((0, +1), (+1, 0)), ((+1, 0), (+1, -1))]
        return possible_inner_bridges, pairs_completing_bridges
    
    def halpha_beta_strategy(self, currentState: GameState, heuristic, remaining_time: int, depth_allowed: int):
    
        start = time.time()
        time_to_play = remaining_time/(max(abs(13 - depth_allowed - currentState.step//2), 1e-4))
        
        if self.piece_type == "R":
            id_my_player = currentState.players[0].id
            id_opponent = currentState.players[1].id
        else:
            id_my_player = currentState.players[1].id
            id_opponent = currentState.players[0].id
        
        def max_value(state: GameState, alpha, beta, depth):
            elapsed_time = time.time() - start
            if state.is_done():
                score1, score2 = 10000*(state.get_player_score(self) - state.get_player_score(state.compute_next_player())), None
                return 10000*(state.scores[id_my_player] - state.scores[id_opponent]), None
            elif (not state.is_done()) and (depth == 0 or elapsed_time > time_to_play): # Si l'on a atteint la profondeur max à un état non terminal ou si le temps accordé pour jouer est épuisé
                return heuristic(state), None
            v_star = - math.inf
            m_star = None
            for a in state.get_possible_heavy_actions():
                next_state = a.get_next_game_state()
                v, _ = min_value(next_state, alpha, beta, depth-1)
                if v > v_star:
                    v_star = v
                    m_star = a
                    alpha = max(alpha, v_star)
                if v_star >= beta:
                    return v_star, m_star
                
            return (v_star, m_star)
        
        def min_value(state: GameState, alpha, beta, depth):
            elapsed_time = time.time() - start
            if state.is_done():
                score1, score2 = 10000*(state.get_player_score(self) - state.get_player_score(state.compute_next_player())), None
                return 10000*(state.scores[id_my_player] - state.scores[id_opponent]), None
            elif (not state.is_done()) and (depth == 0 or elapsed_time > time_to_play): # Si l'on a atteint la profondeur max à un état non terminal ou si le temps accordé pour jouer est épuisé
                return heuristic(state), None
            v_star = math.inf
            m_star = None
            
            for a in state.get_possible_heavy_actions():
                next_state = a.get_next_game_state()
                v, _ = max_value(next_state, alpha, beta, depth-1)
                if v < v_star:
                    v_star = v
                    m_star = a
                    beta = min(beta, v_star)
                if v_star <= alpha:
                    return v_star, m_star
                
            return (v_star, m_star)
        
        return max_value(currentState, -math.inf, math.inf, depth_allowed)[1]

    def heuristic_opponent_influence(self, state: GameState):
        turn_number = state.step
        current_rep = state.get_rep()
        territory = 0
        
        if self.piece_type == "B": #i.e. we are the first player to play
            # Get all the red cases already played
            nb_red_cases, red_cases = current_rep.get_pieces_player(state.players[0])
            visited_empty_tiles = set()
            for red_case in red_cases:
                # Get neighbours list for each red tile
                neighbours = state.get_neighbours(red_case[0], red_case[1])
                for n_type, (ni, nj) in neighbours.values():
                    if n_type == "EMPTY" and (ni, nj) not in visited_empty_tiles:
                        visited_empty_tiles.add((ni, nj))
                        influence = 1/(abs(6.5 - nj) + abs(6.5 - ni))                        # computes the total influence of the occupied territory relative to the number of 
                        territory += influence
        else:
            # Get all the blue cases already played
            nb_blue_cases, blue_cases = current_rep.get_pieces_player(state.players[0])
            visited_empty_tiles = set()
            for blue_case in blue_cases:
                # Get neighbours list for each blue tile
                neighbours = state.get_neighbours(blue_case[0], blue_case[1])
                for n_type, (ni, nj) in neighbours.values():
                    if n_type == "EMPTY" and (ni, nj) not in visited_empty_tiles:
                        visited_empty_tiles.add((ni, nj))
                        influence = 1/(abs(6.5 - nj) + abs(6.5 - ni))                        
                        territory += influence # computes the total influence of the occupied territory
        return territory

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
                        influence = 1/(abs(6.5 - nj)^2 + abs(6.5 - ni)^2)
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
                        influence = 1/(abs(6.5 - nj) + abs(6.5 - ni))
                        territory += influence/nb_empty_cases # computes the total influence of the occupied territory
        #print(territory)
        return territory
    
    def count_maillons(self, state: GameState):
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
    
    def heuristic_maillon(self, state: GameState):
        current_rep = state.get_rep()
        env = current_rep.get_env()
        d = current_rep.get_dimensions()
        board = BoardHex(env=env, dim=d)
        state_opponent = GameStateHex(
            state.scores,
            state.compute_next_player(),
            state.players,
            board,
            step=state.step,
        )
        nb_maillons_me = self.count_maillons(state)
        #nb_maillons_opponent = self.count_maillons(state_opponent)
        return nb_maillons_me
    
    def control_centre(self, state: GameState):
        # Invite the player to play near the centre of the board
        current_rep = state.get_rep()
        territory = 0
        if self.piece_type == "R":
            for empty in current_rep.get_empty():
                ni, nj = empty
                centre_influence = 1/(abs(6.5 - nj) + abs(6.5 - ni))
                if territory < centre_influence:
                    territory = centre_influence
            return territory
        else:
            for empty in current_rep.get_empty():
                ni, nj = empty
                centre_influence = 1/(abs(6.5 - nj) + abs(6.5 - ni))
                if territory < centre_influence:
                    territory = centre_influence
            return territory
    
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
        
        possible_inner_bridges, pairs_completing_bridges = self.bridges_creation()
        
        if self.piece_type == "R":
            for j in range(state.rep.dimensions[1]):
                objectives.append((state.rep.dimensions[0]-1, j))
                if env.get((0,j)) is None:
                    dist[0, j] = 1
                elif env.get((0,j)).piece_type == "R":
                    dist[0, j] = 0
                else:
                    ## DETECTION MAILLONS INTERNES ##
                    for index in range(len(possible_inner_bridges)):
                        dx_bridge, dy_bridge = possible_inner_bridges[index]
                        # Si on détecte un maillon
                        if state.in_board((dx_bridge, j + dy_bridge)) and (env.get((dx_bridge, j + dy_bridge)) is not None) and (env.get((dx_bridge, j + dy_bridge)).piece_type == 'B'):
                            complete_bridge_1, complete_bridge_2 = pairs_completing_bridges[index]
                            dx_b1, dy_b1 = complete_bridge_1
                            dx_b2, dy_b2 = complete_bridge_2
                            # Confirmation que nous avons un maillon
                            if (env.get((dx_b1, j+dy_b1)) is None) and (env.get((dx_b2, j+dy_b2)) is None):
                                # Forcer à passer ailleurs, même si la case est vide
                                dist[dx_b1, j+dy_b1] = np.inf
                                dist[dx_b2, j+dy_b2] = np.inf
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
                    ## DETECTION MAILLONS INTERNES ##
                    for index in range(len(possible_inner_bridges)):
                        dx_bridge, dy_bridge = possible_inner_bridges[index]
                        # Si on détecte un maillon
                        if state.in_board((i + dx_bridge, dy_bridge)) and (env.get((i + dx_bridge, dy_bridge)) is not None) and (env.get((i + dx_bridge, dy_bridge)).piece_type == 'R'):
                            complete_bridge_1, complete_bridge_2 = pairs_completing_bridges[index]
                            dx_b1, dy_b1 = complete_bridge_1
                            dx_b2, dy_b2 = complete_bridge_2
                            # Confirmation que nous avons un maillon
                            if (env.get((i + dx_b1, dy_b1)) is None) and (env.get((i + dx_b2, dy_b2)) is None):
                                # Forcer à passer ailleurs, même si la case est vide
                                dist[i + dx_b1, dy_b1] = np.inf
                                dist[i + dx_b2, dy_b2] = np.inf
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
        current_rep = state.get_rep()
        env = current_rep.get_env()
        
        my_shortest_path = self.shortest_path(state)
        #opponent_shortest_path = self.shortest_path(state_opponent)
        return 13 - my_shortest_path #+ 2*opponent_shortest_path # 13 est la longueur du chemin en ligne droite 

    def shortest_path_graph(self, state:GameState, my_piece_type, opponent_piece_type):
        assert (my_piece_type != opponent_piece_type), "Piece type error : each player must have a different piece type"
        assert (my_piece_type == 'R' or my_piece_type == 'B'), "Piece type error : a player must be blue ('B') or red ('R')"
        # création d'une grille de jeu
        current_rep = state.get_rep()
        env = current_rep.get_env()
        N = state.get_rep().get_dimensions()[0]

        nb_blue_cases, blue_cases = current_rep.get_pieces_player(state.players[1])
        nb_red_cases, red_cases = current_rep.get_pieces_player(state.players[0])
        grid = {(q, r): 'EMPTY' for r in range(N) for q in range(N)}
        for bx, by in blue_cases:
            grid[bx, by] = 'B'
        for rx, ry in red_cases:
            grid[rx, ry] = 'R'
        
        Graph_board  = build_hex_graph(state)
        
        if my_piece_type == 'R':
            max_red = max(red_cases, key=itemgetter(1))[1]
            min_red = min(red_cases, key=itemgetter(1))[1]
            sources = [(0,j) for j in range(min_red, max_red+1)]
            targets = [(13,j) for j in range(min_red, max_red+1)]
        else:
            max_blue = max(blue_cases, key=itemgetter(0))[0]
            min_blue = min(blue_cases, key=itemgetter(0))[0]
            sources = [(j, 0) for j in range(min_blue, max_blue+1)]
            targets = [(j, 13) for j in range(min_blue, max_blue+1)]
        
        def edge_cost(node_1, node_2):
            if grid[node_1] == my_piece_type or grid[node_2] == my_piece_type:
                if grid[node_1] == my_piece_type and grid[node_2] == my_piece_type:
                    return 0
                else:
                    return 1
            elif grid[node_1] == opponent_piece_type or grid[node_2] == opponent_piece_type:
                return math.inf
            else:
                return 2
        
        def bridge_edge_cost(G, node_1, node_2):
            neighbors_node1 = set(Graph_board.neighbors(node_1))
            neighbors_node2 = set(Graph_board.neighbors(node_2))
            # get neighbors of node2 that are not neighbors of node1
            diff_neighbors = neighbors_node2 - neighbors_node1
            for possible_bridge in diff_neighbors:
                # Règle de détection d'un maillon
                if grid[node_1] == grid[possible_bridge] and grid[node_1] == opponent_piece_type:
                    neighbors_possible_bridge = set(Graph_board.neighbors(possible_bridge))
                    common_neighbors = list(neighbors_possible_bridge & neighbors_node1)
                    #check that node1 and node2 are not neighbors and of the type of the opponent
                    if len(common_neighbors) == 2 and grid[common_neighbors[0]] == 'EMPTY' and grid[common_neighbors[1]] == 'EMPTY':
                        for final_node in neighbors_node2:
                            G[node_2][final_node]["weight"] = math.inf
                    
        for u, v in Graph_board.edges():
            Graph_board[u][v]["weight"] = edge_cost(u, v)
            bridge_edge_cost(Graph_board, u, v)
                
        
        current_sp = None
        sp_value = math.inf
        for source in sources:
            for target in targets:
                sp = nx.shortest_path(Graph_board, source, target, weight="weight")
                if len(sp) < sp_value:
                    sp_value = len(sp)
                    current_sp = sp
        for sp_element in current_sp:
            if env.get(sp_element) is not None:
                current_sp.remove(sp_element)
        return current_sp, sp_value
    
    def heuristic_block_avance(self, state):
        block = 0
        current_rep = state.get_rep()
        if self.piece_type == 'R':
            nb_blue_cases, blue_cases = current_rep.get_pieces_player(state.players[1])
            opp_shortest_path, opp_shortest_path_len = self.shortest_path_graph(state, 'B', self.piece_type)
            for blue_case in blue_cases:
                i, j = blue_case
                neighbours_positions = state.get_neighbours(i,j).values()
                # Regarder les voisins des voisins
                for n_type, (ni, nj) in neighbours_positions:
                    if n_type == "EMPTY": # on cherche à se placer à une case vide des cases adverses
                        neighbours_of_neighbor = state.get_neighbours(ni, nj)
                        for m_type, (mi, mj) in neighbours_of_neighbor.values():
                            if (mi, mj) in opp_shortest_path:
                                block += 1
        else:
            nb_red_cases, red_cases = current_rep.get_pieces_player(state.players[0])
            opp_shortest_path, opp_shortest_path_len = self.shortest_path_graph(state, 'R', self.piece_type)
            for red_case in red_cases:
                i, j = red_case
                neighbours_positions = state.get_neighbours(i,j).values()
                # Regarder les voisins des voisins
                for n_type, (ni, nj) in neighbours_positions:
                    if n_type == "EMPTY": # on cherche à se placer à une case vide des cases adverses
                        neighbours_of_neighbor = state.get_neighbours(ni, nj)
                        for m_type, (mi, mj) in neighbours_of_neighbor.values():
                            # Si voisin d'ordre 2 et de notre couleur : forme de blocage
                            if (mi, mj) in opp_shortest_path:
                                block += 1
        return block
    
    def master_heuristic(self, state:GameState):
        '''nb_maillons = self.heuristic_maillon(state)
        params = self.heuristic_parameters(state, nb_maillons)
        pcc = self.heuristic_shortest_path(state)
        if params[0]*pcc - params[1]*nb_maillons > 3: # heuristique de PCC est trop forte
            params[0] = (2 + params[1]*nb_maillons)/pcc # ajuste le paramètre de l'heuristique pour repasser sous une valeur seuil
        elif params[0]*pcc - params[1]*nb_maillons < -3: # heuristique du nmobre de maillons est trop forte
            params[1] = (params[0]*pcc - 2)/nb_maillons # ajuste le paramètre de l'heuristique pour repasser sous une valeur seuil
        if params[2] == 0:
            return (params[0]*pcc + params[1]*nb_maillons)/10
        else:'''
        nb_maillons = self.heuristic_maillon(state)
        pcc = self.heuristic_shortest_path(state)
        params = self.heuristic_parameters(state, nb_maillons, pcc)
        #opp_influence = self.heuristic_opponent_influence(state)
        #get_centre = self.control_centre(state)
        
        '''print("VALUES OF HEURISTICS - PCC - NB_M - OPP_INFL")
        print(params[0]*pcc)
        print(params[1]*nb_maillons)
        print(params[3]*get_centre)'''
        return params[0]*pcc + params[1]*nb_maillons #+ params[3]*get_centre #+ 0.4*params[2]*opp_influence
            

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
        
        '''# Opening moves
        if current_state.step < 2:
            if self.piece_type == "R":
                if current_state.step == 0: #premier tour à jouer pour les rouge (toujours possible de jouer)
                    opening = LightAction({"piece": self.piece_type, "position": (10, 5)})
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
                        return first_answer
                    elif LightAction({"piece": self.piece_type, "position": (4, 8)}) in possible_actions: # If our opening is already taken, try to counter it
                        return LightAction({"piece": self.piece_type, "position": (4, 8)})
                    else:
                        return LightAction({"piece": self.piece_type, "position": (3, 4)})
                else:
                    first_answer = LightAction({"piece": self.piece_type, "position": (10, 3)})
                    if first_answer in possible_actions:
                        return first_answer'''
                
        # Opening moves
        if current_state.step < 2:
            if self.piece_type == "R":
                if current_state.step == 0: #premier tour à jouer pour les rouge (toujours possible de jouer)
                    opening = LightAction({"piece": self.piece_type, "position": (8, 5)})
                    return opening
            else:
                opening = LightAction({"piece": self.piece_type, "position": (5, 5)})
                if opening in possible_actions:
                    return opening
                else: # If our opening is already taken, try to counter it
                    return LightAction({"piece": self.piece_type, "position": (6, 5)})
        
        # First Answer moves
        elif current_state.step < 4:
            if self.piece_type == "R":
                if current_state.step == 2: #second tour à jouer pour les rouge (toujours possible de jouer)
                    nb_blue_cases, blue_cases = current_rep.get_pieces_player(current_state.players[1])
                    first_answer = LightAction({"piece": self.piece_type, "position": (4, 9)})
                    if first_answer in possible_actions:
                        return first_answer
                    else:
                        return LightAction({"piece": self.piece_type, "position": (4, 8)})
            else:
                nb_red_cases, red_cases = current_rep.get_pieces_player(current_state.players[0])
                first_answer = LightAction({"piece": self.piece_type, "position": (7, 9)})
                if first_answer in possible_actions:
                    return first_answer
                elif LightAction({"piece": self.piece_type, "position": (8, 9)}) in possible_actions: # If our opening is already taken, try to counter it
                    return LightAction({"piece": self.piece_type, "position": (8, 9)})
                else:
                    return LightAction({"piece": self.piece_type, "position": (6, 9)})
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
                        if j >= 1 and j<= 12 and env.get((i-1,j)) != None and env.get((i-1,j+1)) == None and env.get((i-1,j)).piece_type == piece_type_opponent:
                            return LightAction({"piece": self.piece_type, "position": (i-1, j+1)})
                        # Maillon nord menacé (cas 2)
                        elif j >= 1 and j<= 12 and env.get((i-1,j)) == None and env.get((i-1,j+1)) != None and env.get((i-1,j+1)).piece_type == piece_type_opponent:
                            return LightAction({"piece": self.piece_type, "position": (i-1, j)})
                    # Check les maillons à la frontière sud
                    elif i == 12:
                        # Maillon sud menacé (cas 1)
                        if j >= 1 and j<= 12 and env.get((i+1,j)) != None and env.get((i+1,j-1)) == None and env.get((i+1,j)).piece_type == piece_type_opponent:
                            return LightAction({"piece": self.piece_type, "position": (i+1, j-1)})
                        # Maillon sud menacé (cas 2)
                        if j >= 1 and j<= 12 and env.get((i+1,j)) == None and env.get((i+1,j-1)) != None and env.get((i+1,j-1)).piece_type == piece_type_opponent:
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
                        if i >= 1 and i <= 12 and env.get((i,j-1)) != None and env.get((i+1,j-1)) == None and env.get((i,j-1)).piece_type == piece_type_opponent:
                            return LightAction({"piece": self.piece_type, "position": (i+1, j-1)})
                        # Maillon ouest menacé (cas 2)
                        elif i >= 1 and i <= 12 and env.get((i,j-1)) == None and env.get((i+1,j-1)) != None and env.get((i+1,j-1)).piece_type == piece_type_opponent:
                            return LightAction({"piece": self.piece_type, "position": (i, j-1)})
                    # Check les maillons à la frontière est
                    elif j == 12:
                        # Maillon est menacé (cas 1)
                        if i >= 1 and i <= 12 and env.get((i,j+1)) != None and env.get((i-1,j+1)) == None and env.get((i,j+1)).piece_type == piece_type_opponent:
                            return LightAction({"piece": self.piece_type, "position": (i-1, j+1)})
                        # Maillon est menacé (cas 2)
                        if i >= 1 and i <= 12 and env.get((i,j+1)) == None and env.get((i-1,j+1)) != None and env.get((i-1,j+1)).piece_type == piece_type_opponent:
                            return LightAction({"piece": self.piece_type, "position": (i, j+1)})
        
        # Si aucun maillon n'est menacé, on prend le move donné par alpha-beta
        print("-------------------------------------")
        print("------ AUCUN MAILLON DE MENACE ------")
        print("-------------------------------------")
        
        depth_allowed = 0
        if current_state.step <= 11:
            depth_allowed = 2
        else:
            depth_allowed = 3
            
        move = self.halpha_beta_strategy(current_state, self.master_heuristic, remaining_time, 2)
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

# -------------------------------
# Construire le graphe complet du plateau Hex
# -------------------------------
def build_hex_graph(state):
    G = nx.Graph()
    neigh_offsets = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, +1)]
    N = state.get_rep().get_dimensions()[0]
    for r in range(N):
        for q in range(N):
            G.add_node((q, r))

    for r in range(N):
        for q in range(N):
            for dq, dr in neigh_offsets:
                nq, nr = q + dq, r + dr
                if state.in_board((nq, nr)):
                    G.add_edge((q, r), (nq, nr))
    return G