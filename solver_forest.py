"""
Forest-based Jigsaw Solver (Kruskal's Algorithm approach).

This solver treats the puzzle assembly as a forest merging problem.
1. Start with N separate trees (one per piece).
2. Calculate all pairwise edge compatibilities.
3. Sort edges by compatibility (best to worst).
4. Iteratively merge trees if:
   - They are not already connected.
   - The merge does not create a spatial conflict (overlap).
   - The merged component fits within the grid dimensions (8x8).
"""

import numpy as np
import cv2
import time

class Piece:
    def __init__(self, pid, image):
        self.pid = pid
        self.image = image
        self.r = 0
        self.c = 0

class Component:
    def __init__(self, piece):
        self.id = piece.pid
        self.pieces = {piece.pid: piece}  
        self.min_r, self.max_r = 0, 0
        self.min_c, self.max_c = 0, 0

    def shift(self, dr, dc):
       
        for p in self.pieces.values():
            p.r += dr
            p.c += dc
        self.update_bounds()

    def update_bounds(self):
        coords = [(p.r, p.c) for p in self.pieces.values()]
        self.min_r = min(r for r, c in coords)
        self.max_r = max(r for r, c in coords)
        self.min_c = min(c for r, c in coords)
        self.max_c = max(c for r, c in coords)

    def width(self):
        return self.max_c - self.min_c + 1

    def height(self):
        return self.max_r - self.min_r + 1

def extract_features(piece):
    
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB).astype(np.float32)
    top = (lab[0, :, :], lab[1, :, :])
    bottom = (lab[-1, :, :], lab[-2, :, :])
    left = (lab[:, 0, :], lab[:, 1, :])
    right = (lab[:, -1, :], lab[:, -2, :])
    
    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}

def calculate_dissimilarity(edge_a, edge_b):
    out_a, in_a = edge_a
    out_b, in_b = edge_b
 
    pred_b = 2 * out_a - in_a
    diff_pred_b = out_b - pred_b
    dist_pred_b = np.sqrt(np.sum(diff_pred_b ** 2, axis=1))
    
    
    pred_a = 2 * out_b - in_b
    diff_pred_a = out_a - pred_a
    dist_pred_a = np.sqrt(np.sum(diff_pred_a ** 2, axis=1))
    
    total_dist = np.mean(dist_pred_b + dist_pred_a)
    
    return total_dist

def get_all_matches(pieces, features=None):
    n = len(pieces)
    if features is None:
        features = [extract_features(p) for p in pieces]
  
    dists = {i: {'top': [], 'bottom': [], 'left': [], 'right': []} for i in range(n)}
    
    print("Computing all pairwise dissimilarities...")
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            d = calculate_dissimilarity(features[i]['right'], features[j]['left'])
            dists[i]['right'].append((d, j, 'left'))
            dists[j]['left'].append((d, i, 'right'))

            d = calculate_dissimilarity(features[i]['bottom'], features[j]['top'])
            dists[i]['bottom'].append((d, j, 'top'))
            dists[j]['top'].append((d, i, 'bottom'))
            
    # 2. Apply Ratio Test and Best Buddy Prioritization
    final_matches = []
    
    print("Applying Ratio Test and Best Buddy filtering...")
    
    best_matches_map = {}
    
    for i in range(n):
        for side in ['top', 'bottom', 'left', 'right']:
            candidates = dists[i][side]
            if not candidates: continue
            candidates.sort(key=lambda x: x[0])
            best_dist, best_pid, best_side = candidates[0]
            best_matches_map[(i, side)] = (best_pid, best_side)

    for i in range(n):
        for side in ['top', 'bottom', 'left', 'right']:
            candidates = dists[i][side]
            if not candidates: continue
            
            candidates.sort(key=lambda x: x[0])
            
            best_dist, best_pid, best_side = candidates[0]
            
            if len(candidates) > 1:
                second_dist = candidates[1][0]
                if second_dist < 1e-6: second_dist = 1e-6
                ratio = best_dist / second_dist
            else:
                ratio = 0.0

            is_best_buddy = False
            if (best_pid, best_side) in best_matches_map:
                reciprocal_pid, reciprocal_side = best_matches_map[(best_pid, best_side)]

                if reciprocal_pid == i and reciprocal_side == side:
                    is_best_buddy = True
            
            
            if is_best_buddy:
           
                combined_score = -1000.0 + ratio
            else:
                combined_score = ratio
            
            final_matches.append((combined_score, i, side, best_pid, best_side))

    final_matches.sort(key=lambda x: x[0])
    
    return final_matches

def solve_forest(pieces, grid_size=8):
    start_time = time.time()
    n = len(pieces)
    

    print("Extracting features...")
    features = [extract_features(p) for p in pieces]
    

    piece_objs = [Piece(i, pieces[i]) for i in range(n)]
    components = {i: Component(piece_objs[i]) for i in range(n)} # pid -> Component

    
    matches = get_all_matches(pieces, features)
    print(f"Generated {len(matches)} candidate matches.")
    
   
    merge_count = 0
    
    for score, pid_a, side_a, pid_b, side_b in matches:
        comp_a = components[pid_a]
        comp_b = components[pid_b]
        
       
        if comp_a is comp_b:
            continue
            
        p_a = comp_a.pieces[pid_a]
        p_b = comp_b.pieces[pid_b]
        target_r_b = p_a.r
        target_c_b = p_a.c
        
        if side_a == 'right':
            target_c_b += 1
        elif side_a == 'bottom':
            target_r_b += 1

        dr = target_r_b - p_b.r
        dc = target_c_b - p_b.c
        

        new_min_r = min(comp_a.min_r, comp_b.min_r + dr)
        new_max_r = max(comp_a.max_r, comp_b.max_r + dr)
        new_min_c = min(comp_a.min_c, comp_b.min_c + dc)
        new_max_c = max(comp_a.max_c, comp_b.max_c + dc)
        
        if (new_max_r - new_min_r + 1) > grid_size:
            continue
        if (new_max_c - new_min_c + 1) > grid_size:
            continue
            

        overlap = False
        occupied_a = {(p.r, p.c) for p in comp_a.pieces.values()}
        
        for p in comp_b.pieces.values():
            if (p.r + dr, p.c + dc) in occupied_a:
                overlap = True
                break
        
        if overlap:
            continue
            
        comp_b.shift(dr, dc)
        comp_a.pieces.update(comp_b.pieces)
        comp_a.update_bounds()
        
        for pid in comp_b.pieces:
            components[pid] = comp_a
            
        merge_count += 1
        if merge_count % 5 == 0:
            print(f"Merged {merge_count} times. Largest component: {len(comp_a.pieces)} pieces.")
            
        if len(comp_a.pieces) == n:
            print("Puzzle fully assembled!")
            break
            

    unique_comps = list({id(c): c for c in components.values()}.values())
    unique_comps.sort(key=lambda c: len(c.pieces), reverse=True)
    
    best_comp = unique_comps[0]
    print(f"Final largest component has {len(best_comp.pieces)} pieces.")

    min_r, min_c = best_comp.min_r, best_comp.min_c
    
    board = [None] * (grid_size * grid_size)
    

    used_pids = set()
    for p in best_comp.pieces.values():
        r = p.r - min_r
        c = p.c - min_c
        if 0 <= r < grid_size and 0 <= c < grid_size:
            board[r * grid_size + c] = p.pid
            used_pids.add(p.pid)
            
    unused = [i for i in range(n) if i not in used_pids]
    print(f"Strict solver placed {len(used_pids)} pieces. {len(unused)} pieces remain.")
    
    print("Starting Backtracking Solver for gaps...")
    if solve_gaps_backtracking(board, unused, features, grid_size):
        print("Backtracking solver found a valid solution.")
    else:
        print("Backtracking solver could not find a perfect fit. Filling remaining linearly.")
        for i in range(len(board)):
            if board[i] is None and unused:
                board[i] = unused.pop(0)


    print("Running Global Refinement (Swapping)...")
    improved = True
    pass_count = 0
    max_passes = 5

    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        current_total_error = calculate_board_error(board, features, grid_size)
        print(f"Swapping Pass {pass_count} (Current Error: {current_total_error:.2f})")
        
        # Try swapping every pair of pieces
        for i in range(len(board)):
            for j in range(i + 1, len(board)):
            
                board[i], board[j] = board[j], board[i]
                
                new_error = calculate_board_error(board, features, grid_size)
                
                if new_error < current_total_error:
                    print(f"Swap improved error: {current_total_error:.2f} -> {new_error:.2f}")
                    current_total_error = new_error
                    improved = True
                else:
                   
                    board[i], board[j] = board[j], board[i]
    

    end_time = time.time()
    print(f"Total solving time: {end_time - start_time:.2f} seconds")
    return board

def solve_gaps_backtracking(board, unused_pids, features, grid_size):

    if not unused_pids:
        return True
        
  
    best_slot = None
    max_neighbors = -1
    best_neighbors_info = []
    
    empty_slots = []
    
    for r in range(grid_size):
        for c in range(grid_size):
            if board[r * grid_size + c] is None:
               
                neighbors = [] 
                if r > 0 and board[(r-1)*grid_size + c] is not None:
                    neighbors.append((board[(r-1)*grid_size + c], 'bottom', 'top'))
                if r < grid_size - 1 and board[(r+1)*grid_size + c] is not None:
                    neighbors.append((board[(r+1)*grid_size + c], 'top', 'bottom'))
                if c > 0 and board[r*grid_size + (c-1)] is not None:
                    neighbors.append((board[r*grid_size + (c-1)], 'right', 'left'))
                if c < grid_size - 1 and board[r*grid_size + (c+1)] is not None:
                    neighbors.append((board[r*grid_size + (c+1)], 'left', 'right'))
                
               
                if neighbors:
                    empty_slots.append({'r': r, 'c': c, 'neighbors': neighbors})
    
    if not empty_slots:

        for r in range(grid_size):
            for c in range(grid_size):
                if board[r * grid_size + c] is None:
                    empty_slots.append({'r': r, 'c': c, 'neighbors': []})
                    break
            if empty_slots: break
            

    empty_slots.sort(key=lambda x: len(x['neighbors']), reverse=True)
    target = empty_slots[0]
    

    candidates = []
    for pid in unused_pids:
        total_error = 0
        max_error = 0
        
        if not target['neighbors']:

            pass
        else:
            for n_pid, n_side, my_side in target['neighbors']:
                err = calculate_dissimilarity(features[n_pid][n_side], features[pid][my_side])
                total_error += err
                if err > max_error:
                    max_error = err

        score = total_error + max_error
        candidates.append((score, pid))
        
    candidates.sort(key=lambda x: x[0])

    for score, pid in candidates:

        idx = target['r'] * grid_size + target['c']
        board[idx] = pid
        unused_pids.remove(pid)
        
        if solve_gaps_backtracking(board, unused_pids, features, grid_size):
            return True

        board[idx] = None
        unused_pids.append(pid)
        unused_pids.sort() # Keep deterministic
        
    return False

def calculate_board_error(board, features, grid_size):
    total_error = 0
    count = 0
    
    for r in range(grid_size):
        for c in range(grid_size):
            pid = board[r * grid_size + c]
            if pid is None: continue
            
            if c < grid_size - 1:
                right_pid = board[r * grid_size + (c + 1)]
                if right_pid is not None:
                    err = calculate_dissimilarity(features[pid]['right'], features[right_pid]['left'])
                    total_error += err
                    count += 1

            if r < grid_size - 1:
                bottom_pid = board[(r + 1) * grid_size + c]
                if bottom_pid is not None:
                    err = calculate_dissimilarity(features[pid]['bottom'], features[bottom_pid]['top'])
                    total_error += err
                    count += 1
                    
    return total_error if count > 0 else float('inf')
