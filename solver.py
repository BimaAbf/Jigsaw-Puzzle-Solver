import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import heapq
import random
import itertools
from jigsaw.extractor import JigsawExtractor
from solver_forest import solve_forest

# --- 1. Feature Extraction & Similarity Metric ---

def extract_borders(img):
    """
    Extracts the 4 borders of a puzzle piece image.
    Returns a dictionary with keys 'top', 'bottom', 'left', 'right'.
    """
    h, w, _ = img.shape
    return {
        'top': img[0, :, :],      # Shape (w, 3)
        'bottom': img[-1, :, :],  # Shape (w, 3)
        'left': img[:, 0, :],     # Shape (h, 3)
        'right': img[:, -1, :]    # Shape (h, 3)
    }

def calculate_nssd(edge_a, edge_b):
    """
    Simple but effective edge dissimilarity using SSD.
    Returns dissimilarity (lower is better match).
    """
    # Ensure same length
    if len(edge_a) != len(edge_b):
        min_len = min(len(edge_a), len(edge_b))
        edge_a = edge_a[:min_len]
        edge_b = edge_b[:min_len]
        
    edge_a = edge_a.astype(np.float32)
    edge_b = edge_b.astype(np.float32)
    
    # Simple SSD - lower is better
    diff = np.sum((edge_a - edge_b) ** 2)
    
    # Normalize by edge length and color range
    normalized_diff = diff / (len(edge_a) * 255.0 * 255.0 * 3.0)
    
    return normalized_diff

def compute_compatibility_matrices(pieces):
    """
    Computes dissimilarity matrices using simple SSD.
    Lower values = better match (we invert for compatibility scoring later).
    """
    n = len(pieces)
    H = np.zeros((n, n))  # Horizontal dissimilarity
    V = np.zeros((n, n))  # Vertical dissimilarity
    
    borders = [extract_borders(p) for p in pieces]
    
    for i in range(n):
        for j in range(n):
            if i == j: 
                H[i, j] = float('inf')  # Invalid
                V[i, j] = float('inf')
                continue
            
            # Horizontal: i's right should match j's left
            H[i, j] = calculate_nssd(borders[i]['right'], borders[j]['left'])
            
            # Vertical: i's bottom should match j's top
            V[i, j] = calculate_nssd(borders[i]['bottom'], borders[j]['top'])
    
    # Convert dissimilarity to compatibility (invert and normalize)
    # Use softmax-like transformation for better discrimination
    H_compat = np.exp(-H * 10.0)  # Amplify differences
    V_compat = np.exp(-V * 10.0)
    
    # Normalize each row so max is 1.0
    for i in range(n):
        max_h = np.max(H_compat[i, :])
        max_v = np.max(V_compat[i, :])
        if max_h > 0:
            H_compat[i, :] = H_compat[i, :] / max_h
        if max_v > 0:
            V_compat[i, :] = V_compat[i, :]/ max_v
    
    return H_compat, V_compat

# --- 2. Solvers ---

def calculate_board_score(board, H, V, grid_size):
    """Calculates the total compatibility score of a grid configuration."""
    score = 0.0
    # Horizontal links
    for r in range(grid_size):
        for c in range(grid_size - 1):
            p1 = board[r * grid_size + c]
            p2 = board[r * grid_size + c + 1]
            if p1 is not None and p2 is not None:
                score += H[p1, p2]
                
    # Vertical links
    for r in range(grid_size - 1):
        for c in range(grid_size):
            p1 = board[r * grid_size + c]
            p2 = board[(r + 1) * grid_size + c]
            if p1 is not None and p2 is not None:
                score += V[p1, p2]
    return score

def solve_backtracking(n_pieces, H, V):
    """
    Exact solver for small puzzles (2x2, 3x3).
    Uses recursion with pruning to find the permutation with max score.
    """
    grid_size = int(np.sqrt(n_pieces))
    best_score = -1.0
    best_board = None
    
    # Pre-compute valid candidates to speed up? 
    # For N=3 (9 pieces), 9! = 362k, feasible to just brute force with slight pruning.
    
    used = [False] * n_pieces
    current_board = [None] * n_pieces
    
    def backtrack(idx, current_score):
        nonlocal best_score, best_board
        
        if idx == n_pieces:
            if current_score > best_score:
                best_score = current_score
                best_board = list(current_board)
            return

        r, c = idx // grid_size, idx % grid_size
        
        # Optimization: If current partial score is already too far behind best, prune?
        # (Skipping complex pruning for simplicity on small N)

        for p_id in range(n_pieces):
            if not used[p_id]:
                # Calculate score contribution
                score_inc = 0.0
                valid = True
                
                # Check Left Neighbor
                if c > 0:
                    left_p = current_board[idx - 1]
                    s = H[left_p, p_id]
                    if s < 0.1: valid = False # Hard prune on very bad matches
                    score_inc += s
                
                # Check Top Neighbor
                if r > 0:
                    top_p = current_board[idx - grid_size]
                    s = V[top_p, p_id]
                    if s < 0.1: valid = False
                    score_inc += s
                
                if valid:
                    used[p_id] = True
                    current_board[idx] = p_id
                    backtrack(idx + 1, current_score + score_inc)
                    current_board[idx] = None
                    used[p_id] = False

    backtrack(0, 0.0)
    return best_board

def solve_beam_search(n_pieces, H, V, beam_width=100):
    """
    Beam Search for medium puzzles (4x4, 5x5).
    Maintains top K partial solutions.
    """
    grid_size = int(np.sqrt(n_pieces))
    
    # State: (score, [list_of_placed_piece_ids], set_of_used_ids)
    # Start with empty board
    beam = [(0.0, [], set())] 
    
    for step in range(n_pieces):
        new_beam = []
        r, c = step // grid_size, step % grid_size
        
        for score, board, used in beam:
            # Try adding every unused piece
            for p_id in range(n_pieces):
                if p_id not in used:
                    new_score = score
                    
                    # Check Left
                    if c > 0:
                        left_p = board[-1]
                        new_score += H[left_p, p_id]
                        
                    # Check Top
                    if r > 0:
                        top_p = board[step - grid_size]
                        new_score += V[top_p, p_id]
                    
                    new_board = board + [p_id]
                    new_used = used.copy()
                    new_used.add(p_id)
                    new_beam.append((new_score, new_board, new_used))
        
        # Keep top K
        # Note: heapq is min-heap, so we use nlargest
        beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])
        
        # Print progress
        # if step % 5 == 0: print(f"Step {step}/{n_pieces}, Best Score: {beam[0][0]:.2f}")

    return beam[0][1] # Return board of best state

def solve_greedy_local(n_pieces, H, V):
    """
    Advanced solver for large puzzles (8x8+).
    Uses best-buddy matching + progressive assembly + refined optimization.
    """
    grid_size = int(np.sqrt(n_pieces))
    
    # --- Phase 1: Best Buddy Analysis ---
    print(f"Phase 1: Best Buddy Analysis...")
    
    # Find "best buddies" - pairs of pieces that mutually prefer each other
    best_buddies_h = []  # Horizontal best buddies
    best_buddies_v = []  # Vertical best buddies
    
    for i in range(n_pieces):
        # Find best right neighbor for piece i
        best_right = np.argmax([H[i, j] if j != i else -1 for j in range(n_pieces)])
        # Check if it's mutual (best left neighbor of best_right is i)
        best_left_of_right = np.argmax([H[j, best_right] if j != best_right else -1 for j in range(n_pieces)])
        if best_left_of_right == i and H[i, best_right] > 0.7:  # Strong mutual match
            best_buddies_h.append((i, best_right, H[i, best_right]))
    
    for i in range(n_pieces):
        # Find best bottom neighbor for piece i
        best_bottom = np.argmax([V[i, j] if j != i else -1 for j in range(n_pieces)])
        # Check if it's mutual
        best_top_of_bottom = np.argmax([V[j, best_bottom] if j != best_bottom else -1 for j in range(n_pieces)])
        if best_top_of_bottom == i and V[i, best_bottom] > 0.7:  # Strong mutual match
            best_buddies_v.append((i, best_bottom, V[i, best_bottom]))
    
    print(f"  Found {len(best_buddies_h)} horizontal best buddies")
    print(f"  Found {len(best_buddies_v)} vertical best buddies")
    
    # --- Phase 2: Use Beam Search with higher beam width ---
    print(f"\nPhase 2: Enhanced Beam Search...")
    beam_width = min(100, n_pieces * 2)  # Increased beam width
    initial_board = solve_beam_search(n_pieces, H, V, beam_width=beam_width)
    initial_score = calculate_board_score(initial_board, H, V, grid_size)
    initial_score = calculate_board_score(initial_board, H, V, grid_size)
    
    # --- Phase 3: Multiple Greedy Starts with Best Buddy Constraints ---
    print(f"\nPhase 3: Constrained Greedy Assembly...")
    
    best_greedy_board = initial_board
    best_greedy_score = initial_score
    
    # Try greedy starts that respect best buddy relationships
    for trial in range(5):
        used = set()
        board = [None] * n_pieces
        
        # Start with the strongest best buddy pair
        if trial == 0 and best_buddies_h:
            # Use the strongest horizontal buddy pair as seed
            best_buddies_h.sort(key=lambda x: x[2], reverse=True)
            seed1, seed2, _ = best_buddies_h[0]
            center = (grid_size // 2) * grid_size + (grid_size // 2)
            board[center] = seed1
            board[center + 1] = seed2
            used.add(seed1)
            used.add(seed2)
        else:
            # Standard center start
            seed = np.argmax([np.sum(H[i, :]) + np.sum(V[i, :]) for i in range(n_pieces)])
            center = (grid_size // 2) * grid_size + (grid_size // 2)
            board[center] = seed
            used.add(seed)
        
        # Build outward, prioritizing high-scoring neighbors
        positions = []
        for r in range(grid_size):
            for c in range(grid_size):
                idx = r * grid_size + c
                if board[idx] is None:
                    positions.append((idx, r, c))
        
        # Sort by number of placed neighbors (fill densely)
        def count_neighbors(idx, r, c):
            count = 0
            if c > 0 and board[idx-1] is not None: count += 1
            if c < grid_size-1 and board[idx+1] is not None: count += 1
            if r > 0 and board[idx-grid_size] is not None: count += 1
            if r < grid_size-1 and board[idx+grid_size] is not None: count += 1
            return count
        
        while any(board[i] is None for i in range(n_pieces)):
            # Find position with most neighbors
            valid_positions = [(idx, r, c) for idx, r, c in positions if board[idx] is None]
            if not valid_positions:
                break
            
            # Sort by neighbor count
            valid_positions.sort(key=lambda x: count_neighbors(x[0], x[1], x[2]), reverse=True)
            idx, r, c = valid_positions[0]
            
            # Find best piece considering ALL neighbors
            best_p = -1
            best_score = -float('inf')
            
            for p in range(n_pieces):
                if p not in used:
                    score = 0
                    count = 0
                    
                    # Weight by number of neighbors (more neighbors = more reliable)
                    if c > 0 and board[idx-1] is not None:
                        score += H[board[idx-1], p] * 2.0  # Left
                        count += 2
                    if c < grid_size-1 and board[idx+1] is not None:
                        score += H[p, board[idx+1]] * 2.0  # Right
                        count += 2
                    if r > 0 and board[idx-grid_size] is not None:
                        score += V[board[idx-grid_size], p] * 2.0  # Top
                        count += 2
                    if r < grid_size-1 and board[idx+grid_size] is not None:
                        score += V[p, board[idx+grid_size]] * 2.0  # Bottom
                        count += 2
                    
                    # Add best buddy bonus
                    for p1, p2, buddy_score in best_buddies_h:
                        if (c > 0 and board[idx-1] == p1 and p == p2) or \
                           (c < grid_size-1 and board[idx+1] == p2 and p == p1):
                            score += 5.0  # Large bonus for buddy pairs
                    
                    for p1, p2, buddy_score in best_buddies_v:
                        if (r > 0 and board[idx-grid_size] == p1 and p == p2) or \
                           (r < grid_size-1 and board[idx+grid_size] == p2 and p == p1):
                            score += 5.0
                    
                    if count > 0:
                        avg_score = score / count
                        if avg_score > best_score:
                            best_score = avg_score
                            best_p = p
            
            if best_p == -1:
                # No scored neighbors, pick any unused
                for p in range(n_pieces):
                    if p not in used:
                        best_p = p
                        break
            
            if best_p != -1:
                board[idx] = best_p
                used.add(best_p)
        
        trial_score = calculate_board_score(board, H, V, grid_size)
        if trial_score > best_greedy_score:
            best_greedy_score = trial_score
            best_greedy_board = board
            print(f"  Trial {trial}: New best score {trial_score:.2f}")

    board = list(best_greedy_board)
    current_score = best_greedy_score
    
    # --- Phase 4: Aggressive Simulated Annealing ---
    print(f"\nPhase 4: Aggressive Simulated Annealing")
    print(f"Initial Score: {current_score:.2f}")
    
    max_iter = 100000  # Even more iterations
    temp = 2.0  # Higher starting temperature
    cooling_rate = 0.99992  # Very slow cooling
    min_temp = 0.0001
    
    best_board = list(board)
    best_score = current_score
    
    accept_count = 0
    reject_count = 0
    
    for it in range(max_iter):
        # Adaptive cooling
        if it % 100 == 0:
            temp *= cooling_rate
            temp = max(temp, min_temp)
        
        # Try different swap strategies
        if random.random() < 0.7:
            # Standard: Random swap
            idx1, idx2 = random.sample(range(n_pieces), 2)
        else:
            # Strategic: Swap pieces with low local scores
            local_scores = []
            for idx in range(n_pieces):
                r, c = idx // grid_size, idx % grid_size
                local_s = 0
                if c > 0: local_s += H[board[idx-1], board[idx]]
                if c < grid_size-1: local_s += H[board[idx], board[idx+1]]
                if r > 0: local_s += V[board[idx-grid_size], board[idx]]
                if r < grid_size-1: local_s += V[board[idx], board[idx+grid_size]]
                local_scores.append((local_s, idx))
            
            # Pick from worst 20%
            local_scores.sort()
            worst_n = max(2, n_pieces // 5)
            idx1, idx2 = random.sample([idx for _, idx in local_scores[:worst_n]], 2)
        
        # Calculate score change
        def get_comprehensive_local_score(idx, p_id, current_board):
            r, c = idx // grid_size, idx % grid_size
            s = 0
            if c > 0 and current_board[idx-1] is not None:
                s += H[current_board[idx-1], p_id]
            if c < grid_size - 1 and current_board[idx+1] is not None:
                s += H[p_id, current_board[idx+1]]
            if r > 0 and current_board[idx-grid_size] is not None:
                s += V[current_board[idx-grid_size], p_id]
            if r < grid_size - 1 and current_board[idx+grid_size] is not None:
                s += V[p_id, current_board[idx+grid_size]]
            return s
        
        s_before = (get_comprehensive_local_score(idx1, board[idx1], board) + 
                   get_comprehensive_local_score(idx2, board[idx2], board))
        
        # Perform swap
        board[idx1], board[idx2] = board[idx2], board[idx1]
        
        s_after = (get_comprehensive_local_score(idx1, board[idx1], board) + 
                  get_comprehensive_local_score(idx2, board[idx2], board))
        
        delta = s_after - s_before
        
        # Acceptance criterion
        if delta > 0 or (temp > min_temp and random.random() < np.exp(delta / temp)):
            # Accept
            current_score += delta
            accept_count += 1
            
            if current_score > best_score:
                best_score = current_score
                best_board = list(board)
                if it % 1000 == 0:
                    print(f"  Iter {it}: Best={best_score:.2f}, Current={current_score:.2f}, "
                          f"Temp={temp:.4f}, Accept Rate={accept_count/(accept_count+reject_count):.2%}")
        else:
            # Reject
            board[idx1], board[idx2] = board[idx2], board[idx1]
            reject_count += 1
        
        # Early stopping if converged
        if it > 10000 and accept_count == 0 and temp < 0.001:
            print(f"Converged at iteration {it}")
            break
    
    print(f"Final Score: {best_score:.2f}")
    print(f"Total Accepted: {accept_count}, Rejected: {reject_count}")
    return best_board

# --- 3. Main Pipeline ---

def slice_image(image, n):
    """Slices an image into n x n pieces."""
    h, w, _ = image.shape
    piece_h = h // n
    piece_w = w // n
    pieces = []
    for r in range(n):
        for c in range(n):
            y = r * piece_h
            x = c * piece_w
            piece = image[y:y+piece_h, x:x+piece_w]
            pieces.append(piece)
    return pieces

def solve_puzzle_lab2(image_path, metadata, output_root="results", shuffle=True, show_plot=True):
    """
    Solves a puzzle given the image path and metadata containing the grid dimension.
    metadata format: {"filename": "...", "predicted_label": "2x2", ...}
    """
    # 1. Determine Grid Size
    # Prioritize predicted_label, fallback to true_label, then default
    label = metadata.get("predicted_label", metadata.get("true_label", "2x2"))
    try:
        grid_size = int(label.split('x')[0])
    except:
        grid_size = 2
        
    # 2. Load Image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    full_image = cv2.imread(image_path)
    if full_image is None:
        print("Failed to load image.")
        return None
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    
    # 3. Slice Image
    pieces = slice_image(full_image, grid_size)
    n_pieces = len(pieces)
    
    # 4. Shuffle Pieces (to create the puzzle)
    if shuffle:
        indices = list(range(n_pieces))
        random.shuffle(indices)
        shuffled_pieces = [pieces[i] for i in indices]
    else:
        shuffled_pieces = pieces
    
    print(f"Solving {os.path.basename(image_path)} ({grid_size}x{grid_size})...")
    
    # 5. Compute Compatibility
    print("Computing compatibility matrices...")
    H, V = compute_compatibility_matrices(shuffled_pieces)
    
    # 6. Select Strategy
    solution_indices = []
    if grid_size <= 3:
        print("Strategy: Exact Backtracking")
        solution_indices = solve_backtracking(n_pieces, H, V)
    elif grid_size <= 5:
        print("Strategy: Beam Search")
        solution_indices = solve_beam_search(n_pieces, H, V, beam_width=200)
    else:
        print("Strategy: Forest Solver (Kruskal's + Backtracking)")
        solution_indices = solve_forest(shuffled_pieces, grid_size)
        
    # 7. Evaluate & Visualize
    if not solution_indices:
        print("Failed to find solution.")
        return None

    h_p, w_p, _ = pieces[0].shape
    canvas = np.zeros((grid_size * h_p, grid_size * w_p, 3), dtype=np.uint8)
    
    for idx, p_id in enumerate(solution_indices):
        if p_id is None: continue
        r, c = idx // grid_size, idx % grid_size
        canvas[r*h_p:(r+1)*h_p, c*w_p:(c+1)*w_p] = shuffled_pieces[p_id]
        
    if show_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas)
        plt.title(f"Solved: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()
    
    # Save Result
    if output_root:
        dim_folder = f"{grid_size}x{grid_size}"
        save_dir = os.path.join(output_root, dim_folder)
        os.makedirs(save_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, filename)
        
        # Convert RGB back to BGR for OpenCV saving
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, canvas_bgr)
        print(f"Saved result to {save_path}")
        
    return canvas

# --- 4. JigsawExtractor Pipeline ---

def get_image_descriptor(image_path):
    """
    Uses JigsawExtractor to analyze the image and return a descriptor.
    """
    # Initialize extractor with the directory of the image
    source_dir = os.path.dirname(image_path)
    extractor = JigsawExtractor(source_dir, output_dir="temp_output", debug_visuals=False)
    
    # Load and process the specific image
    if not extractor.load_image(image_path):
        print(f"Failed to load image: {image_path}")
        return None
        
    extractor.preprocess()
    extractor.detect_grid_size()
    
    # Create descriptor
    n = extractor.detected_n
    descriptor = {
        "filename": os.path.basename(image_path),
        "predicted_label": f"{n}x{n}", 
        "grid_size": n,
        "scores": extractor.scores
    }
    return descriptor

def run_pipeline(image_path, shuffle=True, show_plot=True):
    print(f"--- Pipeline Start: {image_path} ---")
    
    # Step 1: Extract Descriptor
    print("1. Extracting Descriptor...")
    descriptor = get_image_descriptor(image_path)
    if not descriptor:
        return None
    
    print(f"   Detected Grid: {descriptor['predicted_label']}")
    print(f"   Scores: {descriptor['scores']}")
    
    # Step 2: Solve Puzzle
    print("2. Solving Puzzle...")
    result = solve_puzzle_lab2(image_path, descriptor, shuffle=shuffle, show_plot=show_plot)
    print("--- Pipeline End ---")
    return result