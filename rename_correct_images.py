import os
import shutil

folder = r"d:\college\semester5\image processing\project\Jigsaw-Puzzle-Solver\correct"

def get_path(name):
    return os.path.join(folder, f"{name}.png")

def swap(a, b):
    path_a = get_path(a)
    path_b = get_path(b)
    path_temp = get_path(f"temp_{a}")
    
    if os.path.exists(path_a) and os.path.exists(path_b):
        os.rename(path_a, path_temp)
        os.rename(path_b, path_a)
        os.rename(path_temp, path_b)
        print(f"Swapped {a} and {b}")
    else:
        print(f"Cannot swap {a} and {b}, one or both missing.")

def cycle_3(a, b, c, target_a, target_b, target_c):
    # a moves to target_a
    # b moves to target_b
    # c moves to target_c
    # In user request: 41 -> 43, 42 -> 41, 43 -> 42
    # So a=41, target_a=43
    #    b=42, target_b=41
    #    c=43, target_c=42
    
    p_a = get_path(a)
    p_b = get_path(b)
    p_c = get_path(c)
    
    t_a = get_path(f"temp_{a}")
    t_b = get_path(f"temp_{b}")
    t_c = get_path(f"temp_{c}")
    
    if os.path.exists(p_a) and os.path.exists(p_b) and os.path.exists(p_c):
        os.rename(p_a, t_a)
        os.rename(p_b, t_b)
        os.rename(p_c, t_c)
        
        os.rename(t_a, get_path(target_a))
        os.rename(t_b, get_path(target_b))
        os.rename(t_c, get_path(target_c))
        print(f"Cycled {a}->{target_a}, {b}->{target_b}, {c}->{target_c}")
    else:
        print(f"Cannot cycle {a}, {b}, {c}, missing files.")

# 1. 3,4 swap
swap(3, 4)
# 2. 5,6 swap
swap(5, 6)
# 3. 31,32 swap
swap(31, 32)
# 4. 33,34 swap
swap(33, 34)

# 5. 41 --> 43, 42 --> 41, 43 -> 42
cycle_3(41, 42, 43, 43, 41, 42)

# 6. 44 -> 45, 45 -> 44 (Swap)
swap(44, 45)

# 7. 51 -> 48, 48-> 51 (Swap)
swap(51, 48)

# 8. 83 -> 80, 80 -> 83 (Swap)
swap(83, 80)

# 9. 73,74 swap
swap(73, 74)

# 10. 53,54 swap
swap(53, 54)

# 11. 78-> 76, 77-> 78, 76->77
# a=78 -> target=76
# b=77 -> target=78
# c=76 -> target=77
cycle_3(78, 77, 76, 76, 78, 77)

# 12. 97,98 swap
swap(97, 98)

# 13. 93,94 swap
swap(93, 94)

# 14. 84,85 swap
swap(84, 85)
