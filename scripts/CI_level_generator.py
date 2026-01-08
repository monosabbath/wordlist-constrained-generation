import math

def generate_ci_levels(start_n, end_n, encounter_rate):
    """
    Calculates vocabulary levels where new words account for a 
    specific percentage of the current corpus coverage.
    """
    gamma = 0.5772156649  # Euler-Mascheroni constant
    levels = []
    current_n = start_n
    level_idx = 1
    
    # Store initial level
    levels.append({
        "Level": level_idx,
        "N": int(current_n),
        "Increase": 0,
        "Growth %": 0.0
    })

    while current_n < end_n:
        level_idx += 1
        
        # We solve for N_next using: 
        # (H_next - H_curr) / H_next = rate
        # H_next * (1 - rate) = H_curr
        # H_next = H_curr / (1 - rate)
        
        h_curr = math.log(current_n) + gamma
        h_next_target = h_curr / (1 - encounter_rate)
        
        # Convert back from Harmonic to N: N = e^(H - gamma)
        next_n = math.exp(h_next_target - gamma)
        
        # Edge case: if rate is too small or N is too large, it might not move
        if int(next_n) <= int(current_n):
            next_n = current_n + 1
            
        increase = int(next_n) - int(current_n)
        growth_pct = (increase / current_n) * 100
        
        current_n = next_n
        
        levels.append({
            "Level": level_idx,
            "N": int(current_n),
            "Increase": increase,
            "Growth %": round(growth_pct, 2)
        })
        
        if current_n > end_n:
            break

    return levels

# --- CONFIGURATION ---
START_N = 500
END_N = 20000
RATE = 0.03  # encounter rate for new words

results = generate_ci_levels(START_N, END_N, RATE)

# Print Header
print(f"{'Level':<8} | {'Vocab (N)':<12} | {'New Words':<12} | {'Growth %':<10}")
print("-" * 50)

for row in results:
    print(f"{row['Level']:<8} | {row['N']:<12,} | {row['Increase']:<12,} | {row['Growth %']}%")