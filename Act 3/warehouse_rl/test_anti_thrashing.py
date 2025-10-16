#!/usr/bin/env python3
"""
Test script to verify anti-thrashing mechanism works correctly.
"""

from environment.warehouse_env import WarehouseEnv
from agents.standardized_agents import create_aggressive_swap_agent

def main():
    env = WarehouseEnv(grid_width=12, grid_height=12, max_employees=15, order_arrival_rate=0.8)
    agent = create_aggressive_swap_agent(env)
    obs = env.reset()
    
    print("=== Anti-Thrashing Test ===")
    print(f"Swap cooldown period: {agent.swap_cooldown_period} timesteps")
    
    swaps_proposed = []
    swaps_blocked = []
    
    for step in range(2000):  # Longer test
        obs = env._get_observation()
        action = agent.get_action(obs)
        
        # Check for swap proposals
        layout_swap = action.get('layout_swap', [0, 0])
        if layout_swap != [0, 0] and layout_swap[0] != layout_swap[1]:
            pos1 = (layout_swap[0] % env.grid_width, layout_swap[0] // env.grid_width)
            pos2 = (layout_swap[1] % env.grid_width, layout_swap[1] // env.grid_width)
            
            swap_key = tuple(sorted([layout_swap[0], layout_swap[1]]))
            
            print(f"Step {step}: SWAP PROPOSED {pos1} <-> {pos2} (key: {swap_key})")
            swaps_proposed.append((step, swap_key))
            
            # Check if this is a repeat of a recent swap (excluding current step)
            for prev_step, prev_key in swaps_proposed[:-1]:  # Exclude current swap
                if prev_key == swap_key and step - prev_step < agent.swap_cooldown_period:
                    print(f"  ⚠️ WARNING: Potential thrashing detected! Same swap proposed at step {prev_step}")
                    swaps_blocked.append((step, swap_key, prev_step))
        
        env.step(action)
        
        # Show swap history status
        if step % 100 == 0 and agent.recent_swaps:
            print(f"Step {step}: {len(agent.recent_swaps)} swaps in cooldown history")
    
    print(f"\n=== RESULTS ===")
    print(f"Total swaps proposed: {len(swaps_proposed)}")
    print(f"Potential thrashing incidents: {len(swaps_blocked)}")
    
    if swaps_blocked:
        print(f"\n⚠️ Thrashing detected:")
        for step, key, prev_step in swaps_blocked:
            cooldown_remaining = agent.swap_cooldown_period - (step - prev_step)
            print(f"  Step {step}: Swap {key} (previous at step {prev_step}, {cooldown_remaining} steps remaining)")
    else:
        print(f"✅ No thrashing detected - anti-thrashing mechanism working!")
    
    # Show unique swaps vs total
    unique_swaps = set(key for step, key in swaps_proposed)
    print(f"Unique swap pairs: {len(unique_swaps)}")
    print(f"Repeat factor: {len(swaps_proposed) / len(unique_swaps) if unique_swaps else 0:.2f}")

if __name__ == "__main__":
    main()