#!/usr/bin/env python3
"""
Debug script to verify that grid updates are happening correctly during layout swaps.
"""

import numpy as np
from environment.warehouse_env import WarehouseEnv
from agents.standardized_agents import create_aggressive_swap_agent

def print_grid_items(grid, title="Grid State"):
    """Print current items in grid for debugging"""
    print(f"\n=== {title} ===")
    items_found = {}
    for y in range(grid.height):
        for x in range(grid.width):
            item = grid.get_item_at_position(x, y)
            if item is not None:
                if item not in items_found:
                    items_found[item] = []
                items_found[item].append((x, y))
    
    for item_type in sorted(items_found.keys()):
        positions = items_found[item_type]
        print(f"  Item {item_type}: {positions}")
    print(f"Total items: {sum(len(positions) for positions in items_found.values())}")
    return items_found

def main():
    # Create environment
    env = WarehouseEnv(
        grid_width=12,
        grid_height=12,
        max_employees=15,
        order_arrival_rate=0.8,
        render_mode=None
    )
    
    # Create agent
    agent = create_aggressive_swap_agent(env)
    
    # Reset environment
    obs = env.reset()
    
    print("=== Grid Update Debug Test ===")
    
    # Track grid state before swaps
    initial_items = print_grid_items(env.warehouse_grid, "Initial Grid State")
    
    # Run simulation and track grid changes
    for step in range(2000):  # Comprehensive test
        obs = env._get_observation()
        action = agent.get_action(obs)
        
        # Check if a swap was proposed
        layout_swap = action.get('layout_swap', [0, 0])
        if layout_swap[0] != layout_swap[1] and layout_swap[0] != 0 and layout_swap[1] != 0:
            pos1 = (layout_swap[0] % env.grid_width, layout_swap[0] // env.grid_width)
            pos2 = (layout_swap[1] % env.grid_width, layout_swap[1] // env.grid_width)
            
            item1_before = env.warehouse_grid.get_item_at_position(pos1[0], pos1[1])
            item2_before = env.warehouse_grid.get_item_at_position(pos2[0], pos2[1])
            
            print(f"\n[Step {step}] SWAP PROPOSED:")
            print(f"  Position 1: {pos1}, Item: {item1_before}")
            print(f"  Position 2: {pos2}, Item: {item2_before}")
            
            # Execute the action
            obs, reward, done, truncated, info = env.step(action)
            
            # Check grid state immediately after
            item1_after = env.warehouse_grid.get_item_at_position(pos1[0], pos1[1])
            item2_after = env.warehouse_grid.get_item_at_position(pos2[0], pos2[1])
            
            print(f"  AFTER STEP:")
            print(f"  Position 1: {pos1}, Item: {item1_after}")
            print(f"  Position 2: {pos2}, Item: {item2_after}")
            
            if item1_before != item1_after or item2_before != item2_after:
                print("  ✅ GRID CHANGED IMMEDIATELY")
            else:
                print("  ⚠️  NO IMMEDIATE GRID CHANGE (manager needs to complete task)")
        else:
            # Execute the action normally
            obs, reward, done, truncated, info = env.step(action)
        
        # Check for completed relocations
        if hasattr(env, 'last_swap_info') and env.last_swap_info:
            swap_info = env.last_swap_info
            print(f"\n[Step {step}] RELOCATION COMPLETED:")
            print(f"  Source: {swap_info['source_pos']}, Item: {swap_info['source_item']}")
            print(f"  Target: {swap_info['target_pos']}, Item: {swap_info['target_item']}")
            
            # Verify the swap actually happened
            actual_source = env.warehouse_grid.get_item_at_position(swap_info['source_pos'][0], swap_info['source_pos'][1])
            actual_target = env.warehouse_grid.get_item_at_position(swap_info['target_pos'][0], swap_info['target_pos'][1])
            
            print(f"  VERIFICATION:")
            print(f"  Source position now has: {actual_source} (expected: {swap_info['target_item']})")
            print(f"  Target position now has: {actual_target} (expected: {swap_info['source_item']})")
            
            if (actual_source == swap_info['target_item'] and 
                actual_target == swap_info['source_item']):
                print("  ✅ SWAP VERIFIED SUCCESSFUL")
            else:
                print("  ❌ SWAP VERIFICATION FAILED")
            
            env.last_swap_info = None  # Clear for next detection
        
        if done or truncated:
            break
    
    # Show final grid state
    final_items = print_grid_items(env.warehouse_grid, "Final Grid State")
    
    # Compare initial vs final
    print(f"\n=== COMPARISON ===")
    changed_items = 0
    for item_type in set(initial_items.keys()) | set(final_items.keys()):
        initial_pos = set(initial_items.get(item_type, []))
        final_pos = set(final_items.get(item_type, []))
        
        if initial_pos != final_pos:
            changed_items += 1
            print(f"Item {item_type}: {list(initial_pos)} -> {list(final_pos)}")
    
    if changed_items > 0:
        print(f"✅ {changed_items} item types changed positions")
    else:
        print("⚠️  No items changed positions")

if __name__ == "__main__":
    main()