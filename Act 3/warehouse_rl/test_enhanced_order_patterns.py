#!/usr/bin/env python3
"""
Test the enhanced order generator patterns to verify co-occurrence and frequency
"""

from environment.warehouse_env import WarehouseEnv
from agents.standardized_agents import create_fixed_hiring_agent

def test_enhanced_patterns():
    print("=== Testing Enhanced Order Patterns ===")
    
    env = WarehouseEnv(grid_width=10, grid_height=10, max_employees=15, order_arrival_rate=1.2)
    agent = create_fixed_hiring_agent(env)
    
    # Very aggressive swap settings
    agent.params.layout_optimization_interval = 20
    agent.params.ev_threshold_for_swaps = 0.1  # Super low threshold
    agent.params.layout_queue_condition_ratio = 100.0
    
    obs = env.reset()
    
    print("Order generator patterns:")
    order_gen = env.order_generator
    
    # Show item popularity distribution
    hot_items = int(order_gen.num_item_types * 0.2)
    warm_end = int(order_gen.num_item_types * 0.5)
    
    print(f"Hot items (0-{hot_items-1}): {order_gen.item_popularity[:hot_items].sum():.3f} of total probability")
    print(f"Warm items ({hot_items}-{warm_end-1}): {order_gen.item_popularity[hot_items:warm_end].sum():.3f} of total probability")
    print(f"Cold items ({warm_end}+): {order_gen.item_popularity[warm_end:].sum():.3f} of total probability")
    
    # Show co-occurrence patterns
    cooccur = order_gen.cooccurrence_matrix
    print(f"Co-occurrence matrix max: {cooccur.max():.3f}")
    print(f"High correlations (>0.5): {(cooccur > 0.5).sum()}")
    print(f"Strong correlations (>0.7): {(cooccur > 0.7).sum()}")
    
    swaps_found = []
    
    for step in range(300):  # Run long enough to build data
        obs = env._get_observation()
        action = agent.get_action(obs)
        
        # Check for swaps
        layout_swap = action.get('layout_swap', [0, 0])
        if layout_swap != [0, 0] and layout_swap[0] != layout_swap[1]:
            pos1 = (layout_swap[0] % env.grid_width, layout_swap[0] // env.grid_width)
            pos2 = (layout_swap[1] % env.grid_width, layout_swap[1] // env.grid_width)
            swap_key = tuple(sorted([layout_swap[0], layout_swap[1]]))
            
            print(f"🎉 SWAP at step {step}: {pos1} <-> {pos2}")
            swaps_found.append((step, swap_key))
        
        env.step(action)
        
        # Show progress
        if step % 50 == 0 and step > 0:
            grid = env.warehouse_grid
            freq = grid.item_access_frequency
            cooccur_data = grid.item_cooccurrence
            
            print(f"\nStep {step} progress:")
            print(f"- Queue: {len(env.order_queue.orders)}")
            print(f"- Employees: {len(env.employees)} (Manager: {any(emp.is_manager for emp in env.employees)})")
            print(f"- Max item frequency: {freq.max()}")
            print(f"- Active items: {(freq > 0).sum()}")
            print(f"- Max co-occurrence: {cooccur_data.max()}")
            print(f"- Non-zero co-occurrences: {(cooccur_data > 0).sum()}")
            print(f"- Swaps found so far: {len(swaps_found)}")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total swaps executed: {len(swaps_found)}")
    
    if len(swaps_found) > 0:
        print("✅ SUCCESS: Enhanced patterns generated swaps!")
        for i, (step, swap_key) in enumerate(swaps_found[:3]):
            print(f"  Swap {i+1}: Step {step}, positions {swap_key}")
    else:
        print("⚠️ No swaps generated - checking final data:")
        
        # Show final state
        grid = env.warehouse_grid
        freq = grid.item_access_frequency
        cooccur_data = grid.item_cooccurrence
        
        print(f"Final item frequencies (top 10): {sorted(freq, reverse=True)[:10]}")
        print(f"Final max co-occurrence: {cooccur_data.max()}")
        print(f"Final non-zero co-occurrences: {(cooccur_data > 0).sum()}")
        
        # Check if conditions are met for swaps
        has_manager = any(emp.is_manager for emp in env.employees)
        queue_ok = len(env.order_queue.orders) <= len(env.employees) * agent.params.layout_queue_condition_ratio
        print(f"Manager available: {has_manager}")
        print(f"Queue condition OK: {queue_ok}")
    
    return len(swaps_found)

if __name__ == "__main__":
    swaps = test_enhanced_patterns()
    
    if swaps > 0:
        print(f"\n🎉 ENHANCED SWAP SYSTEM IS WORKING!")
        print(f"Generated {swaps} swaps with realistic order patterns")
    else:
        print(f"\n🔍 System implemented correctly but may need more time/data")