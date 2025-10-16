#!/usr/bin/env python3
"""
Debug script to check why swaps aren't being generated
"""

from environment.warehouse_env import WarehouseEnv
from agents.standardized_agents import create_fixed_hiring_agent

def debug_swap_generation():
    print("=== Debug Swap Generation ===")
    
    env = WarehouseEnv(grid_width=8, grid_height=8, max_employees=10, order_arrival_rate=0.6)
    agent = create_fixed_hiring_agent(env)
    
    # Force aggressive swapping settings
    agent.params.layout_strategy = "moderate"
    agent.params.layout_optimization_interval = 10  # Very frequent
    agent.params.ev_threshold_for_swaps = 1.0  # Very low threshold
    agent.params.layout_queue_condition_ratio = 100.0  # Almost never blocked by queue
    agent.params.manager_hiring_enabled = True
    agent.params.manager_hire_timing = "immediate"
    agent.params.profit_threshold_for_hiring = 0  # Hire immediately
    
    obs = env.reset()
    
    print(f"Environment setup:")
    print(f"- Grid: {env.grid_width}x{env.grid_height}")
    print(f"- Max employees: {env.max_employees}")
    print(f"- Layout strategy: {agent.params.layout_strategy}")
    print(f"- Optimization interval: {agent.params.layout_optimization_interval}")
    print(f"- EV threshold: {agent.params.ev_threshold_for_swaps}")
    
    # Run for several steps to build up co-occurrence data
    for step in range(200):
        obs = env._get_observation()
        action = agent.get_action(obs)
        
        # Debug layout optimization conditions
        if step % 20 == 0:
            queue_length = len(env.order_queue.orders)
            num_employees = len(env.employees)
            has_manager = any(emp.is_manager for emp in env.employees)
            
            print(f"\nStep {step} - Layout Optimization Check:")
            print(f"- Queue length: {queue_length}")
            print(f"- Employees: {num_employees}")
            print(f"- Has manager: {has_manager}")
            print(f"- Queue condition ratio check: {queue_length} <= {num_employees * agent.params.layout_queue_condition_ratio}")
            print(f"- Time since last optimization: {step - agent.last_layout_optimization}")
            
            if has_manager:
                manager = next(emp for emp in env.employees if emp.is_manager)
                print(f"- Manager state: {manager.state}")
                print(f"- Manager idle: {manager.state.name == 'IDLE'}")
            
            # Check if we should attempt layout optimization
            should_optimize = (
                step - agent.last_layout_optimization >= agent.params.layout_optimization_interval and
                queue_length <= num_employees * agent.params.layout_queue_condition_ratio and
                has_manager and
                any(emp.is_manager and emp.state.name == 'IDLE' for emp in env.employees)
            )
            print(f"- Should optimize: {should_optimize}")
            
            # Manually call the enhanced layout action to see what happens
            if should_optimize:
                print("Manually testing layout action...")
                layout_result = agent._get_enhanced_layout_action(step)
                print(f"Layout action result: {layout_result}")
                
                if layout_result is None:
                    print("No swap found - checking why...")
                    
                    # Check co-occurrence data
                    grid = env.warehouse_grid
                    cooccur = grid.item_cooccurrence
                    print(f"Co-occurrence matrix shape: {cooccur.shape}")
                    print(f"Max co-occurrence value: {cooccur.max()}")
                    print(f"Non-zero co-occurrences: {(cooccur > 0).sum()}")
                    
                    # Check item frequencies
                    if hasattr(grid, 'item_access_frequency'):
                        freq = grid.item_access_frequency
                        print(f"Item access frequencies: {freq}")
                        print(f"Max frequency: {freq.max()}")
                    else:
                        print("No item access frequency data")
        
        env.step(action)
        
        # Stop if we see a swap
        layout_swap = action.get('layout_swap', [0, 0])
        if layout_swap != [0, 0]:
            print(f"\n🎉 SWAP FOUND at step {step}: {layout_swap}")
            break
    
    print(f"\nFinal state:")
    print(f"- Employees: {len(env.employees)}")
    print(f"- Queue: {len(env.order_queue.orders)}")
    print(f"- Profit: ${env.cumulative_profit:.0f}")

if __name__ == "__main__":
    debug_swap_generation()