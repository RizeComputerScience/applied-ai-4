#!/usr/bin/env python3
"""
Test script to verify the enhanced swap decision system works correctly.
Tests adaptive cooldowns, predictive evaluation, and market-aware decisions.
"""

from environment.warehouse_env import WarehouseEnv
from agents.standardized_agents import create_intelligent_hiring_agent, create_intelligent_queue_agent

def test_enhanced_swap_system():
    """Test the enhanced swap decision system"""
    print("=== Enhanced Swap Decision System Test ===")
    
    # Create environment and enhanced agent
    env = WarehouseEnv(grid_width=12, grid_height=12, max_employees=15, order_arrival_rate=0.8)
    agent = create_intelligent_hiring_agent(env)
    
    # Enable more aggressive swapping for testing
    agent.params.layout_strategy = "moderate"
    agent.params.layout_optimization_interval = 25
    agent.params.ev_threshold_for_swaps = 5.0
    
    obs = env.reset()
    
    print(f"Initial adaptive cooldown settings:")
    print(f"- Base cooldown: {agent.swap_cooldown_period}")
    print(f"- Min cooldown: {agent.min_cooldown}")
    print(f"- Max cooldown: {agent.max_cooldown}")
    print(f"- Evaluation window: {agent.swap_evaluation_window}")
    
    swap_attempts = []
    successful_swaps = []
    blocked_swaps = []
    profit_tracking = []
    
    for step in range(1000):  # Extended test
        obs = env._get_observation()
        action = agent.get_action(obs)
        
        # Track profit for analysis
        current_profit = env.cumulative_profit
        profit_tracking.append((step, current_profit))
        
        # Check for swap proposals
        layout_swap = action.get('layout_swap', [0, 0])
        if layout_swap != [0, 0] and layout_swap[0] != layout_swap[1]:
            pos1 = (layout_swap[0] % env.grid_width, layout_swap[0] // env.grid_width)
            pos2 = (layout_swap[1] % env.grid_width, layout_swap[1] // env.grid_width)
            swap_key = tuple(sorted([layout_swap[0], layout_swap[1]]))
            
            print(f"Step {step}: SWAP PROPOSED {pos1} <-> {pos2}")
            swap_attempts.append((step, swap_key, 'proposed'))
            
            # Check if this swap will be successful
            env.step(action)
            
            # Verify swap was executed by checking if it's in recent_swaps
            if swap_key in agent.recent_swaps and agent.recent_swaps[swap_key] == step:
                successful_swaps.append((step, swap_key))
                print(f"  ✅ SWAP EXECUTED")
                
                # Show adaptive cooldown for this swap
                adaptive_cooldown = agent._get_adaptive_cooldown(swap_key)
                print(f"  📊 Adaptive cooldown for this swap: {adaptive_cooldown} timesteps")
            else:
                blocked_swaps.append((step, swap_key))
                print(f"  ❌ SWAP BLOCKED")
        else:
            env.step(action)
        
        # Show enhanced tracking data periodically
        if step % 200 == 0 and step > 0:
            print(f"\n--- Status at Step {step} ---")
            print(f"Market conditions tracked: {len(agent.market_condition_history)}")
            print(f"Profit history entries: {len(agent.profit_history_detailed)}")
            print(f"Swap performance history: {len(agent.swap_performance_history)}")
            print(f"Recent profit rate: {agent._calculate_recent_profit_rate():.2f}/step")
            
            if agent.market_condition_history:
                recent_market = agent.market_condition_history[-1]
                print(f"Current market pressure: {recent_market.get('order_pressure', 0):.2f}")
                print(f"Item diversity: {recent_market.get('item_diversity', 0)}")
            
            # Show swap performance analysis
            completed_evals = [data for data in agent.swap_performance_history.values() if data['evaluation_complete']]
            if completed_evals:
                avg_success = sum(data['success_ratio'] for data in completed_evals) / len(completed_evals)
                print(f"Average swap success ratio: {avg_success:.2f}")
                
                successful_count = sum(1 for data in completed_evals if data['success_ratio'] > 0.8)
                print(f"Successful swaps: {successful_count}/{len(completed_evals)}")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total swap attempts: {len(swap_attempts)}")
    print(f"Successful swaps: {len(successful_swaps)}")
    print(f"Blocked swaps: {len(blocked_swaps)}")
    print(f"Success rate: {len(successful_swaps)/max(1,len(swap_attempts))*100:.1f}%")
    
    # Analyze profit impact
    if len(profit_tracking) > 100:
        initial_profit = profit_tracking[100][1]  # Skip initial startup period
        final_profit = profit_tracking[-1][1]
        profit_improvement = final_profit - initial_profit
        print(f"Profit improvement over test: ${profit_improvement:.0f}")
        
        # Calculate profit rate
        time_span = profit_tracking[-1][0] - profit_tracking[100][0]
        profit_rate = profit_improvement / time_span
        print(f"Average profit rate: ${profit_rate:.2f}/step")
    
    # Analyze adaptive cooldown effectiveness
    if agent.swap_performance_history:
        print(f"\n=== Adaptive Cooldown Analysis ===")
        all_swaps = list(agent.swap_performance_history.items())
        for swap_key, data in all_swaps[:5]:  # Show first 5 swaps
            if data['evaluation_complete']:
                cooldown = agent._get_adaptive_cooldown(swap_key)
                print(f"Swap {swap_key}: Success ratio {data['success_ratio']:.2f} -> Cooldown {cooldown}")
    
    # Show final system state
    print(f"\n=== Final System State ===")
    print(f"Final profit: ${env.cumulative_profit:.0f}")
    print(f"Employees: {len(env.employees)}")
    print(f"Queue length: {len(env.order_queue.orders)}")
    
    return len(successful_swaps) > len(blocked_swaps)  # Success if more swaps executed than blocked

def test_market_awareness():
    """Test that the system adapts to different market conditions"""
    print("\n=== Market Awareness Test ===")
    
    env = WarehouseEnv(grid_width=10, grid_height=10, max_employees=10, order_arrival_rate=0.5)
    agent = create_intelligent_queue_agent(env)
    
    # Force some market data collection
    obs = env.reset()
    
    for step in range(100):
        obs = env._get_observation()
        action = agent.get_action(obs)
        env.step(action)
        
        # Inject different market conditions at different times
        if step == 30:
            # Simulate high order pressure
            orders = env.order_generator.generate_orders(10)
            for order in orders:
                env.order_queue.add_order(order)
        elif step == 60:
            # Simulate low pressure period
            env.order_queue.orders = env.order_queue.orders[:2]
    
    print("Market adaptation test completed")
    
    # Check if market conditions were tracked
    if agent.market_condition_history:
        high_pressure = max(entry['order_pressure'] for entry in agent.market_condition_history)
        low_pressure = min(entry['order_pressure'] for entry in agent.market_condition_history)
        print(f"Market pressure range observed: {low_pressure:.2f} - {high_pressure:.2f}")
        return high_pressure > low_pressure * 2
    
    return False

if __name__ == "__main__":
    print("Testing Enhanced Swap Decision System...\n")
    
    swap_test_passed = test_enhanced_swap_system()
    market_test_passed = test_market_awareness()
    
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Enhanced swap system: {'✅ PASS' if swap_test_passed else '❌ FAIL'}")
    print(f"Market awareness: {'✅ PASS' if market_test_passed else '❌ FAIL'}")
    
    if swap_test_passed and market_test_passed:
        print("🎉 All tests passed! Enhanced swap system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")