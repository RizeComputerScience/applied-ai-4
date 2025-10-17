#!/usr/bin/env python3
"""
Quick performance test to show students how poorly the skeleton RL agent performs
compared to baseline agents.
"""

from environment.warehouse_env import WarehouseEnv
from agents.baselines import get_baseline_agents

def test_agent_performance(agent_name, episodes=3):
    """Test an agent's performance over multiple episodes"""
    env = WarehouseEnv(episode_length=1000, render_mode=None)  # Shorter episodes for testing
    agents = get_baseline_agents(env)
    
    if agent_name not in agents:
        print(f"Agent {agent_name} not found!")
        return
    
    agent = agents[agent_name]
    
    profits = []
    completion_rates = []
    orders_completed = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        agent.reset()
        
        episode_reward = 0
        
        while True:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        profits.append(info['profit'])
        completion_rates.append(info['completion_rate'])
        orders_completed.append(info['orders_completed'])
    
    avg_profit = sum(profits) / len(profits)
    avg_completion = sum(completion_rates) / len(completion_rates)
    avg_orders = sum(orders_completed) / len(orders_completed)
    
    print(f"\n{agent_name} Performance ({episodes} episodes):")
    print(f"  Average Profit: ${avg_profit:.2f}")
    print(f"  Average Completion Rate: {avg_completion:.1%}")
    print(f"  Average Orders Completed: {avg_orders:.1f}")
    
    return avg_profit, avg_completion, avg_orders

def main():
    print("=== Warehouse Agent Performance Comparison ===")
    print("Testing how poorly the skeleton RL agent performs...")
    
    # Test skeleton RL agent
    skeleton_results = test_agent_performance("skeleton_rl", episodes=2)
    
    # Test different baseline agents
    print("\n" + "="*50)
    greedy_results = test_agent_performance("greedy_std", episodes=2)
    intelligent_results = test_agent_performance("intelligent_hiring", episodes=2) 
    queue_results = test_agent_performance("intelligent_queue", episodes=2)
    distance_results = test_agent_performance("distance_based", episodes=2)
    aggressive_results = test_agent_performance("aggressive_swap", episodes=2)
    
    print(f"\n=== PERFORMANCE RANKING ===")
    agents = [
        ("Skeleton RL", skeleton_results),
        ("Greedy (Bad)", greedy_results),
        ("Intelligent Hiring", intelligent_results),
        ("Intelligent Queue", queue_results),
        ("Distance Based", distance_results),
        ("Aggressive Swap", aggressive_results)
    ]
    
    # Sort by profit
    agents.sort(key=lambda x: x[1][0], reverse=True)
    
    for i, (name, results) in enumerate(agents, 1):
        print(f"{i}. {name:18s} | ${results[0]:8.0f} profit | {results[1]:6.1%} completion | {results[2]:4.0f} orders")
    
    # Show gap to best performer
    best_profit = agents[0][1][0]
    best_completion = agents[0][1][1]
    skeleton_profit = skeleton_results[0]
    skeleton_completion = skeleton_results[1]
    
    print(f"\n=== STUDENT TARGET ===")
    print(f"Best Agent Performance: ${best_profit:.0f} profit, {best_completion:.1%} completion")
    print(f"Skeleton RL Performance: ${skeleton_profit:.0f} profit, {skeleton_completion:.1%} completion")
    print(f"")
    print(f"IMPROVEMENT NEEDED:")
    print(f"  Profit Gap: ${best_profit - skeleton_profit:,.0f}")
    print(f"  Completion Gap: {best_completion - skeleton_completion:.1%}")
    print(f"")
    print(f"Students must improve RL agent to beat the best baseline!")

if __name__ == "__main__":
    main()