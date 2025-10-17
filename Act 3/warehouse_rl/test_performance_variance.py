#!/usr/bin/env python3
"""
Test performance variance across multiple runs to understand reliability
"""

import numpy as np
from environment.warehouse_env import WarehouseEnv
from agents.baselines import get_baseline_agents

def test_agent_variance(agent_name, episodes=10):
    """Test an agent's performance variance over multiple episodes"""
    env = WarehouseEnv(episode_length=1000, render_mode=None)
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
        
        while True:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        profits.append(info['profit'])
        completion_rates.append(info['completion_rate'])
        orders_completed.append(info['orders_completed'])
    
    # Calculate statistics
    profit_mean = np.mean(profits)
    profit_std = np.std(profits)
    profit_ci = 1.96 * profit_std / np.sqrt(episodes)  # 95% CI
    
    completion_mean = np.mean(completion_rates)
    completion_std = np.std(completion_rates)
    completion_ci = 1.96 * completion_std / np.sqrt(episodes)
    
    orders_mean = np.mean(orders_completed)
    orders_std = np.std(orders_completed)
    orders_ci = 1.96 * orders_std / np.sqrt(episodes)
    
    print(f"\n{agent_name} Performance ({episodes} episodes):")
    print(f"  Profit: ${profit_mean:.0f} ± ${profit_ci:.0f} (95% CI)")
    print(f"  Range: ${min(profits):.0f} to ${max(profits):.0f}")
    print(f"  Std Dev: ${profit_std:.0f}")
    print(f"  ")
    print(f"  Completion: {completion_mean:.1%} ± {completion_ci:.1%} (95% CI)")
    print(f"  Range: {min(completion_rates):.1%} to {max(completion_rates):.1%}")
    print(f"  ")
    print(f"  Orders: {orders_mean:.1f} ± {orders_ci:.1f} (95% CI)")
    print(f"  Range: {min(orders_completed)} to {max(orders_completed)}")
    
    return {
        'profits': profits,
        'profit_mean': profit_mean,
        'profit_ci': profit_ci,
        'completion_rates': completion_rates,
        'completion_mean': completion_mean,
        'completion_ci': completion_ci
    }

def compare_variance():
    """Compare variance across key agents"""
    print("=== Performance Variance Analysis ===")
    print("Testing performance stability over 10 episodes...")
    
    agents_to_test = ['skeleton_rl', 'greedy_std', 'distance_based', 'intelligent_hiring']
    results = {}
    
    for agent in agents_to_test:
        results[agent] = test_agent_variance(agent, episodes=10)
    
    print(f"\n=== RELIABILITY RANKING (by profit variance) ===")
    # Sort by coefficient of variation (std/mean) - lower is more reliable
    variance_ranking = []
    for agent, data in results.items():
        cv = (data['profit_ci'] * 2) / abs(data['profit_mean'])  # CI range as % of mean
        variance_ranking.append((agent, data['profit_mean'], data['profit_ci'], cv))
    
    variance_ranking.sort(key=lambda x: x[3])  # Sort by CV
    
    for i, (agent, mean, ci, cv) in enumerate(variance_ranking, 1):
        reliability = "Highly Reliable" if cv < 0.1 else "Moderately Reliable" if cv < 0.3 else "Highly Variable"
        print(f"{i}. {agent:18s} | ${mean:6.0f} ± ${ci:4.0f} | CV: {cv:.1%} | {reliability}")
    
    print(f"\n=== KEY INSIGHTS ===")
    skeleton_data = results['skeleton_rl']
    best_deterministic = min(((agent, data) for agent, data in results.items() 
                           if agent != 'skeleton_rl'), key=lambda x: x[1]['profit_ci'] * 2 / abs(x[1]['profit_mean']))
    
    print(f"Skeleton RL Variability: ${skeleton_data['profit_ci']:.0f} (95% CI)")
    print(f"Most Stable Agent: {best_deterministic[0]} with ${best_deterministic[1]['profit_ci']:.0f} (95% CI)")
    
    skeleton_cv = (skeleton_data['profit_ci'] * 2) / abs(skeleton_data['profit_mean'])
    if skeleton_cv > 0.3:
        print(f"⚠️  Skeleton RL is HIGHLY VARIABLE (CV: {skeleton_cv:.1%})")
        print(f"   This makes it hard to measure student improvements reliably!")
    else:
        print(f"✅ Skeleton RL variability is acceptable (CV: {skeleton_cv:.1%})")

if __name__ == "__main__":
    compare_variance()