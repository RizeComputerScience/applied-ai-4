#!/usr/bin/env python3
"""
Multi-Objective Optimization Demo

Demonstrates Pareto frontier tradeoffs between profit maximization and service quality
in the warehouse environment. Shows clear intersection points and optimization choices.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
from typing import Dict, List, Tuple
import time

# Suppress warnings
warnings.filterwarnings("ignore")

from environment.warehouse_env import WarehouseEnv
from agents.multi_objective_agent import create_multi_objective_agents

def run_multi_objective_experiment(episodes: int = 5, episode_length: int = 2000) -> Dict:
    """Run multi-objective optimization experiment"""
    
    print("🎯 Multi-Objective Optimization Demo")
    print("=====================================")
    print(f"Running {episodes} episodes per configuration...")
    print("Objectives: Profit Maximization vs Service Quality")
    print("💰 New Feature: Wage-based productivity tradeoffs")
    print("   • Low wage ($0.20): Slow workers, low cost")
    print("   • Medium wage ($0.50): Balanced speed/cost")  
    print("   • High wage ($0.80): Fast workers, high cost")
    print()
    
    # Create environment with controlled parameters for clear tradeoffs
    env = WarehouseEnv(
        episode_length=episode_length,
        order_arrival_rate=0.40,  # Moderate pressure to allow strategy differences
        initial_employees=2,      # Start lean
        max_employees=12,         # Limit staffing options
        employee_salary=0.5,      # Medium baseline wage
        grid_width=15,
        grid_height=15,
        num_item_types=30
    )
    
    # Create multi-objective agents
    agents = create_multi_objective_agents(env)
    
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"Testing {agent_name}...", end='', flush=True)
        
        episode_results = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            agent.reset()
            
            episode_profit = 0
            episode_service_metrics = []
            timestep = 0
            
            while timestep < episode_length:
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track metrics every 100 timesteps for stability
                if timestep % 100 == 0:
                    current_profit = info.get('profit', 0)
                    completion_rate = info.get('completion_rate', 0)
                    episode_service_metrics.append(completion_rate)
                
                timestep += 1
                
                if terminated or truncated:
                    break
            
            # Record episode results
            final_profit = info.get('profit', 0)
            avg_service_rate = np.mean(episode_service_metrics) if episode_service_metrics else 0
            
            episode_results.append({
                'profit': final_profit,
                'service_rate': avg_service_rate,
                'completion_rate': info.get('completion_rate', 0),
                'orders_completed': info.get('orders_completed', 0),
                'orders_cancelled': info.get('orders_cancelled', 0)
            })
            
            # Progress indicator
            percent = int(((episode + 1) / episodes) * 100)
            print(f'\r{agent_name}... [{percent}%]', end='', flush=True)
        
        # Calculate statistics
        profits = [r['profit'] for r in episode_results]
        service_rates = [r['service_rate'] for r in episode_results]
        completion_rates = [r['completion_rate'] for r in episode_results]
        
        results[agent_name] = {
            'avg_profit': np.mean(profits),
            'avg_service_rate': np.mean(service_rates),
            'avg_completion_rate': np.mean(completion_rates),
            'profit_std': np.std(profits),
            'service_std': np.std(service_rates),
            'profit_weight': agent.profit_weight,
            'service_weight': agent.service_weight,
            'all_results': episode_results
        }
        
        print(f'\r{agent_name}... Done! ')
        print(f"  Avg Profit: ${results[agent_name]['avg_profit']:.1f}, "
              f"Service Rate: {results[agent_name]['avg_completion_rate']:.1%}")
    
    env.close()
    return results

def plot_pareto_frontier(results: Dict):
    """Create Pareto frontier visualization"""
    
    # Extract data for plotting
    configurations = []
    for agent_name, data in results.items():
        configurations.append({
            'name': agent_name,
            'profit': data['avg_profit'],
            'service': data['avg_completion_rate'],
            'profit_weight': data['profit_weight'],
            'service_weight': data['service_weight'],
            'profit_std': data['profit_std'],
            'service_std': data['service_std']
        })
    
    # Sort by profit weight for better visualization
    configurations.sort(key=lambda x: x['profit_weight'], reverse=True)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Extract coordinates
    profits = [c['profit'] for c in configurations]
    services = [c['service'] for c in configurations]
    profit_weights = [c['profit_weight'] for c in configurations]
    
    # Create color map based on profit weight
    colors = plt.cm.RdYlBu_r(profit_weights)
    
    # Plot points with error bars
    for i, config in enumerate(configurations):
        plt.scatter(config['service'], config['profit'], 
                   c=[colors[i]], s=150, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add error bars
        plt.errorbar(config['service'], config['profit'], 
                    xerr=config['service_std'], yerr=config['profit_std'],
                    color='gray', alpha=0.5, capsize=3)
        
        # Label points
        label = f"P:{config['profit_weight']:.1f}"
        plt.annotate(label, (config['service'], config['profit']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Draw Pareto frontier line
    frontier_x = [c['service'] for c in configurations]
    frontier_y = [c['profit'] for c in configurations]
    plt.plot(frontier_x, frontier_y, 'k--', alpha=0.6, linewidth=2, label='Pareto Frontier')
    
    # Highlight intersection/knee point
    if len(configurations) >= 3:
        knee_idx = len(configurations) // 2  # Middle configuration
        knee_config = configurations[knee_idx]
        plt.scatter(knee_config['service'], knee_config['profit'], 
                   c='red', s=300, marker='*', edgecolors='black', linewidth=2,
                   label='Balanced Solution', zorder=5)
    
    # Formatting
    plt.xlabel('Service Quality (Completion Rate)', fontsize=12, fontweight='bold')
    plt.ylabel('Profit ($)', fontsize=12, fontweight='bold')
    plt.title('Multi-Objective Optimization: Profit vs Service Quality\nPareto Frontier', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=10)
    
    # Add objective preference annotations
    plt.text(0.02, 0.98, 'Service-Focused\n(High completion rate,\nLower profit)', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
             verticalalignment='top')
    
    plt.text(0.98, 0.02, 'Profit-Focused\n(Higher profit,\nLower service)', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
             verticalalignment='bottom', horizontalalignment='right')
    
    # Format axes
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('multi_objective_pareto_frontier.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Pareto frontier plot saved as 'multi_objective_pareto_frontier.png'")
    
    plt.show()

def print_detailed_results(results: Dict):
    """Print detailed analysis of results"""
    
    print("\n" + "="*60)
    print("DETAILED MULTI-OBJECTIVE ANALYSIS")
    print("="*60)
    
    # Sort by profit weight
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['profit_weight'], reverse=True)
    
    print(f"{'Configuration':<25} {'Profit':<12} {'Service':<12} {'Tradeoff':<15}")
    print("-" * 70)
    
    for agent_name, data in sorted_results:
        profit_focus = "Profit" if data['profit_weight'] > 0.6 else "Service" if data['service_weight'] > 0.6 else "Balanced"
        
        print(f"{agent_name:<25} "
              f"${data['avg_profit']:>8.1f} "
              f"{data['avg_completion_rate']:>10.1%} "
              f"{profit_focus:>13}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    # Find extreme points
    max_profit_config = max(sorted_results, key=lambda x: x[1]['avg_profit'])
    max_service_config = max(sorted_results, key=lambda x: x[1]['avg_completion_rate'])
    
    print(f"🏆 Highest Profit: {max_profit_config[0]}")
    print(f"   Profit: ${max_profit_config[1]['avg_profit']:.1f}, Service: {max_profit_config[1]['avg_completion_rate']:.1%}")
    
    print(f"\n⭐ Best Service: {max_service_config[0]}")
    print(f"   Profit: ${max_service_config[1]['avg_profit']:.1f}, Service: {max_service_config[1]['avg_completion_rate']:.1%}")
    
    # Calculate tradeoff ratios
    profit_range = max_profit_config[1]['avg_profit'] - max_service_config[1]['avg_profit']
    service_range = max_service_config[1]['avg_completion_rate'] - max_profit_config[1]['avg_completion_rate']
    
    if service_range > 0:
        tradeoff_ratio = profit_range / (service_range * 100)  # Profit per percentage point of service
        print(f"\n📈 Tradeoff Rate: ${tradeoff_ratio:.1f} profit per percentage point of service quality")
    
    # Find balanced solution
    balanced_configs = [x for x in sorted_results if abs(x[1]['profit_weight'] - 0.5) < 0.1]
    if balanced_configs:
        balanced = balanced_configs[0]
        print(f"\n⚖️  Balanced Solution: {balanced[0]}")
        print(f"   Offers {balanced[1]['avg_completion_rate']:.1%} service at ${balanced[1]['avg_profit']:.1f} profit")

def main():
    """Run multi-objective optimization demo"""
    
    print("Starting Multi-Objective Warehouse Optimization Demo...")
    
    # Run experiment
    results = run_multi_objective_experiment(episodes=3, episode_length=1500)
    
    # Display results
    print_detailed_results(results)
    
    # Create visualization
    plot_pareto_frontier(results)
    
    print("\n✅ Multi-objective optimization demo completed!")
    print("The graph shows the Pareto frontier - the set of optimal tradeoffs")
    print("between profit maximization and service quality.")
    print("\nKey takeaways:")
    print("• Each point represents a different optimization strategy")
    print("• No single solution dominates all others")
    print("• The choice depends on business priorities and constraints")
    print("• The red star shows a balanced compromise solution")

if __name__ == "__main__":
    main()