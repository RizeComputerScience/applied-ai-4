#!/usr/bin/env python3
"""
Test script to compare economic performance of different agents with higher salary costs.
"""

from environment.warehouse_env import WarehouseEnv
from agents.standardized_agents import (
    create_greedy_agent, 
    create_fixed_hiring_agent, 
    create_intelligent_hiring_agent,
    create_aggressive_swap_agent
)

def test_agent_performance(agent_name, agent_creator, steps=1000):
    """Test an agent's economic performance"""
    env = WarehouseEnv(
        grid_width=12, 
        grid_height=12, 
        max_employees=15, 
        order_arrival_rate=0.8,
        employee_salary=1.5  # Higher cost
    )
    
    agent = agent_creator(env)
    obs = env.reset()
    
    print(f"\n=== Testing {agent_name} ===")
    print(f"Employee salary: ${env.employee_salary}/timestep")
    print(f"Manager salary: $2.0/timestep")
    
    # Track metrics
    initial_profit = env.cumulative_profit
    initial_employees = len(env.employees)
    swaps_completed = 0
    orders_completed = 0
    max_employees = 0
    manager_hired_step = None
    
    for step in range(steps):
        obs = env._get_observation()
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Track metrics
        current_employees = len(env.employees)
        max_employees = max(max_employees, current_employees)
        
        # Check for manager hiring
        if manager_hired_step is None:
            for emp in env.employees:
                if emp.is_manager:
                    manager_hired_step = step
                    break
        
        # Check for completed swaps
        if hasattr(env, 'last_swap_info') and env.last_swap_info:
            swaps_completed += 1
            env.last_swap_info = None
        
        # Track orders completed
        orders_completed = info.get('total_completed_orders', 0)
        
        if done or truncated:
            break
    
    # Calculate final metrics
    final_profit = env.cumulative_profit
    profit_change = final_profit - initial_profit
    final_employees = len(env.employees)
    total_costs = env.total_costs
    revenue = env.total_revenue
    
    print(f"Results after {step + 1} steps:")
    print(f"  Profit: ${profit_change:.2f} (${initial_profit:.2f} -> ${final_profit:.2f})")
    print(f"  Revenue: ${revenue:.2f}")
    print(f"  Costs: ${total_costs:.2f}")
    print(f"  Profit Margin: {(profit_change/revenue*100 if revenue > 0 else 0):.1f}%")
    print(f"  Employees: {initial_employees} -> {final_employees} (max: {max_employees})")
    print(f"  Manager hired: {'Step ' + str(manager_hired_step) if manager_hired_step else 'No'}")
    print(f"  Orders completed: {orders_completed}")
    print(f"  Layout swaps: {swaps_completed}")
    print(f"  Efficiency: ${profit_change/orders_completed:.2f} profit per order" if orders_completed > 0 else "  No orders completed")
    
    return {
        'agent_name': agent_name,
        'profit_change': profit_change,
        'revenue': revenue,
        'costs': total_costs,
        'final_employees': final_employees,
        'max_employees': max_employees,
        'manager_hired': manager_hired_step is not None,
        'orders_completed': orders_completed,
        'swaps_completed': swaps_completed,
        'profit_per_order': profit_change/orders_completed if orders_completed > 0 else 0
    }

def main():
    print("=== Economic Impact Test with Higher Salaries ===")
    print("Employee cost: $1.50/timestep (15x increase from $0.10)")
    print("Manager cost: $2.00/timestep")
    
    agents_to_test = [
        ("Greedy", create_greedy_agent),
        ("FixedHiring", create_fixed_hiring_agent),
        ("IntelligentHiring", create_intelligent_hiring_agent),
        ("AggressiveSwap", create_aggressive_swap_agent)
    ]
    
    results = []
    
    for agent_name, agent_creator in agents_to_test:
        result = test_agent_performance(agent_name, agent_creator, steps=1500)
        results.append(result)
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    # Sort by profit
    results.sort(key=lambda x: x['profit_change'], reverse=True)
    
    print(f"{'Agent':<15} {'Profit':<10} {'Orders':<8} {'Swaps':<6} {'Employees':<10} {'Manager':<8} {'Efficiency':<12}")
    print("-" * 80)
    
    for r in results:
        manager_status = "Yes" if r['manager_hired'] else "No"
        efficiency = f"${r['profit_per_order']:.2f}/ord" if r['profit_per_order'] > 0 else "N/A"
        print(f"{r['agent_name']:<15} ${r['profit_change']:<9.2f} {r['orders_completed']:<8} {r['swaps_completed']:<6} {r['final_employees']:<10} {manager_status:<8} {efficiency:<12}")
    
    print(f"\nBest performing agent: {results[0]['agent_name']} with ${results[0]['profit_change']:.2f} profit")

if __name__ == "__main__":
    main()