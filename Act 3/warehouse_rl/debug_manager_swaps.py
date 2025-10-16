#!/usr/bin/env python3
"""
Debug script to investigate why managers aren't performing inventory swaps.
Tracks manager state, swap conditions, and decision logic.
"""

import numpy as np
from environment.warehouse_env import WarehouseEnv
from agents.standardized_agents import create_fixed_hiring_agent, create_intelligent_hiring_agent, create_aggressive_swap_agent
from environment.employee import EmployeeState

class ManagerSwapDebugger:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.debug_log = []
        self.swap_attempts = 0
        self.swap_successes = 0
        
    def log_debug(self, message, category="INFO"):
        timestep = self.env.current_timestep
        self.debug_log.append(f"[{timestep:04d}] [{category}] {message}")
        print(f"[{timestep:04d}] [{category}] {message}")
    
    def check_manager_status(self):
        """Check current manager status and log details"""
        managers = [emp for emp in self.env.employees if emp.is_manager]
        
        if not managers:
            self.log_debug("NO MANAGERS FOUND", "WARNING")
            return False
        
        for manager in managers:
            self.log_debug(f"Manager {manager.id}: State={manager.state.name}, Pos={manager.position}", "MANAGER")
            if manager.state != EmployeeState.IDLE:
                self.log_debug(f"  - Not idle: task_timer={getattr(manager, 'task_timer', 'N/A')}", "MANAGER")
                if hasattr(manager, 'relocation_task') and manager.relocation_task:
                    task = manager.relocation_task
                    self.log_debug(f"  - Relocation task: {task[0]} -> {task[1]}, stage={task[2]}", "MANAGER")
        
        return any(manager.state == EmployeeState.IDLE for manager in managers)
    
    def check_layout_conditions(self):
        """Check if conditions are met for layout optimization"""
        # Check if layout strategy allows swaps
        if self.agent.params.layout_strategy == "none":
            self.log_debug("Layout strategy is 'none' - no swaps allowed", "CONDITION")
            return False
        
        # Check timing interval
        current_time = self.env.current_timestep
        last_optimization = self.agent.last_layout_optimization
        interval = self.agent.params.layout_optimization_interval
        
        if current_time - last_optimization < interval:
            self.log_debug(f"Layout interval not met: {current_time - last_optimization} < {interval}", "CONDITION")
            return False
        
        # Check queue condition
        queue_length = len(self.env.order_queue.orders)
        num_employees = len(self.env.employees)
        queue_ratio = self.agent.params.layout_queue_condition_ratio
        
        if queue_length > num_employees * queue_ratio:
            self.log_debug(f"Queue too long for optimization: {queue_length} > {num_employees} * {queue_ratio}", "CONDITION")
            return False
        
        self.log_debug(f"Layout conditions MET: interval={current_time - last_optimization}, queue={queue_length}, employees={num_employees}", "CONDITION")
        return True
    
    def check_cooccurrence_data(self):
        """Check if there's meaningful cooccurrence data for swaps"""
        cooccurrence = self.env.warehouse_grid.item_cooccurrence
        
        # Find pairs with significant cooccurrence
        significant_pairs = []
        for i in range(cooccurrence.shape[0]):
            for j in range(i + 1, cooccurrence.shape[1]):
                if cooccurrence[i, j] > 2:  # Threshold from agent code (updated)
                    significant_pairs.append((i, j, cooccurrence[i, j]))
        
        # Also check frequency data as fallback
        frequency_data = False
        if hasattr(self.env.warehouse_grid, 'item_access_frequency'):
            freq_array = self.env.warehouse_grid.item_access_frequency
            high_freq_items = [i for i in range(len(freq_array)) if freq_array[i] > 2]
            if high_freq_items:
                frequency_data = True
                self.log_debug(f"Found frequency data: {len(high_freq_items)} items with freq > 2", "FREQUENCY")
                for item in high_freq_items[:5]:  # Show top 5
                    self.log_debug(f"  Item {item}: accessed {freq_array[item]:.1f} times", "FREQUENCY")
        
        if not significant_pairs and not frequency_data:
            self.log_debug("No significant item cooccurrence patterns OR frequency data found", "COOCCUR")
            return False
        
        if significant_pairs:
            self.log_debug(f"Found {len(significant_pairs)} significant cooccurrence pairs:", "COOCCUR")
            for item1, item2, count in significant_pairs[:5]:  # Show top 5
                self.log_debug(f"  Items {item1}-{item2}: {count} cooccurrences", "COOCCUR")
        
        return True
    
    def simulate_swap_calculation(self):
        """Simulate the agent's swap calculation logic"""
        self.log_debug("=== SIMULATING SWAP CALCULATION ===", "SWAP_CALC")
        
        grid = self.env.warehouse_grid
        cooccurrence = grid.item_cooccurrence
        
        best_swap = None
        best_benefit = 0
        calculations_done = 0
        
        # Simple co-occurrence based optimization (from agent code)
        for item1 in range(grid.num_item_types):
            for item2 in range(item1 + 1, grid.num_item_types):
                if cooccurrence[item1, item2] > 2:  # Frequently ordered together (updated)
                    calculations_done += 1
                    
                    item1_locs = grid.find_item_locations(item1)
                    item2_locs = grid.find_item_locations(item2)
                    
                    if item1_locs and item2_locs:
                        current_distance = grid.manhattan_distance(item1_locs[0], item2_locs[0])
                        
                        # Find potential swap partners
                        for other_item in range(grid.num_item_types):
                            if other_item != item1 and other_item != item2:
                                other_locs = grid.find_item_locations(other_item)
                                if other_locs:
                                    new_distance = grid.manhattan_distance(other_locs[0], item2_locs[0])
                                    benefit = (current_distance - new_distance) * cooccurrence[item1, item2]
                                    
                                    if benefit > best_benefit and benefit > self.agent.params.ev_threshold_for_swaps:
                                        best_benefit = benefit
                                        pos1_idx = item1_locs[0][1] * grid.width + item1_locs[0][0]
                                        pos2_idx = other_locs[0][1] * grid.width + other_locs[0][0]
                                        best_swap = [pos1_idx, pos2_idx]
                                        
                                        self.log_debug(f"Found beneficial swap: items {item1}->{other_item}, benefit={benefit:.1f}", "SWAP_CALC")
        
        self.log_debug(f"Swap calculations completed: {calculations_done} pairs checked", "SWAP_CALC")
        self.log_debug(f"EV threshold: {self.agent.params.ev_threshold_for_swaps}", "SWAP_CALC")
        
        if best_swap:
            self.log_debug(f"BEST SWAP FOUND: positions {best_swap}, benefit={best_benefit:.1f}", "SWAP_CALC")
            return best_swap
        else:
            self.log_debug("NO BENEFICIAL SWAPS FOUND", "SWAP_CALC")
            return None
    
    def check_swap_validity(self, pos1_idx, pos2_idx):
        """Check if a proposed swap is valid"""
        grid = self.env.warehouse_grid
        pos1 = (pos1_idx % grid.width, pos1_idx // grid.width)
        pos2 = (pos2_idx % grid.width, pos2_idx // grid.width)
        
        self.log_debug(f"Checking swap validity: {pos1} <-> {pos2}", "VALIDITY")
        
        # Check if positions are different
        if pos1 == pos2:
            self.log_debug("Invalid: same position", "VALIDITY")
            return False
        
        # Check if both positions are storage cells
        from environment.warehouse_grid import CellType
        if (not grid.is_valid_position(pos1[0], pos1[1]) or 
            grid.cell_types[pos1[1], pos1[0]] != CellType.STORAGE.value):
            self.log_debug(f"Invalid: pos1 {pos1} not storage", "VALIDITY")
            return False
        
        if (not grid.is_valid_position(pos2[0], pos2[1]) or 
            grid.cell_types[pos2[1], pos2[0]] != CellType.STORAGE.value):
            self.log_debug(f"Invalid: pos2 {pos2} not storage", "VALIDITY")
            return False
        
        # Check if at least one position has an item
        item1 = grid.get_item_at_position(pos1[0], pos1[1])
        item2 = grid.get_item_at_position(pos2[0], pos2[1])
        
        if item1 is None and item2 is None:
            self.log_debug("Invalid: both positions empty", "VALIDITY")
            return False
        
        self.log_debug(f"Valid swap: {pos1}(item={item1}) <-> {pos2}(item={item2})", "VALIDITY")
        return True
    
    def debug_step(self):
        """Perform one debug step - check all conditions and simulate decisions"""
        self.log_debug(f"=== DEBUG STEP {self.env.current_timestep} ===", "STEP")
        
        # Check basic conditions
        has_idle_manager = self.check_manager_status()
        layout_conditions_met = self.check_layout_conditions()
        has_cooccurrence = self.check_cooccurrence_data()
        
        if not has_idle_manager:
            self.log_debug("BLOCKED: No idle manager available", "BLOCKED")
            return
        
        if not layout_conditions_met:
            self.log_debug("BLOCKED: Layout optimization conditions not met", "BLOCKED")
            return
        
        if not has_cooccurrence:
            self.log_debug("BLOCKED: No significant cooccurrence data", "BLOCKED")
            return
        
        # Simulate swap calculation
        proposed_swap = self.simulate_swap_calculation()
        
        if proposed_swap:
            self.swap_attempts += 1
            if self.check_swap_validity(proposed_swap[0], proposed_swap[1]):
                self.log_debug("SWAP SHOULD EXECUTE", "SUCCESS")
                self.swap_successes += 1
            else:
                self.log_debug("SWAP BLOCKED: Invalid swap", "BLOCKED")
        else:
            self.log_debug("NO SWAP PROPOSED", "NO_ACTION")
    
    def run_debug_simulation(self, steps=1000):
        """Run a debug simulation for specified steps"""
        print(f"\n=== Starting Manager Swap Debug Simulation ===")
        print(f"Agent: {self.agent.name}")
        print(f"Layout strategy: {self.agent.params.layout_strategy}")
        print(f"Manager hiring: {self.agent.params.manager_hiring_enabled}")
        print(f"EV threshold: {self.agent.params.ev_threshold_for_swaps}")
        print("=" * 50)
        
        for step in range(steps):
            obs = self.env._get_observation()
            action = self.agent.get_action(obs)
            
            # Debug before executing action
            if step % 50 == 0:  # Debug every 50 steps
                self.debug_step()
            
            # Execute the action
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Check if any swaps actually occurred
            if hasattr(self.env, 'last_swap_info') and self.env.last_swap_info:
                self.log_debug(f"SWAP EXECUTED: {self.env.last_swap_info}", "EXECUTED")
                self.env.last_swap_info = None  # Clear for next detection
            
            if done or truncated:
                break
        
        # Summary
        print(f"\n=== DEBUG SUMMARY ===")
        print(f"Steps simulated: {step + 1}")
        print(f"Swap attempts analyzed: {self.swap_attempts}")
        print(f"Valid swaps found: {self.swap_successes}")
        print(f"Recent debug log:")
        for line in self.debug_log[-20:]:  # Show last 20 entries
            print(line)

def main():
    # Test with different agents that should perform swaps
    agents_to_test = [
        ("AggressiveSwap", create_aggressive_swap_agent),
        ("FixedHiring", create_fixed_hiring_agent)
    ]
    
    for agent_name, agent_creator in agents_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING AGENT: {agent_name}")
        print(f"{'='*60}")
        
        # Create environment
        env = WarehouseEnv(
            grid_width=12,
            grid_height=12,
            max_employees=15,
            order_arrival_rate=0.8,  # Higher rate to generate more orders
            render_mode=None
        )
        
        # Create agent
        agent = agent_creator(env)
        
        # Reset environment
        obs = env.reset()
        
        # Create debugger and run
        debugger = ManagerSwapDebugger(env, agent)
        debugger.run_debug_simulation(steps=500)

if __name__ == "__main__":
    main()