#!/usr/bin/env python3
"""
Skeleton RL Agent - Intentionally naive implementation for students to improve

This agent demonstrates the basic structure but makes poor decisions:
- Random staffing decisions
- No layout optimization 
- Ignores order priorities
- No learning from experience

Students should improve each component to beat the baseline agents.
"""

import numpy as np
from typing import Dict, Optional
from .standardized_agents import BaselineAgent

class SkeletonRLAgent(BaselineAgent):
    """
    Skeleton RL agent that makes obviously bad decisions.
    Students should improve this to create an effective RL agent.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "SkeletonRL"
        
        # TODO: Students should implement proper state tracking
        self.action_history = []
        self.reward_history = []
        
        # TODO: Students should implement proper policy networks
        # This naive approach just uses random weights
        self.staffing_weights = np.random.randn(4)  # Random decision weights
        self.layout_weights = np.random.randn(3)    # Random layout weights
        
        # TODO: Students should implement learning mechanisms
        self.learning_rate = 0.1  # Not actually used in this skeleton
        self.exploration_rate = 0.5  # Very high - always exploring
        
    def reset(self):
        """Reset agent state - students should expand this"""
        self.action_history = []
        self.reward_history = []
        # TODO: Reset any neural network states, replay buffers, etc.
    
    def get_action(self, observation: Dict) -> Dict:
        """
        Generate action based on observation.
        Current implementation is intentionally terrible - students should improve!
        """
        
        # Extract basic info from observation
        current_timestep = observation['time'][0]
        financial_state = observation['financial']  # [profit, revenue, costs, burn_rate]
        queue_info = observation['order_queue']
        employee_info = observation['employees']
        
        action = {
            'staffing_action': self._get_naive_staffing_action(financial_state, employee_info),
            'layout_swap': self._get_naive_layout_action(current_timestep),
            'order_assignments': self._get_naive_order_assignments(queue_info, employee_info)
        }
        
        # TODO: Students should implement proper action recording for learning
        self.action_history.append(action.copy())
        
        return action
    
    def _get_naive_staffing_action(self, financial_state, employee_info) -> int:
        """
        Naive staffing decisions - students should improve this!
        
        Current problems:
        - Ignores queue length and workload
        - Random decisions regardless of profit
        - No consideration of employee efficiency
        """
        
        # TODO: Students should implement intelligent staffing logic
        # Current approach: Make random decisions based on "vibes"
        
        current_profit = financial_state[0]
        num_employees = np.sum(employee_info[:, 0] > 0)  # Count active employees
        
        # Terrible logic: Random decisions with slight bias
        if np.random.random() < 0.3:  # 30% chance to hire
            if num_employees < 20:  # Don't go completely overboard
                return 1  # Hire worker
        elif np.random.random() < 0.1:  # 10% chance to fire
            if num_employees > 1:  # Don't fire everyone
                return 2  # Fire worker
        elif np.random.random() < 0.05:  # 5% chance to hire manager
            return 3  # Hire manager
        
        return 0  # No action
    
    def _get_naive_layout_action(self, current_timestep) -> list:
        """
        Naive layout optimization - students should improve this!
        
        Current problems:
        - Random swaps with no strategic purpose
        - Ignores item co-occurrence patterns
        - No consideration of delivery distances
        - Wastes manager time on pointless moves
        """
        
        # TODO: Students should implement intelligent layout optimization
        # Current approach: Occasionally make random swaps
        
        if current_timestep % 100 == 0 and np.random.random() < 0.2:  # Random timing
            # Pick two random positions to swap
            grid_size = self.env.grid_width * self.env.grid_height
            pos1 = np.random.randint(0, grid_size)
            pos2 = np.random.randint(0, grid_size)
            return [pos1, pos2]
        
        return [0, 0]  # No swap
    
    def _get_naive_order_assignments(self, queue_info, employee_info) -> list:
        """
        Naive order assignment - students should improve this!
        
        Current problems:
        - Ignores employee locations
        - No consideration of order priority/value
        - Random assignments regardless of efficiency
        - Doesn't check if employees are actually available
        """
        
        # TODO: Students should implement intelligent order assignment
        # Current approach: Random assignments
        
        assignments = [0] * 20  # No assignments by default
        
        # Count how many employees we have (very naive)
        num_employees = int(np.sum(employee_info[:, 0] > 0))
        
        # Randomly assign first few orders to first few employees
        if num_employees > 0:
            for i in range(min(3, num_employees)):  # Only assign 3 orders max
                if np.random.random() < 0.6:  # 60% chance to assign
                    assignments[i] = (i % num_employees) + 1  # Random employee
        
        return assignments
    
    def record_reward(self, reward: float):
        """
        Record reward for learning - students should expand this
        
        TODO: Students should implement:
        - Proper reward tracking
        - Experience replay buffers
        - Policy gradient calculations
        - Q-value updates
        """
        self.reward_history.append(reward)
        
        # Skeleton learning - doesn't actually learn anything useful
        if len(self.reward_history) > 10:
            # "Update" weights randomly (this doesn't actually improve performance)
            if reward > 0:
                self.staffing_weights += np.random.randn(4) * 0.01
                self.layout_weights += np.random.randn(3) * 0.01
    
    def should_update_policy(self) -> bool:
        """
        Determine when to update policy - students should improve this
        
        TODO: Students should implement proper update schedules
        """
        return len(self.action_history) % 50 == 0  # Arbitrary update frequency
    
    def get_performance_metrics(self) -> Dict:
        """
        Get agent performance metrics for analysis
        Students can use this to debug their improvements
        """
        if not self.reward_history:
            return {"avg_reward": 0, "total_actions": 0}
        
        return {
            "avg_reward": np.mean(self.reward_history[-100:]),  # Last 100 rewards
            "total_actions": len(self.action_history),
            "exploration_rate": self.exploration_rate,
            "recent_performance": np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0
        }


def create_skeleton_rl_agent(env) -> SkeletonRLAgent:
    """Factory function to create skeleton RL agent"""
    return SkeletonRLAgent(env)


# TODO: Students should implement these advanced components:

class StudentRLAgent(SkeletonRLAgent):
    """
    Template for students to implement their improved RL agent
    
    Suggested improvements:
    1. Replace random staffing with demand-based hiring
    2. Implement proper layout optimization using item frequencies
    3. Add distance-based order assignment
    4. Implement basic Q-learning or policy gradients
    5. Add proper exploration vs exploitation balance
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentRL"
        
        # TODO: Students implement these
        # self.q_table = {}
        # self.policy_network = SimpleNeuralNetwork()
        # self.experience_buffer = []
        # self.target_network = None
        
    def _get_improved_staffing_action(self, financial_state, employee_info, queue_info):
        """
        TODO: Students implement intelligent staffing:
        - Hire when queue is growing
        - Fire when queue is empty for extended periods
        - Consider profit margins before hiring
        - Balance managers vs workers
        """
        pass
    
    def _get_improved_layout_action(self, observation):
        """
        TODO: Students implement smart layout optimization:
        - Move frequently accessed items closer to delivery
        - Group items that are often ordered together
        - Only optimize when queue is manageable
        - Track swap effectiveness
        """
        pass
    
    def _get_improved_order_assignments(self, queue_info, employee_info):
        """
        TODO: Students implement efficient order assignment:
        - Assign orders to closest available employees
        - Prioritize high-value or urgent orders
        - Consider employee current locations
        - Balance workload across employees
        """
        pass
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """
        TODO: Students implement learning algorithm:
        - Q-learning updates
        - Policy gradient calculations
        - Experience replay
        - Target network updates
        """
        pass