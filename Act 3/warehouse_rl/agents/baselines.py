from typing import Dict
from .standardized_agents import get_standardized_agents, BaselineAgent
from .skeleton_rl_agent import create_skeleton_rl_agent

def get_baseline_agents(env) -> Dict[str, BaselineAgent]:
    """Get all available baseline agents"""
    agents = get_standardized_agents(env)
    
    # Add skeleton RL agent for students to improve
    agents['skeleton_rl'] = create_skeleton_rl_agent(env)
    
    return agents