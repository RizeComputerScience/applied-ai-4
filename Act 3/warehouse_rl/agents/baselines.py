from typing import Dict
from .standardized_agents import get_standardized_agents, BaselineAgent

def get_baseline_agents(env) -> Dict[str, BaselineAgent]:
    """Get all available baseline agents"""
    return get_standardized_agents(env)