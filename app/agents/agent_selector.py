"""
Dynamic Agent Selection System
Hệ thống chọn agent động dựa trên context và requirements
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json


class AgentCapability(Enum):
    """Các khả năng của agents"""
    DATA_RETRIEVAL = "data_retrieval"
    DATA_PROCESSING = "data_processing"
    ML_PREDICTION = "ml_prediction"
    DATA_VISUALIZATION = "data_visualization"
    WEB_SEARCH = "web_search"
    EXPLANATION = "explanation"
    REPORTING = "reporting"
    INVESTMENT_ANALYSIS = "investment_analysis"


@dataclass
class AgentProfile:
    """Profile của một agent"""
    agent_id: str
    capabilities: List[AgentCapability]
    performance_score: float = 1.0
    error_rate: float = 0.0
    avg_response_time: float = 1.0
    specializations: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.specializations is None:
            self.specializations = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TaskRequirement:
    """Yêu cầu của một task"""
    task_type: str
    required_capabilities: List[AgentCapability]
    priority: int = 1  # 1=high, 2=medium, 3=low
    deadline: Optional[float] = None  # seconds
    data_requirements: List[str] = None
    quality_threshold: float = 0.8
    
    def __post_init__(self):
        if self.data_requirements is None:
            self.data_requirements = []


class AgentSelector:
    """Hệ thống chọn agent động"""
    
    def __init__(self):
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
    
    def register_agent(self, agent_id: str, capabilities: List[AgentCapability],
                      specializations: List[str] = None, dependencies: List[str] = None):
        """Đăng ký agent với capabilities"""
        profile = AgentProfile(
            agent_id=agent_id,
            capabilities=capabilities,
            specializations=specializations or [],
            dependencies=dependencies or []
        )
        self.agent_profiles[agent_id] = profile
        print(f"Agent {agent_id} đã đăng ký với capabilities: {[c.value for c in capabilities]}")
    
    def select_agent(self, requirement: TaskRequirement, 
                    context: Dict[str, Any] = None) -> Optional[str]:
        """Chọn agent phù hợp nhất cho task"""
        available_agents = self._get_available_agents()
        if not available_agents:
            return None
        
        # Filter agents by capabilities
        capable_agents = self._filter_by_capabilities(available_agents, requirement)
        if not capable_agents:
            return None
        
        # Score agents based on multiple factors
        scored_agents = self._score_agents(capable_agents, requirement, context)
        
        # Select best agent
        best_agent = max(scored_agents.items(), key=lambda x: x[1])
        return best_agent[0]
    
    def select_agent_chain(self, requirements: List[TaskRequirement],
                          context: Dict[str, Any] = None) -> List[str]:
        """Chọn chain of agents cho multiple tasks"""
        selected_agents = []
        used_agents = set()
        
        for req in requirements:
            # Exclude already used agents
            available_agents = [aid for aid in self._get_available_agents() 
                              if aid not in used_agents]
            
            if not available_agents:
                break
            
            # Select agent for this requirement
            agent_id = self._select_agent_for_requirement(
                req, available_agents, context
            )
            
            if agent_id:
                selected_agents.append(agent_id)
                used_agents.add(agent_id)
        
        return selected_agents
    
    def _get_available_agents(self) -> List[str]:
        """Lấy danh sách agents có sẵn"""
        return list(self.agent_profiles.keys())
    
    def _filter_by_capabilities(self, agent_ids: List[str], 
                              requirement: TaskRequirement) -> List[str]:
        """Filter agents by required capabilities"""
        capable_agents = []
        
        for agent_id in agent_ids:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                continue
            
            # Check if agent has all required capabilities
            has_all_capabilities = all(
                cap in profile.capabilities 
                for cap in requirement.required_capabilities
            )
            
            if has_all_capabilities:
                capable_agents.append(agent_id)
        
        return capable_agents
    
    def _score_agents(self, agent_ids: List[str], requirement: TaskRequirement,
                     context: Dict[str, Any] = None) -> Dict[str, float]:
        """Score agents based on multiple factors"""
        scores = {}
        
        for agent_id in agent_ids:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                continue
            
            score = 0.0
            
            # Base performance score
            score += profile.performance_score * 0.3
            
            # Error rate penalty
            score -= profile.error_rate * 0.2
            
            # Response time penalty
            score -= min(profile.avg_response_time / 10.0, 0.2)
            
            # Specialization bonus
            if context and 'company_sector' in context:
                if context['company_sector'] in profile.specializations:
                    score += 0.2
            
            # Priority handling
            if requirement.priority == 1:  # High priority
                score += 0.1
            
            # Quality threshold check
            if profile.performance_score >= requirement.quality_threshold:
                score += 0.1
            
            scores[agent_id] = max(score, 0.0)  # Ensure non-negative
        
        return scores
    
    def _select_agent_for_requirement(self, requirement: TaskRequirement,
                                   available_agents: List[str],
                                   context: Dict[str, Any] = None) -> Optional[str]:
        """Select agent for specific requirement"""
        capable_agents = self._filter_by_capabilities(available_agents, requirement)
        if not capable_agents:
            return None
        
        scored_agents = self._score_agents(capable_agents, requirement, context)
        if not scored_agents:
            return None
        
        return max(scored_agents.items(), key=lambda x: x[1])[0]
    
    def update_agent_performance(self, agent_id: str, success: bool, 
                               response_time: float):
        """Cập nhật performance metrics của agent"""
        if agent_id not in self.performance_metrics:
            self.performance_metrics[agent_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "total_response_time": 0.0,
                "error_count": 0
            }
        
        metrics = self.performance_metrics[agent_id]
        metrics["total_tasks"] += 1
        metrics["total_response_time"] += response_time
        
        if success:
            metrics["successful_tasks"] += 1
        else:
            metrics["error_count"] += 1
        
        # Update agent profile
        if agent_id in self.agent_profiles:
            profile = self.agent_profiles[agent_id]
            profile.performance_score = metrics["successful_tasks"] / metrics["total_tasks"]
            profile.error_rate = metrics["error_count"] / metrics["total_tasks"]
            profile.avg_response_time = metrics["total_response_time"] / metrics["total_tasks"]
    
    def get_agent_recommendations(self, task_type: str, 
                                context: Dict[str, Any] = None) -> List[str]:
        """Lấy recommendations cho task type"""
        # Map task types to capabilities
        task_capability_map = {
            "data_retrieval": [AgentCapability.DATA_RETRIEVAL],
            "prediction": [AgentCapability.ML_PREDICTION],
            "visualization": [AgentCapability.DATA_VISUALIZATION],
            "web_search": [AgentCapability.WEB_SEARCH],
            "explanation": [AgentCapability.EXPLANATION],
            "reporting": [AgentCapability.REPORTING],
            "investment_analysis": [AgentCapability.INVESTMENT_ANALYSIS]
        }
        
        required_capabilities = task_capability_map.get(task_type, [])
        if not required_capabilities:
            return []
        
        requirement = TaskRequirement(
            task_type=task_type,
            required_capabilities=required_capabilities
        )
        
        available_agents = self._get_available_agents()
        capable_agents = self._filter_by_capabilities(available_agents, requirement)
        scored_agents = self._score_agents(capable_agents, requirement, context)
        
        # Return sorted by score
        return sorted(scored_agents.items(), key=lambda x: x[1], reverse=True)


# Global agent selector
agent_selector = AgentSelector()
