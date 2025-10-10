"""
Shared State Management System
Hệ thống quản lý state chung giữa các agents
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading
import json
import copy


@dataclass
class AgentState:
    """State của một agent"""
    agent_id: str
    status: str = "idle"  # idle, busy, error, completed
    current_task: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    success_count: int = 0


@dataclass
class SharedContext:
    """Context chung giữa các agents"""
    session_id: str
    company: str
    features: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class StateManager:
    """Quản lý state chung của Multi-Agent System"""
    
    def __init__(self):
        self.agent_states: Dict[str, AgentState] = {}
        self.shared_contexts: Dict[str, SharedContext] = {}
        self.global_state: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def register_agent(self, agent_id: str):
        """Đăng ký agent vào state manager"""
        with self.lock:
            self.agent_states[agent_id] = AgentState(agent_id=agent_id)
            print(f"Agent {agent_id} đã đăng ký vào state manager")
    
    def update_agent_status(self, agent_id: str, status: str, task: Optional[str] = None):
        """Cập nhật status của agent"""
        with self.lock:
            if agent_id in self.agent_states:
                self.agent_states[agent_id].status = status
                self.agent_states[agent_id].current_task = task
                self.agent_states[agent_id].last_updated = datetime.now()
    
    def set_agent_data(self, agent_id: str, data: Dict[str, Any]):
        """Set data cho agent"""
        with self.lock:
            if agent_id in self.agent_states:
                self.agent_states[agent_id].data.update(data)
                self.agent_states[agent_id].last_updated = datetime.now()
    
    def get_agent_data(self, agent_id: str, key: Optional[str] = None) -> Any:
        """Lấy data của agent"""
        with self.lock:
            if agent_id not in self.agent_states:
                return None
            
            agent_data = self.agent_states[agent_id].data
            if key:
                return agent_data.get(key)
            return agent_data
    
    def create_shared_context(self, session_id: str, company: str, 
                            initial_features: Optional[Dict[str, Any]] = None) -> SharedContext:
        """Tạo shared context mới"""
        with self.lock:
            context = SharedContext(
                session_id=session_id,
                company=company,
                features=initial_features or {}
            )
            self.shared_contexts[session_id] = context
            return context
    
    def update_shared_context(self, session_id: str, updates: Dict[str, Any]):
        """Cập nhật shared context"""
        with self.lock:
            if session_id in self.shared_contexts:
                context = self.shared_contexts[session_id]
                
                # Update features
                if 'features' in updates:
                    context.features.update(updates['features'])
                
                # Update results
                if 'results' in updates:
                    context.results.update(updates['results'])
                
                # Update metadata
                if 'metadata' in updates:
                    context.metadata.update(updates['metadata'])
                
                context.updated_at = datetime.now()
    
    def get_shared_context(self, session_id: str) -> Optional[SharedContext]:
        """Lấy shared context"""
        with self.lock:
            return self.shared_contexts.get(session_id)
    
    def set_global_state(self, key: str, value: Any):
        """Set global state"""
        with self.lock:
            self.global_state[key] = value
    
    def get_global_state(self, key: str) -> Any:
        """Lấy global state"""
        with self.lock:
            return self.global_state.get(key)
    
    def get_all_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Lấy status của tất cả agents"""
        with self.lock:
            return {
                agent_id: {
                    "status": state.status,
                    "current_task": state.current_task,
                    "last_updated": state.last_updated.isoformat(),
                    "error_count": state.error_count,
                    "success_count": state.success_count
                }
                for agent_id, state in self.agent_states.items()
            }
    
    def increment_error_count(self, agent_id: str):
        """Tăng error count của agent"""
        with self.lock:
            if agent_id in self.agent_states:
                self.agent_states[agent_id].error_count += 1
    
    def increment_success_count(self, agent_id: str):
        """Tăng success count của agent"""
        with self.lock:
            if agent_id in self.agent_states:
                self.agent_states[agent_id].success_count += 1
    
    def get_system_health(self) -> Dict[str, Any]:
        """Lấy health status của toàn bộ system"""
        with self.lock:
            total_agents = len(self.agent_states)
            busy_agents = sum(1 for state in self.agent_states.values() if state.status == "busy")
            error_agents = sum(1 for state in self.agent_states.values() if state.status == "error")
            
            return {
                "total_agents": total_agents,
                "busy_agents": busy_agents,
                "error_agents": error_agents,
                "idle_agents": total_agents - busy_agents - error_agents,
                "active_sessions": len(self.shared_contexts),
                "global_state_keys": list(self.global_state.keys())
            }


# Global state manager
state_manager = StateManager()
