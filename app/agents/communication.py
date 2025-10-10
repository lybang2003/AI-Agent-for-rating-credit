"""
Agent Communication System
Hệ thống giao tiếp giữa các agents trong Multi-Agent System
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from datetime import datetime


class MessageType(Enum):
    """Loại message giữa các agents"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class AgentMessage:
    """Message structure giữa các agents"""
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    message_id: str
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low


class AgentCommunicationHub:
    """Hub trung tâm để các agents giao tiếp"""
    
    def __init__(self):
        self.message_queue: List[AgentMessage] = []
        self.agent_registry: Dict[str, Any] = {}
        self.message_history: List[AgentMessage] = []
    
    def register_agent(self, agent_id: str, agent_instance: Any):
        """Đăng ký agent vào hệ thống"""
        self.agent_registry[agent_id] = agent_instance
        print(f"Agent {agent_id} đã đăng ký")
    
    def send_message(self, message: AgentMessage):
        """Gửi message giữa các agents"""
        self.message_queue.append(message)
        self.message_history.append(message)
        
        # Xử lý message ngay lập tức nếu receiver đã đăng ký
        if message.receiver in self.agent_registry:
            self._process_message(message)
    
    def _process_message(self, message: AgentMessage):
        """Xử lý message đến agent"""
        receiver = self.agent_registry.get(message.receiver)
        if receiver and hasattr(receiver, 'handle_message'):
            try:
                receiver.handle_message(message)
            except Exception as e:
                print(f"Error processing message to {message.receiver}: {e}")
    
    def get_messages_for_agent(self, agent_id: str) -> List[AgentMessage]:
        """Lấy tất cả messages cho một agent"""
        return [msg for msg in self.message_queue if msg.receiver == agent_id]
    
    def clear_processed_messages(self, agent_id: str):
        """Xóa messages đã xử lý"""
        self.message_queue = [msg for msg in self.message_queue if msg.receiver != agent_id]


class AgentBase:
    """Base class cho tất cả agents với communication capabilities"""
    
    def __init__(self, agent_id: str, communication_hub: AgentCommunicationHub):
        self.agent_id = agent_id
        self.communication_hub = communication_hub
        self.message_history: List[AgentMessage] = []
        
        # Đăng ký agent
        self.communication_hub.register_agent(agent_id, self)
    
    def send_message(self, receiver: str, message_type: MessageType, 
                    content: Dict[str, Any], correlation_id: Optional[str] = None):
        """Gửi message đến agent khác"""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            message_id=f"{self.agent_id}_{datetime.now().timestamp()}",
            correlation_id=correlation_id
        )
        self.communication_hub.send_message(message)
    
    def handle_message(self, message: AgentMessage):
        """Xử lý message nhận được - override trong subclass"""
        self.message_history.append(message)
        print(f"Agent {self.agent_id} nhận message từ {message.sender}: {message.content}")
    
    def request_data(self, target_agent: str, data_type: str, **kwargs):
        """Request dữ liệu từ agent khác"""
        self.send_message(
            receiver=target_agent,
            message_type=MessageType.REQUEST,
            content={
                "action": "get_data",
                "data_type": data_type,
                "parameters": kwargs
            }
        )
    
    def notify_completion(self, target_agent: str, task: str, result: Any):
        """Thông báo hoàn thành task"""
        self.send_message(
            receiver=target_agent,
            message_type=MessageType.NOTIFICATION,
            content={
                "action": "task_completed",
                "task": task,
                "result": result
            }
        )


# Global communication hub
communication_hub = AgentCommunicationHub()
