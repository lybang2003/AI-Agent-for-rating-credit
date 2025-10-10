from __future__ import annotations

from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

try:
    from langgraph.graph import StateGraph
except Exception:
    StateGraph = None  # optional

from app.agents.core import InputNormalizer, DataCatalog, InternalRetrieval, WebGather, FeatureMerger
from app.agents.ml_report import PredictorAgent, ExplainerAgent, ReporterAgent, InvestmentPredictorAgent
from app.agents.communication import AgentBase, MessageType, communication_hub
from app.agents.state_manager import state_manager, SharedContext
from app.agents.agent_selector import agent_selector, TaskRequirement, AgentCapability


class EnhancedOrchestrator:
    """Enhanced Orchestrator với Multi-Agent capabilities"""
    
    def __init__(self) -> None:
        # Initialize agents with communication capabilities
        self.agents = {}
        self.session_id = None
        self._initialize_agents()
        self._register_agents()
    
    def _initialize_agents(self):
        """Khởi tạo tất cả agents"""
        self.agents = {
            "normalizer": InputNormalizer(),
            "catalog": DataCatalog(),
            "internal": InternalRetrieval(),
            "web": WebGather(),
            "merger": FeatureMerger(),
            "predictor": PredictorAgent(),
            "explainer": ExplainerAgent(),
            "reporter": ReporterAgent(),
            "investment_predictor": InvestmentPredictorAgent()
        }
    
    def _register_agents(self):
        """Đăng ký agents vào communication hub và state manager"""
        # Register with communication hub
        for agent_id, agent in self.agents.items():
            communication_hub.register_agent(agent_id, agent)
            state_manager.register_agent(agent_id)
        
        # Register with agent selector
        agent_selector.register_agent(
            "normalizer", 
            [AgentCapability.DATA_PROCESSING],
            ["input_normalization"]
        )
        agent_selector.register_agent(
            "catalog", 
            [AgentCapability.DATA_PROCESSING],
            ["data_planning"]
        )
        agent_selector.register_agent(
            "internal", 
            [AgentCapability.DATA_RETRIEVAL],
            ["database_queries", "csv_processing"]
        )
        agent_selector.register_agent(
            "web", 
            [AgentCapability.WEB_SEARCH, AgentCapability.DATA_RETRIEVAL],
            ["web_scraping", "api_integration"]
        )
        agent_selector.register_agent(
            "merger", 
            [AgentCapability.DATA_PROCESSING],
            ["feature_engineering"]
        )
        agent_selector.register_agent(
            "predictor", 
            [AgentCapability.ML_PREDICTION],
            ["credit_rating", "financial_analysis"]
        )
        agent_selector.register_agent(
            "explainer", 
            [AgentCapability.EXPLANATION],
            ["model_interpretation", "feature_importance"]
        )
        agent_selector.register_agent(
            "reporter", 
            [AgentCapability.REPORTING, AgentCapability.DATA_VISUALIZATION],
            ["report_generation", "chart_creation"]
        )
        agent_selector.register_agent(
            "investment_predictor", 
            [AgentCapability.INVESTMENT_ANALYSIS, AgentCapability.ML_PREDICTION],
            ["investment_recommendation", "risk_assessment"]
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced run với Multi-Agent coordination"""
        # Create session
        self.session_id = str(uuid.uuid4())
        
        # Create shared context
        context = state_manager.create_shared_context(
            session_id=self.session_id,
            company=payload.get("company", ""),
            initial_features=payload.get("features", {})
        )
        
        try:
            # Phase 1: Data Preparation
            norm_result = self._run_data_preparation_phase(payload, context)
            
            # Phase 2: Analysis Phase
            analysis_result = self._run_analysis_phase(norm_result, context)
            
            # Phase 3: Reporting Phase
            report_result = self._run_reporting_phase(analysis_result, context)
            
            return {
                "session_id": self.session_id,
                "normalized": norm_result,
                "analysis": analysis_result,
                "reporting": report_result,
                "context": context.__dict__,
                "agent_status": state_manager.get_all_agent_status(),
                "system_health": state_manager.get_system_health()
            }
            
        except Exception as e:
            # Error handling
            state_manager.set_global_state("last_error", str(e))
            return {
                "error": str(e),
                "session_id": self.session_id,
                "context": context.__dict__ if context else {}
            }
    
    def _run_data_preparation_phase(self, payload: Dict[str, Any], 
                                  context: SharedContext) -> Dict[str, Any]:
        """Phase 1: Data Preparation với agent coordination"""
        print("🔄 Phase 1: Data Preparation")
        
        # Update agent status
        state_manager.update_agent_status("normalizer", "busy", "normalize_input")
        
        # Normalize input
        norm = self.agents["normalizer"].normalize(payload)
        state_manager.set_agent_data("normalizer", {"normalized_input": norm})
        state_manager.update_agent_status("normalizer", "completed")
        
        # Plan data collection
        state_manager.update_agent_status("catalog", "busy", "plan_data_collection")
        plan = self.agents["catalog"].plan(norm["company"], list((norm.get("features") or {}).keys()))
        state_manager.set_agent_data("catalog", {"plan": plan})
        state_manager.update_agent_status("catalog", "completed")
        
        # Retrieve internal data
        state_manager.update_agent_status("internal", "busy", "retrieve_internal_data")
        internal_feats = self.agents["internal"].get_features(norm["company"]) if norm.get("company") else {}
        state_manager.set_agent_data("internal", {"features": internal_feats})
        state_manager.update_agent_status("internal", "completed")
        
        # Web search if needed
        external_feats = {}
        if not internal_feats:
            state_manager.update_agent_status("web", "busy", "web_search")
            external_feats = self.agents["web"].search_and_extract(norm.get("company", ""))
            state_manager.set_agent_data("web", {"features": external_feats})
            state_manager.update_agent_status("web", "completed")
        
        # Merge features
        state_manager.update_agent_status("merger", "busy", "merge_features")
        merged = self.agents["merger"].merge(norm.get("features", {}), internal_feats, external_feats)
        state_manager.set_agent_data("merger", {"merged_features": merged})
        state_manager.update_agent_status("merger", "completed")
        
        # Update shared context
        state_manager.update_shared_context(self.session_id, {
            "features": merged,
            "metadata": {"phase": "data_preparation", "completed_at": datetime.now().isoformat()}
        })
        
        return {"normalized": norm, "plan": plan, "features": merged}
    
    def _run_analysis_phase(self, norm_result: Dict[str, Any], 
                          context: SharedContext) -> Dict[str, Any]:
        """Phase 2: Analysis với parallel agent execution"""
        print("🔄 Phase 2: Analysis")
        
        features = norm_result["features"]
        
        # Parallel prediction tasks
        prediction_tasks = []
        
        # Credit rating prediction
        state_manager.update_agent_status("predictor", "busy", "credit_rating_prediction")
        try:
            pred = self.agents["predictor"].predict(features)
            state_manager.set_agent_data("predictor", {"prediction": pred})
            state_manager.update_agent_status("predictor", "completed")
            prediction_tasks.append(("credit_rating", pred))
        except Exception as e:
            state_manager.update_agent_status("predictor", "error")
            state_manager.increment_error_count("predictor")
            print(f"Credit rating prediction error: {e}")
        
        # Investment prediction
        state_manager.update_agent_status("investment_predictor", "busy", "investment_prediction")
        try:
            investment_result = self.agents["investment_predictor"].get_investment_recommendation(features)
            state_manager.set_agent_data("investment_predictor", {"investment_result": investment_result})
            state_manager.update_agent_status("investment_predictor", "completed")
            prediction_tasks.append(("investment", investment_result))
        except Exception as e:
            state_manager.update_agent_status("investment_predictor", "error")
            state_manager.increment_error_count("investment_predictor")
            print(f"Investment prediction error: {e}")
        
        # Explanation
        state_manager.update_agent_status("explainer", "busy", "generate_explanation")
        try:
            exp = self.agents["explainer"].explain(features)
            state_manager.set_agent_data("explainer", {"explanation": exp})
            state_manager.update_agent_status("explainer", "completed")
        except Exception as e:
            state_manager.update_agent_status("explainer", "error")
            state_manager.increment_error_count("explainer")
            print(f"Explanation error: {e}")
        
        # Update shared context
        state_manager.update_shared_context(self.session_id, {
            "results": {"predictions": prediction_tasks},
            "metadata": {"phase": "analysis", "completed_at": datetime.now().isoformat()}
        })
        
        return {"predictions": prediction_tasks}
    
    def _run_reporting_phase(self, analysis_result: Dict[str, Any], 
                           context: SharedContext) -> Dict[str, Any]:
        """Phase 3: Reporting"""
        print("🔄 Phase 3: Reporting")
        
        state_manager.update_agent_status("reporter", "busy", "generate_report")
        try:
            # Generate report
            report = self.agents["reporter"].generate_report(analysis_result)
            state_manager.set_agent_data("reporter", {"report": report})
            state_manager.update_agent_status("reporter", "completed")
        except Exception as e:
            state_manager.update_agent_status("reporter", "error")
            state_manager.increment_error_count("reporter")
            print(f"Reporting error: {e}")
        
        # Update shared context
        state_manager.update_shared_context(self.session_id, {
            "metadata": {"phase": "reporting", "completed_at": datetime.now().isoformat()}
        })
        
        return {"report": report if 'report' in locals() else {}}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Lấy status của toàn bộ system"""
        return {
            "agent_status": state_manager.get_all_agent_status(),
            "system_health": state_manager.get_system_health(),
            "active_sessions": len(state_manager.shared_contexts),
            "communication_hub": {
                "registered_agents": list(communication_hub.agent_registry.keys()),
                "message_queue_size": len(communication_hub.message_queue)
            }
        }


# Enhanced orchestrator instance
orchestrator = EnhancedOrchestrator()

