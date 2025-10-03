from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class CompanyInput(BaseModel):
    company: str = Field(..., description="Tên công ty hoặc định danh")
    as_of_date: Optional[str] = Field(None, description="Ngày hiệu lực dữ liệu (YYYY-MM-DD)")
    features: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    input: CompanyInput


class PredictionResult(BaseModel):
    rating: str
    probDist: Optional[List[float]] = None
    confidence: Optional[float] = None
    modelVersion: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None


class ETLLoadResult(BaseModel):
    loaded: int
    source: str
    storage: str


class ChatMessage(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None


class ChartRequest(BaseModel):
    company: str
    metrics: List[str]

