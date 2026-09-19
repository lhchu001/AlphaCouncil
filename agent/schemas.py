# agent/schemas.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Signal = Literal["buy", "sell", "hold"]
AnalysisStatus = Literal["ok", "unavailable", "degraded"]


class UserRequest(BaseModel):
    ticker: str
    start: str
    end: str
    interval: str
    period: str
    query: str


class IndicatorResult(BaseModel):
    ticker: str
    start: str
    end: str
    interval: str
    indicators: List[Dict[str, Any]] = Field(default_factory=list)
    data_points: int = 0
    error: Optional[str] = None


class FundamentalResult(BaseModel):
    ticker: str
    period: str
    fundamentals: Dict[str, Any] = Field(default_factory=dict)
    fundamentals_analysis: Dict[str, Any] = Field(default_factory=dict)
    balance_sheet: Dict[str, Any] = Field(default_factory=dict)
    cash_flow: Dict[str, Any] = Field(default_factory=dict)
    income_statement: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class RecommendationResult(BaseModel):
    ticker: str
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    combined_score: float = 0.0
    signal: Signal = "hold"
    technical_analysis: str = ""
    fundamental_analysis: str = ""
    final_analysis: str = ""


class Provenance(BaseModel):
    source: str
    retrieved_at: datetime
    market_date: Optional[str] = None
    period: Optional[str] = None
    ticker: str
    cache_hit: bool = False
    data_hash: Optional[str] = None


class DataQuality(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)
    stale: bool = False
    freshness_seconds: Optional[float] = None
    max_age_seconds: Optional[float] = None
    status: AnalysisStatus = "ok"


class Evidence(BaseModel):
    facts: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance
    quality: DataQuality


class SpecialistInterpretation(BaseModel):
    status: AnalysisStatus = "ok"
    signal: Signal = "hold"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    report: str = ""
    model_provider: Optional[str] = None
    model_name: Optional[str] = None


class Decision(BaseModel):
    signal: Signal = "hold"
    score: float = 0.0
    reason: str = ""
    council_used: bool = False
    human_review_required: bool = False
    risk_flags: List[str] = Field(default_factory=list)
    risk_dashboard: Optional[Dict[str, Any]] = None


class AnalysisState(UserRequest):
    messages: Optional[List[Any]] = None

    technical_evidence: Optional[Evidence] = None
    fundamental_evidence: Optional[Evidence] = None

    technical_interpretation: Optional[SpecialistInterpretation] = None
    fundamental_interpretation: Optional[SpecialistInterpretation] = None

    decision: Optional[Decision] = None
    final_analysis: str = ""

    # Compatibility fields for older specialist agents
    indicators_requested: Optional[List[str]] = None
    indicator_results: Optional[Dict[str, Any]] = None
    selected_indicators: Optional[List[str]] = None
    indicator_selection_reason: Optional[str] = None
    technical_signal: Optional[Signal] = None
    technical_score: Optional[float] = None
    technical_analysis: Optional[str] = None

    statement: Optional[str] = None
    fundamentals_result: Optional[Dict[str, Any]] = None
    selected_fundamental_skills: Optional[List[str]] = None
    fundamental_selection_reason: Optional[str] = None
    fundamental_signal: Optional[Signal] = None
    fundamental_score: Optional[float] = None
    fundamental_analysis: Optional[str] = None