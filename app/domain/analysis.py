"""Structured contracts for the LangGraph financial-analysis workflow."""

from enum import StrEnum

from pydantic import BaseModel, Field


class AnalysisRoute(StrEnum):
    """Supervisor-selected workflow paths."""

    RETRIEVAL = "retrieval"
    COMPLIANCE = "compliance"
    QUANTITATIVE = "quantitative"
    BOTH = "both"


class Citation(BaseModel):
    """Source location for a statement, compliance flag, or numeric input."""

    document_id: str
    chunk_id: str
    source_filename: str
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None
    excerpt: str = Field(min_length=1, max_length=1_000)


class RetrievedPassage(BaseModel):
    """A vector-search result bound to a specific uploaded document."""

    citation: Citation
    score: float
    text: str = Field(min_length=1)
    kind: str


class ComplianceFinding(BaseModel):
    """Result of applying one deterministic compliance rule to source passages."""

    rule_id: str
    title: str
    severity: str
    status: str
    rationale: str
    citations: list[Citation] = Field(default_factory=list)


class CalculationInput(BaseModel):
    """One source-derived operand used by the code interpreter."""

    metric: str
    value: str
    citation: Citation


class QuantitativeResult(BaseModel):
    """Deterministic financial calculation output, or an explicit inability to calculate."""

    status: str
    metric: str | None = None
    formula: str | None = None
    value: str | None = None
    inputs: list[CalculationInput] = Field(default_factory=list)
    executed_code: str | None = None
    error: str | None = None


class AgentStep(BaseModel):
    """An auditable description of one LangGraph node's completed action."""

    name: str
    status: str
    detail: str


class TokenUsage(BaseModel):
    """Provider-reported model token counts, kept separate from deterministic tools."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class QueryResponse(BaseModel):
    """Clean, structured response assembled only from agent outputs and citations."""

    document_id: str
    query: str
    route: AnalysisRoute
    summary: str
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    compliance_findings: list[ComplianceFinding] = Field(default_factory=list)
    quantitative_result: QuantitativeResult | None = None
    citations: list[Citation] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    errors: list[str] = Field(default_factory=list)
