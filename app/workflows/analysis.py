"""Supervisor-routed LangGraph workflow for financial-document analysis."""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.compliance import ComplianceAgent
from app.agents.quantitative import QuantitativeAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.supervisor import SupervisorAgent
from app.domain.analysis import (
    AgentStep,
    AnalysisRoute,
    ComplianceFinding,
    QuantitativeResult,
    QueryResponse,
    RetrievedPassage,
    TokenUsage,
)


class AgentState(TypedDict, total=False):
    """State shared by every node in the FinDocs analysis graph.

    The state retains source context, routing decisions, calculation output, and
    final structured output. List reducers preserve the full agent trace rather
    than allowing a later node to overwrite earlier evidence or errors.
    """

    document_id: str
    query: str
    route: AnalysisRoute
    retrieved_passages: list[RetrievedPassage]
    compliance_findings: list[ComplianceFinding]
    quantitative_result: QuantitativeResult | None
    steps: Annotated[list[AgentStep], operator.add]
    token_usage: TokenUsage
    errors: Annotated[list[str], operator.add]
    final_response: QueryResponse


class FinancialAnalysisWorkflow:
    """Compile and invoke the supervisor, retrieval, compliance, and quant agents."""

    def __init__(
        self,
        supervisor: SupervisorAgent,
        retrieval: RetrievalAgent,
        compliance: ComplianceAgent,
        quantitative: QuantitativeAgent,
    ) -> None:
        self._supervisor = supervisor
        self._retrieval = retrieval
        self._compliance = compliance
        self._quantitative = quantitative
        self._graph = self._compile()

    async def invoke(self, document_id: str, query: str) -> QueryResponse:
        """Run the compiled graph and release the request-scoped retrieval client."""

        try:
            state = await self._graph.ainvoke(
                {
                    "document_id": document_id,
                    "query": query,
                    "retrieved_passages": [],
                    "compliance_findings": [],
                    "quantitative_result": None,
                    "steps": [],
                    "token_usage": TokenUsage(),
                    "errors": [],
                }
            )
        finally:
            await self._retrieval.close()
        return state["final_response"]

    def _compile(self) -> object:
        """Build the acyclic supervisor-routed graph and compile it for execution."""

        graph = StateGraph(AgentState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("compliance", self._compliance_node)
        graph.add_node("quantitative", self._quantitative_node)
        graph.add_node("synthesizer", self._synthesizer_node)
        graph.add_edge(START, "supervisor")
        graph.add_edge("supervisor", "retrieval")
        graph.add_conditional_edges(
            "retrieval",
            self._after_retrieval,
            {"compliance": "compliance", "quantitative": "quantitative", "synthesizer": "synthesizer"},
        )
        graph.add_conditional_edges(
            "compliance",
            self._after_compliance,
            {"quantitative": "quantitative", "synthesizer": "synthesizer"},
        )
        graph.add_edge("quantitative", "synthesizer")
        graph.add_edge("synthesizer", END)
        return graph.compile()

    async def _supervisor_node(self, state: AgentState) -> dict[str, object]:
        """Select the next analysis path, using Gemini only for routing intent."""

        route, detail, usage = await self._supervisor.route(state["query"])
        return {"route": route, "token_usage": usage, "steps": [AgentStep(name="supervisor", status="completed", detail=detail)]}

    async def _retrieval_node(self, state: AgentState) -> dict[str, object]:
        """Fetch document-filtered context and preserve every source citation."""

        try:
            passages = await self._retrieval.retrieve(state["document_id"], state["query"])
            return {
                "retrieved_passages": passages,
                "steps": [
                    AgentStep(name="retrieval", status="completed", detail=f"Retrieved {len(passages)} source-cited passage(s) from Qdrant.")
                ],
            }
        except Exception as exc:
            return {
                "retrieved_passages": [],
                "errors": [f"Retrieval failed: {exc}"],
                "steps": [AgentStep(name="retrieval", status="failed", detail="No source context could be retrieved.")],
            }

    def _compliance_node(self, state: AgentState) -> dict[str, object]:
        """Evaluate JSON rules against retrieved evidence only."""

        try:
            findings = self._compliance.analyze(state.get("retrieved_passages", []))
            return {
                "compliance_findings": findings,
                "steps": [
                    AgentStep(name="compliance", status="completed", detail=f"Evaluated rules and produced {len(findings)} triggered finding(s).")
                ],
            }
        except Exception as exc:
            return {
                "compliance_findings": [],
                "errors": [f"Compliance analysis failed: {exc}"],
                "steps": [AgentStep(name="compliance", status="failed", detail="Compliance rules could not be evaluated.")],
            }

    def _quantitative_node(self, state: AgentState) -> dict[str, object]:
        """Extract exact operands and execute a restricted Decimal calculation."""

        try:
            result = self._quantitative.analyze(state["query"], state.get("retrieved_passages", []))
            return {
                "quantitative_result": result,
                "steps": [
                    AgentStep(
                        name="quantitative",
                        status=result.status,
                        detail=("Executed validated Decimal calculation." if result.status == "completed" else (result.error or "No calculation was performed.")),
                    )
                ],
            }
        except Exception as exc:
            return {
                "quantitative_result": QuantitativeResult(status="failed", error="Restricted calculation failed."),
                "errors": [f"Quantitative analysis failed: {exc}"],
                "steps": [AgentStep(name="quantitative", status="failed", detail="Restricted calculation could not run.")],
            }

    @staticmethod
    def _after_retrieval(state: AgentState) -> Literal["compliance", "quantitative", "synthesizer"]:
        """Route to the specialized post-retrieval agents selected by the supervisor."""

        if state["route"] in {AnalysisRoute.COMPLIANCE, AnalysisRoute.BOTH}:
            return "compliance"
        if state["route"] is AnalysisRoute.QUANTITATIVE:
            return "quantitative"
        return "synthesizer"

    @staticmethod
    def _after_compliance(state: AgentState) -> Literal["quantitative", "synthesizer"]:
        """Invoke deterministic math after compliance only when the supervisor requested both."""

        return "quantitative" if state["route"] is AnalysisRoute.BOTH else "synthesizer"

    @staticmethod
    def _synthesizer_node(state: AgentState) -> dict[str, object]:
        """Aggregate verified outputs into JSON without generating unsupported financial facts."""

        passages = state.get("retrieved_passages", [])
        findings = state.get("compliance_findings", [])
        result = state.get("quantitative_result")
        errors = state.get("errors", [])
        citations = FinancialAnalysisWorkflow._unique_citations(passages, findings, result)
        summary_parts: list[str] = []
        if passages:
            summary_parts.append(f"Retrieved {len(passages)} source-cited passage(s).")
        else:
            summary_parts.append("No source-cited passages were available for analysis.")
        if findings:
            flags = sum(finding.status == "flag" for finding in findings)
            summary_parts.append(f"Compliance rules produced {flags} flag(s) across {len(findings)} triggered rule(s).")
        if result:
            if result.status == "completed":
                summary_parts.append(f"{result.metric} was calculated deterministically as {result.value}.")
            else:
                summary_parts.append(result.error or "No deterministic calculation was available.")
        if errors:
            summary_parts.append("One or more agent stages failed; no unsupported conclusion was generated.")
        response = QueryResponse(
            document_id=state["document_id"],
            query=state["query"],
            route=state["route"],
            summary=" ".join(summary_parts),
            retrieved_passages=passages,
            compliance_findings=findings,
            quantitative_result=result,
            citations=citations,
            steps=state.get("steps", []) + [AgentStep(name="synthesizer", status="completed", detail="Assembled structured, evidence-bound response.")],
            token_usage=state.get("token_usage", TokenUsage()),
            errors=errors,
        )
        return {"final_response": response}

    @staticmethod
    def _unique_citations(
        passages: list[RetrievedPassage],
        findings: list[ComplianceFinding],
        result: QuantitativeResult | None,
    ) -> list[object]:
        seen: set[tuple[str, str]] = set()
        citations = []
        for citation in [passage.citation for passage in passages] + [
            citation for finding in findings for citation in finding.citations
        ] + ([item.citation for item in result.inputs] if result else []):
            key = (citation.document_id, citation.chunk_id)
            if key not in seen:
                seen.add(key)
                citations.append(citation)
        return citations
