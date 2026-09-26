"""Offline tests for LangGraph routing and deterministic financial math."""

import asyncio
from decimal import Decimal
from pathlib import Path

from app.agents.compliance import ComplianceAgent
from app.agents.quantitative import CodeExecutionError, DeterministicCodeInterpreter, QuantitativeAgent
from app.agents.supervisor import SupervisorAgent
from app.api.v1.router import get_analysis_workflow, get_audit_repository
from app.domain.analysis import AnalysisRoute, Citation, QueryResponse, RetrievedPassage
from app.main import app
from app.domain.audit import QueryAuditEvent
from app.services.audit import AuditRepository
from app.workflows.analysis import FinancialAnalysisWorkflow
from fastapi.testclient import TestClient


def _passage(text: str) -> RetrievedPassage:
    return RetrievedPassage(
        citation=Citation(
            document_id="doc-1",
            chunk_id="doc-1:000001",
            source_filename="credit-agreement.csv",
            page_number=2,
            section="Financial statements",
            excerpt=text,
        ),
        score=0.92,
        text=text,
        kind="table",
    )


def test_restricted_interpreter_executes_decimal_division() -> None:
    """The calculation runner emits a rounded Decimal result from its fixed template."""

    value, code = DeterministicCodeInterpreter().execute_division(Decimal("1200"), Decimal("800"))

    assert value == "1.5000"
    assert "from decimal import Decimal" in code


def test_restricted_interpreter_refuses_zero_denominator() -> None:
    """A failed division produces no guessed financial result."""

    try:
        DeterministicCodeInterpreter().execute_division(Decimal("1200"), Decimal("0"))
    except CodeExecutionError as exc:
        assert "failed" in str(exc).lower()
    else:
        raise AssertionError("Expected deterministic runner to reject zero denominator")


def test_quantitative_agent_uses_cited_table_operands() -> None:
    """Debt-to-equity is calculated only from exact values in one retrieved table."""

    passage = _passage("| Metric | Amount |\n| --- | --- |\n| Total Debt | 1200 |\n| Total Equity | 800 |")
    result = QuantitativeAgent().analyze("Calculate the debt-to-equity ratio", [passage])

    assert result.status == "completed"
    assert result.value == "1.5000"
    assert [item.value for item in result.inputs] == ["1200", "800"]
    assert all(item.citation.chunk_id == "doc-1:000001" for item in result.inputs)


def test_quantitative_agent_does_not_substitute_ebitda_for_ebit() -> None:
    """Near-match labels cannot silently supply a different financial metric."""

    passage = _passage("| Metric | Amount |\n| --- | --- |\n| EBITDA | 1200 |\n| Interest Expense | 800 |")
    result = QuantitativeAgent().analyze("Calculate interest coverage", [passage])

    assert result.status == "unavailable"
    assert result.value is None


def test_langgraph_quantitative_route_returns_structured_response() -> None:
    """The compiled graph routes, retrieves, calculates, synthesizes, and closes resources."""

    class RetrievalStub:
        closed = False

        async def retrieve(self, document_id: str, query: str) -> list[RetrievedPassage]:
            assert document_id == "doc-1"
            assert "debt" in query.lower()
            return [_passage("| Metric | Amount |\n| --- | --- |\n| Total Debt | 1200 |\n| Total Equity | 800 |")]

        async def close(self) -> None:
            self.closed = True

    retrieval = RetrievalStub()
    workflow = FinancialAnalysisWorkflow(
        supervisor=SupervisorAgent(None, "gemini-2.0-flash"),
        retrieval=retrieval,  # type: ignore[arg-type]
        compliance=ComplianceAgent(Path("app/rules/credit_agreement_rules.json")),
        quantitative=QuantitativeAgent(),
    )

    response = asyncio.run(workflow.invoke("doc-1", "Calculate the debt-to-equity ratio"))

    assert response.route.value == "quantitative"
    assert response.quantitative_result is not None
    assert response.quantitative_result.value == "1.5000"
    assert [step.name for step in response.steps] == ["supervisor", "retrieval", "quantitative", "synthesizer"]
    assert retrieval.closed


def test_compliance_agent_flags_undefined_prepayment_penalty() -> None:
    """Compliance flags use the rule file and carry the triggering citation."""

    findings = ComplianceAgent(Path("app/rules/credit_agreement_rules.json")).analyze(
        [_passage("Borrower may incur an early repayment penalty upon termination.")]
    )

    assert len(findings) == 1
    assert findings[0].rule_id == "prepayment-penalty-disclosure"
    assert findings[0].status == "flag"
    assert findings[0].citations[0].chunk_id == "doc-1:000001"


def test_query_endpoint_invokes_workflow_dependency(tmp_path: Path) -> None:
    """The public query endpoint returns the graph's structured response contract."""

    class WorkflowStub:
        async def invoke(self, document_id: str, query: str) -> QueryResponse:
            return QueryResponse(
                document_id=document_id,
                query=query,
                route=AnalysisRoute.RETRIEVAL,
                summary="Retrieved 0 source-cited passage(s).",
            )

    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    app.dependency_overrides[get_analysis_workflow] = lambda: WorkflowStub()
    app.dependency_overrides[get_audit_repository] = lambda: repository
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/query", json={"doc_id": "doc-1", "query": "Find repayment clauses"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-1"
    request_id = response.headers["X-Request-ID"]
    saved_event = repository.get(request_id)
    assert saved_event is not None
    assert saved_event.status == "completed"
    assert saved_event.route == "retrieval"


def test_audit_repository_records_structured_event(tmp_path: Path) -> None:
    """SQLite audit storage keeps route, steps, token usage, and final output."""

    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    event = QueryAuditEvent(
        request_id="14c5c5f0-6db4-4a64-85f9-6d247e711e70",
        document_id="doc-1",
        query="Calculate debt-to-equity",
        route="quantitative",
        prompt_tokens=12,
        completion_tokens=3,
        total_tokens=15,
        latency_ms=42,
        status="completed",
    )

    repository.record(event)
    saved_event = repository.get(event.request_id)

    assert saved_event is not None
    assert saved_event.total_tokens == 15
    assert saved_event.route == "quantitative"
