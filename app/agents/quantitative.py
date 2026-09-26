"""Deterministic, source-bound financial ratio calculation agent."""

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.domain.analysis import CalculationInput, QuantitativeResult, RetrievedPassage


@dataclass(frozen=True)
class RatioDefinition:
    """A supported ratio with source-label aliases for its operands."""

    metric: str
    formula: str
    numerator_aliases: tuple[str, ...]
    denominator_aliases: tuple[str, ...]


RATIOS: tuple[RatioDefinition, ...] = (
    RatioDefinition("debt_to_equity", "Total Debt / Total Equity", ("total debt", "debt"), ("total equity", "equity")),
    RatioDefinition(
        "current_ratio",
        "Current Assets / Current Liabilities",
        ("current assets",),
        ("current liabilities",),
    ),
    RatioDefinition("debt_to_ebitda", "Total Debt / EBITDA", ("total debt", "debt"), ("ebitda",)),
    RatioDefinition("interest_coverage", "EBIT / Interest Expense", ("ebit",), ("interest expense", "interest")),
)


class CodeExecutionError(RuntimeError):
    """Raised when restricted calculation code cannot be validated or executed."""


class DeterministicCodeInterpreter:
    """Execute fixed Decimal arithmetic in an isolated Python subprocess.

    The interpreter receives no model-authored or user-authored Python. It emits
    a narrow code template, validates it against an AST allow-list, and runs it
    with isolated Python mode and a strict timeout. A production deployment
    should run this component in a dedicated jailed worker or microVM.
    """

    def __init__(self, timeout_seconds: int = 3) -> None:
        if not 1 <= timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be between 1 and 30")
        self._timeout_seconds = timeout_seconds

    def execute_division(self, numerator: Decimal, denominator: Decimal) -> tuple[str, str]:
        """Compute a decimal division using restricted, validated Python code."""

        code = (
            "from decimal import Decimal, ROUND_HALF_UP\n"
            f"numerator = Decimal({str(numerator)!r})\n"
            f"denominator = Decimal({str(denominator)!r})\n"
            "result = numerator / denominator\n"
            "print(result.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))\n"
        )
        self._validate_code(code)
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodeExecutionError("Calculation timed out") from exc
        if completed.returncode != 0:
            raise CodeExecutionError("Calculation failed, such as a zero denominator")
        value = completed.stdout.strip()
        if not value:
            raise CodeExecutionError("Calculation produced no output")
        return value, code

    @staticmethod
    def _validate_code(code: str) -> None:
        allowed_nodes = {
            ast.Module,
            ast.ImportFrom,
            ast.alias,
            ast.Assign,
            ast.Expr,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Constant,
            ast.BinOp,
            ast.Div,
            ast.Call,
            ast.Attribute,
            ast.keyword,
        }
        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise CodeExecutionError(f"Disallowed syntax in calculation template: {type(node).__name__}")
            if isinstance(node, ast.ImportFrom) and (
                node.module != "decimal" or any(alias.name not in {"Decimal", "ROUND_HALF_UP"} for alias in node.names)
            ):
                raise CodeExecutionError("Only Decimal imports are permitted")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"Decimal", "print"}:
                    continue
                if isinstance(node.func, ast.Attribute) and node.func.attr == "quantize":
                    continue
                raise CodeExecutionError("Only fixed Decimal arithmetic calls are permitted")


class QuantitativeAgent:
    """Calculate supported ratios from exact table values and source citations."""

    def __init__(self, interpreter: DeterministicCodeInterpreter | None = None) -> None:
        self._interpreter = interpreter or DeterministicCodeInterpreter()

    def analyze(self, query: str, passages: list[RetrievedPassage]) -> QuantitativeResult:
        """Identify a supported ratio, extract exact source operands, and execute code."""

        definition = self._select_ratio(query)
        if definition is None:
            return QuantitativeResult(
                status="unavailable",
                error="No supported deterministic ratio was requested. Supported ratios: debt-to-equity, current ratio, debt-to-EBITDA, interest coverage.",
            )
        numerator = self._find_metric(definition.numerator_aliases, passages)
        denominator = self._find_metric(definition.denominator_aliases, passages)
        if numerator is None or denominator is None:
            missing = "numerator" if numerator is None else "denominator"
            return QuantitativeResult(
                status="unavailable",
                metric=definition.metric,
                formula=definition.formula,
                error=f"Unable to calculate: no exact source-cited {missing} value was found in retrieved tables.",
            )
        numerator_value, numerator_input = numerator
        denominator_value, denominator_input = denominator
        try:
            value, executed_code = self._interpreter.execute_division(numerator_value, denominator_value)
        except CodeExecutionError as exc:
            return QuantitativeResult(
                status="failed",
                metric=definition.metric,
                formula=definition.formula,
                inputs=[numerator_input, denominator_input],
                error=str(exc),
            )
        return QuantitativeResult(
            status="completed",
            metric=definition.metric,
            formula=definition.formula,
            value=value,
            inputs=[numerator_input, denominator_input],
            executed_code=executed_code,
        )

    @staticmethod
    def _select_ratio(query: str) -> RatioDefinition | None:
        normalized = query.lower().replace("-", " ")
        aliases = {
            "debt_to_equity": ("debt to equity", "debt equity"),
            "current_ratio": ("current ratio",),
            "debt_to_ebitda": ("debt to ebitda",),
            "interest_coverage": ("interest coverage",),
        }
        return next((ratio for ratio in RATIOS if any(alias in normalized for alias in aliases[ratio.metric])), None)

    def _find_metric(
        self,
        aliases: tuple[str, ...],
        passages: list[RetrievedPassage],
    ) -> tuple[Decimal, CalculationInput] | None:
        for passage in passages:
            for line in passage.text.splitlines():
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if len(cells) < 2 or all(set(cell) <= {"-", ":", " "} for cell in cells):
                    continue
                label = cells[0].lower()
                if not any(re.search(rf"\b{re.escape(alias)}\b", label) for alias in aliases):
                    continue
                value = self._parse_decimal(cells[-1])
                if value is not None:
                    return value, CalculationInput(metric=cells[0], value=str(value), citation=passage.citation)
        return None

    @staticmethod
    def _parse_decimal(raw_value: str) -> Decimal | None:
        cleaned = raw_value.strip().replace(",", "").replace("$", "").replace("%", "")
        is_negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("() ")
        match = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cleaned)
        if not match:
            return None
        try:
            value = Decimal(cleaned)
            return -value if is_negative else value
        except InvalidOperation:
            return None
