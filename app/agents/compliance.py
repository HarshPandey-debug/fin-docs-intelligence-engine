"""Strict JSON-rule compliance agent that never infers facts beyond retrieved text."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.analysis import ComplianceFinding, RetrievedPassage


class ComplianceRule(BaseModel):
    """Declarative rule evaluated only with case-insensitive string matching."""

    id: str
    title: str
    severity: str
    when_any: list[str] = Field(min_length=1)
    require_any: list[str] = Field(min_length=1)
    pass_rationale: str
    fail_rationale: str


class ComplianceAgent:
    """Apply predefined rules to retrieved text and return only evidenced findings."""

    def __init__(self, rules_path: Path) -> None:
        self._rules = self._load_rules(rules_path)

    def analyze(self, passages: list[RetrievedPassage]) -> list[ComplianceFinding]:
        """Evaluate every triggered rule against its matching retrieved passages."""

        findings: list[ComplianceFinding] = []
        for rule in self._rules:
            triggered = [passage for passage in passages if self._has_any(passage.text, rule.when_any)]
            if not triggered:
                continue
            satisfied = [passage for passage in triggered if self._has_any(passage.text, rule.require_any)]
            findings.append(
                ComplianceFinding(
                    rule_id=rule.id,
                    title=rule.title,
                    severity=rule.severity,
                    status="pass" if satisfied else "flag",
                    rationale=rule.pass_rationale if satisfied else rule.fail_rationale,
                    citations=[passage.citation for passage in (satisfied or triggered)],
                )
            )
        return findings

    @staticmethod
    def _load_rules(rules_path: Path) -> list[ComplianceRule]:
        try:
            raw_rules = json.loads(rules_path.read_text(encoding="utf-8"))
            return [ComplianceRule.model_validate(rule) for rule in raw_rules]
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Unable to load compliance rules: {exc}") from exc

    @staticmethod
    def _has_any(text: str, terms: list[str]) -> bool:
        normalized = text.lower()
        return any(term.lower() in normalized for term in terms)
