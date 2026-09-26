"""Query routing agent with a deterministic fallback for unavailable LLMs."""

import asyncio
import re

from app.domain.analysis import AnalysisRoute, TokenUsage


class SupervisorAgent:
    """Classify a question into retrieval, compliance, quantitative, or both routes."""

    def __init__(self, api_key: str | None, model: str) -> None:
        self._api_key = api_key
        self._model = model

    async def route(self, query: str) -> tuple[AnalysisRoute, str, TokenUsage]:
        """Use Gemini for routing when configured, otherwise apply an explicit fallback."""

        if self._api_key:
            try:
                route, usage = await asyncio.to_thread(self._route_with_gemini, query)
                return route, "Gemini supervisor selected the analysis route.", usage
            except Exception:
                return self._heuristic_route(query), "Gemini routing was unavailable; deterministic route fallback used.", TokenUsage()
        return self._heuristic_route(query), "Deterministic route fallback used because no Gemini API key is configured.", TokenUsage()

    def _route_with_gemini(self, query: str) -> tuple[AnalysisRoute, TokenUsage]:
        from google import genai

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=(
                "Classify this financial-document question. Reply with exactly one token: "
                "retrieval, compliance, quantitative, or both. Choose both only if it asks for "
                f"both a compliance assessment and numerical computation. Question: {query}"
            ),
        )
        decision = (response.text or "").strip().lower()
        metadata = response.usage_metadata
        usage = TokenUsage(
            prompt_tokens=int(getattr(metadata, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(metadata, "candidates_token_count", 0) or 0),
            total_tokens=int(getattr(metadata, "total_token_count", 0) or 0),
        )
        return AnalysisRoute(decision), usage

    @staticmethod
    def _heuristic_route(query: str) -> AnalysisRoute:
        normalized = re.sub(r"\s+", " ", query.lower())
        compliance = any(term in normalized for term in ("compliance", "violation", "breach", "compliant", "rule"))
        quantitative = any(
            term in normalized
            for term in (
                "ratio",
                "calculate",
                "calculation",
                "debt-to-equity",
                "debt to equity",
                "current ratio",
                "debt to ebitda",
                "interest coverage",
            )
        )
        if compliance and quantitative:
            return AnalysisRoute.BOTH
        if compliance:
            return AnalysisRoute.COMPLIANCE
        if quantitative:
            return AnalysisRoute.QUANTITATIVE
        return AnalysisRoute.RETRIEVAL
