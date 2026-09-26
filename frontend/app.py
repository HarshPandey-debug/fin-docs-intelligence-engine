"""Analyst-facing Streamlit dashboard for the FinDocs intelligence API."""

import os
from collections.abc import Mapping
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("FINDOCS_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("FINDOCS_API_KEY", "")
REQUEST_TIMEOUT_SECONDS = 45.0


def api_request(method: str, path: str, **kwargs: Any) -> Mapping[str, Any]:
    """Call the backend and surface safe, human-readable API failures in the UI."""

    try:
        headers = dict(kwargs.pop("headers", {}))
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        response = httpx.request(method, f"{API_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", "The API rejected the request.")
        except ValueError:
            detail = "The API rejected the request."
        raise RuntimeError(str(detail)) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("Unable to reach the FinDocs API. Confirm the platform is running.") from exc


def render_analysis(response: Mapping[str, Any]) -> None:
    """Render structured evidence, agent trace, compliance findings, and math output."""

    st.markdown(response["summary"])
    if response.get("errors"):
        for error in response["errors"]:
            st.warning(error)

    quantitative = response.get("quantitative_result")
    if quantitative:
        st.subheader("Deterministic calculation")
        if quantitative["status"] == "completed":
            st.metric(quantitative.get("metric", "Result"), quantitative.get("value", "—"))
            st.caption(quantitative.get("formula", ""))
            inputs = quantitative.get("inputs", [])
            if inputs:
                st.dataframe(
                    [{"Metric": item["metric"], "Source value": item["value"], "Chunk": item["citation"]["chunk_id"]} for item in inputs],
                    use_container_width=True,
                    hide_index=True,
                )
            with st.expander("Validated calculation script"):
                st.code(quantitative.get("executed_code", ""), language="python")
        else:
            st.info(quantitative.get("error", "A deterministic calculation was not available."))

    findings = response.get("compliance_findings", [])
    if findings:
        st.subheader("Compliance findings")
        st.dataframe(
            [
                {
                    "Rule": finding["title"],
                    "Severity": finding["severity"].upper(),
                    "Outcome": finding["status"].upper(),
                    "Rationale": finding["rationale"],
                }
                for finding in findings
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(f"Source citations ({len(response.get('citations', []))})"):
        for citation in response.get("citations", []):
            location = citation.get("section") or f"Page {citation.get('page_number') or 'not available'}"
            st.markdown(f"**{citation['source_filename']} · {location}** · `{citation['chunk_id']}`")
            st.caption(citation["excerpt"])

    with st.expander("Agent execution trace"):
        for step in response.get("steps", []):
            st.markdown(f"**{step['name'].title()} · {step['status']}** — {step['detail']}")
        usage = response.get("token_usage", {})
        st.caption(f"LLM tokens — prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)}")


def render_status(document_id: str) -> None:
    """Fetch and display current ingestion state without blocking the dashboard."""

    try:
        status_payload = api_request("GET", f"/api/v1/documents/{document_id}")
    except RuntimeError as exc:
        st.caption(str(exc))
        return
    current_status = status_payload["status"]
    if current_status == "completed":
        st.success(f"Ingestion complete: {status_payload.get('chunk_count', 0)} retrievable chunks.")
    elif current_status == "failed":
        st.error(status_payload.get("error") or "Ingestion failed.")
    else:
        st.info(f"Document ingestion is {current_status}.")


st.set_page_config(page_title="FinDocs Intelligence", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """<style>
    .block-container { max-width: 1320px; padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 2.1rem; }
    </style>""",
    unsafe_allow_html=True,
)

if "document_id" not in st.session_state:
    st.session_state.document_id = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("FinDocs Compliance Intelligence")
st.caption("Evidence-bound financial document analysis with deterministic calculations and a full agent trace.")

with st.sidebar:
    st.header("Document workspace")
    uploaded_file = st.file_uploader("Upload a SEC filing, credit agreement, or financial CSV", type=["pdf", "csv"])
    if uploaded_file and st.button("Ingest document", type="primary", use_container_width=True):
        with st.spinner("Sending document to the ingestion pipeline..."):
            try:
                upload_response = api_request(
                    "POST",
                    "/api/v1/upload",
                    files={"document": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")},
                )
                st.session_state.document_id = upload_response["document_id"]
                st.session_state.chat_history = []
                st.success(f"Accepted {uploaded_file.name}")
            except RuntimeError as exc:
                st.error(str(exc))

    st.session_state.document_id = st.text_input(
        "Active document ID",
        value=st.session_state.document_id,
        placeholder="Upload a document or paste its ID",
    )
    if st.session_state.document_id:
        if st.button("Refresh ingestion status", use_container_width=True):
            render_status(st.session_state.document_id)
        elif st.session_state.document_id:
            render_status(st.session_state.document_id)
    st.divider()
    st.caption(f"API endpoint: {API_URL}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_analysis(message["content"])

question = st.chat_input(
    "Ask about covenants, repayment penalties, compliance, or a supported financial ratio…",
    disabled=not bool(st.session_state.document_id),
)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Running the multi-agent analysis workflow..."):
            try:
                analysis = api_request("POST", "/api/v1/query", json={"doc_id": st.session_state.document_id, "query": question})
                render_analysis(analysis)
                st.session_state.chat_history.append({"role": "assistant", "content": analysis})
            except RuntimeError as exc:
                st.error(str(exc))

if not st.session_state.document_id:
    st.info("Upload a PDF/CSV or enter a document ID to begin analysis.")
