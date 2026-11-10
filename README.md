
# RootCause AI

An AI-powered production incident investigation and
root-cause analysis platform for distributed systems.

## Problem Statement

Modern microservices generate large volumes of metrics,
logs, and distributed traces. Identifying the original
cause of a production incident often requires manually
correlating evidence across multiple services.

RootCause AI aims to automate evidence collection,
anomaly detection, root-cause candidate ranking,
and incident investigation.

## Planned Capabilities

- Distributed telemetry collection
- Anomaly detection and time-series analysis
- Microservice dependency graph construction
- Evidence-based root-cause ranking
- Agentic AI investigation using LangGraph
- Retrieval-augmented generation for incident runbooks
- Human-reviewed remediation recommendations
- Incident investigation dashboard
- Evaluation and cloud deployment

## Technology Stack

**Backend:** Python, FastAPI

**Observability:** OpenTelemetry, Jaeger,
Prometheus, OpenSearch

**Machine Learning:** scikit-learn, PyTorch

**Agentic AI:** LangGraph, MCP, RAG

**Frontend:** React, TypeScript

**Infrastructure:** Docker, AWS, GitHub Actions

## Project Status

Phase 0 — Repository initialization

## Development Approach

Learn → Implement → Test → Commit → Push

## Safety

RootCause AI begins with read-only incident
investigation. Automated remediation of production
systems is outside the initial project scope.
