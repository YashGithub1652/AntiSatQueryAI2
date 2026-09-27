\# SatQuery AI — System Architecture



\## 1. Overview



SatQuery AI is an agentic multimodal Earth-observation analysis system designed for natural-language interaction with remote-sensing imagery.



The system receives:



\- A natural-language query

\- A single satellite image

\- A bi-temporal image pair

\- A SAR + Optical image pair



The agent determines the required analysis task and routes the input to the appropriate specialist model or processing pipeline.



\---



\## 2. High-Level Architecture



```text

&#x20;                        USER

&#x20;                         │

&#x20;                         ▼

&#x20;               Natural-Language Query

&#x20;                         │

&#x20;                         ▼

&#x20;                ┌─────────────────┐

&#x20;                │  Query Parser   │

&#x20;                └────────┬────────┘

&#x20;                         │

&#x20;                         ▼

&#x20;                ┌─────────────────┐

&#x20;                │ Input Validator │

&#x20;                └────────┬────────┘

&#x20;                         │

&#x20;                         ▼

&#x20;                ┌─────────────────┐

&#x20;                │ Coregistration  │

&#x20;                └────────┬────────┘

&#x20;                         │

&#x20;                         ▼

&#x20;                ┌─────────────────┐

&#x20;                │ Task Classifier │

&#x20;                └────────┬────────┘

&#x20;                         │

&#x20;                         ▼

&#x20;                ┌─────────────────┐

&#x20;                │  Model Router   │

&#x20;                └────────┬────────┘

&#x20;                         │

&#x20;       ┌─────────────────┼──────────────────┐

&#x20;       │                 │                  │

&#x20;       ▼                 ▼                  ▼

&#x20;  SINGLE IMAGE      BI-TEMPORAL       SAR + OPTICAL

&#x20;       │                 │                  │

&#x20;       ▼                 ▼                  ▼

&#x20;  GeoChat-7B        ChangeFormerV6     SAR / Optical

&#x20;  RemoteCLIP        + LEVIR-CD         Fusion Pipeline

&#x20;       │                 │                  │

&#x20;       └─────────────────┼──────────────────┘

&#x20;                         ▼

&#x20;                ┌─────────────────┐

&#x20;                │   Aggregator    │

&#x20;                └────────┬────────┘

&#x20;                         │

&#x20;                         ▼

&#x20;                ┌─────────────────┐

&#x20;                │  Audit / Trace  │

&#x20;                └────────┬────────┘

&#x20;                         │

&#x20;                         ▼

&#x20;               Answer + Visual Evidence

