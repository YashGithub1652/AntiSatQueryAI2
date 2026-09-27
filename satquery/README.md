\# 🛰️ SatQuery AI



\### An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries



\*\*Smart India Hackathon — Problem Statement 26167\*\*



SatQuery AI is an agentic multimodal Earth-observation analysis system that allows users to interact with satellite imagery through natural-language queries.



The system analyzes the user's query and uploaded imagery, identifies the required remote-sensing task, routes the request to the appropriate specialist model or pipeline, and returns an interpretable result together with an observable execution trace.



\---



\## 🚀 Core Capabilities



\- 🛰️ Single-image remote-sensing VQA

\- 📝 Satellite-image captioning

\- 🎯 Text-guided visual grounding

\- 🔄 Bi-temporal change detection

\- 📡 SAR + Optical analysis

\- 🗺️ GeoTIFF and remote-sensing metadata ingestion

\- 🤖 Agentic task classification

\- 🔀 Automatic model routing

\- 🔎 Evidence-oriented analysis output

\- 📋 Node-level execution trace

\- 🌐 REST API

\- ⚡ WebSocket-based live execution updates

\- 📊 Analysis history and reporting workflow



\---



\# 🧠 Agentic Architecture



```text

&#x20;                   Natural-Language Query

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │  Query Parser   │

&#x20;                   └────────┬────────┘

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │ Input Validator │

&#x20;                   └────────┬────────┘

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │ Coregistration  │

&#x20;                   └────────┬────────┘

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │ Task Classifier │

&#x20;                   └────────┬────────┘

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │  Model Router   │

&#x20;                   └────────┬────────┘

&#x20;                            │

&#x20;         ┌──────────────────┼──────────────────┐

&#x20;         ▼                  ▼                  ▼

&#x20;    Single Image       Bi-temporal       SAR + Optical

&#x20;         │                  │                  │

&#x20;         ▼                  ▼                  ▼

&#x20;     GeoChat-7B        ChangeFormerV6     SAR / Optical

&#x20;     RemoteCLIP        + LEVIR-CD         Fusion Pipeline

&#x20;         │                  │                  │

&#x20;         └──────────────────┼──────────────────┘

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │   Aggregator    │

&#x20;                   └────────┬────────┘

&#x20;                            ▼

&#x20;                   ┌─────────────────┐

&#x20;                   │  Audit / Trace  │

&#x20;                   └────────┬────────┘

&#x20;                            ▼

&#x20;                 Answer + Visual Evidence

```



\---



\## 🛠️ Technology Stack



\- Python

\- FastAPI

\- Uvicorn

\- PyTorch

\- Transformers

\- RemoteCLIP

\- GeoChat-7B

\- ChangeFormerV6

\- RSVG / MGVLF

\- Rasterio

\- HTML5

\- CSS3

\- JavaScript

\- WebSocket



\---



\## 🔬 Supported Analysis



\### Single Image

GeoChat-7B + RemoteCLIP for remote-sensing VQA and scene understanding.



\### Bi-Temporal

ChangeFormerV6 with LEVIR-CD for detecting changes between two images.



\### Visual Grounding

RSVG / MGVLF for text-guided localization of objects and regions.



\### SAR + Optical

SAR and optical feature processing with a cross-modal fusion pipeline.



\---



\## ⚠️ Validation Status



\*\*Verified:\*\*

\- GeoChat-7B base inference

\- RemoteCLIP ViT-B/32

\- ChangeFormerV6 + LEVIR-CD checkpoint

\- Visual grounding inference

\- Agentic task classification and routing

\- WebSocket execution trace

\- GeoTIFF ingestion

\- Visual-only PNG/JPEG change comparison



\*\*Integration / validation in progress:\*\*

\- SAR-optical fusion training

\- BigEarthNet adaptation

\- Full RSVG benchmark evaluation

\- Confidence calibration



\---



\## 🎥 Demo



Recorded project demonstration:



\*\*Coming soon\*\*



\---



\## 🛰️ Smart India Hackathon



\*\*Problem Statement:\*\* 26167



\*\*SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries\*\*


