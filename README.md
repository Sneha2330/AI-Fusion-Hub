# 🚀 AI-Fusion-Hub

AI-Fusion-Hub is a multi-agent AI application built using Python and Azure OpenAI services. The platform combines audio transcription, Retrieval-Augmented Generation (RAG), vector search, and AI-powered SQL querying into a single intelligent workspace.

The application enables users to transcribe audio files, query structured and unstructured data using natural language, retrieve context-aware information through RAG pipelines, and leverage Azure OpenAI models for intelligent responses and data exploration.

---

## 🎯 Project Overview

AI-Fusion-Hub brings together multiple AI capabilities into a unified solution:

- Convert speech into text using Azure OpenAI Whisper.
- Perform semantic search using vector embeddings.
- Query structured databases through AI-powered SQL agents.
- Retrieve relevant information using Retrieval-Augmented Generation (RAG).
- Generate context-aware responses using Azure OpenAI.
- Store and retrieve knowledge from SQLite and Vector Stores.

---

## ✨ Features

- 🎤 Audio Transcription using Azure OpenAI Whisper
- 🤖 Conversational AI powered by Azure OpenAI
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic Search using Vector Store
- 🗃️ AI-powered SQL Query Agent
- 📄 Context-Aware Document Retrieval
- 🧠 Knowledge Base Question Answering
- ☁️ Azure OpenAI Integration
- 💾 SQLite Database Support
- ⚡ Fast Python-based Architecture

---

## 🛠️ Technology Stack

- Python
- Azure OpenAI
- Whisper
- SQLite
- Vector Database
- RAG Architecture
- Pandas
- OpenAI SDK

---

## ⚙️ Setup & Run

### Install Python

- Install Python 3.12 or above

### Verify Installation

```bash
python -V
python -m pip --version
```

### Upgrade Pip

```bash
python -m pip install --upgrade pip
```

### Create Virtual Environment

```bash
python -m venv gradio-env
```

### Activate Virtual Environment

```bash
gradio-env\Scripts\activate
```

### Install Gradio

```bash
pip install gradio
```

or

```bash
python -m pip install gradio
```

### Verify Gradio Installation

```bash
pip show gradio
```

### Install Required Packages

```bash
pip install openai azure-core fastapi uvicorn python-multipart
```

### Install Document Support Packages

```bash
pip install pypdf2 python-docx numpy
```

### Upgrade HTTPX

```bash
pip install --upgrade httpx
```

### Create Environment File

Create a `.env` file and add Azure OpenAI credentials.

### Test Azure Connection

```bash
python azure_test.py
```

### Run Application

```bash
python app.py
```

### Whisper Test

```bash
python whisper_test.py
```
