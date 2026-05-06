# CareerPilot AI Job Assistant

CareerPilot is a chat-based AI assistant designed to help users manage their job search, explore live job opportunities, and retrieve market data. The system integrates multiple services into a single conversational interface using modern AI techniques.

---

## Overview

This application combines:

- API integration (Marketstack)
- Semantic search (ChromaDB + embeddings)
- MCP server connection (LinkedIn)
- GPT-based response generation
- Guardrails for safe interaction
- Chat-based UI using Gradio

The assistant is designed with a **friendly career coach personality**, providing clear, supportive, and professional responses.

---

## Chat Client Design

The chat interface is implemented using **Gradio**, providing:

- A conversational UI
- Real-time responses
- Persistent chat history (memory across the session)

### Personality

The assistant, **CareerPilot**, acts as:

> A warm, supportive career coach who helps users navigate job searching, understand opportunities, and make informed decisions.

---

## System Architecture

User Input
↓
Routing Logic
↓
| Service 1 | Service 2 | Service 3 |

↓
GPT Response Formatting
↓
Final Answer


---

##  Services

### 🔹 Service 1: Marketstack API (External API)

- Retrieves real-time stock market data
- Uses the Marketstack API
- Output is rewritten using GPT into natural language

**Example queries:**
- "Show me Apple stock price"
- "Compare Apple and Microsoft"
- "Show top performing stocks"

---

### 🔹 Service 2: Semantic Job Tracker (ChromaDB)

- Uses a CSV dataset (`Job_Search_Tracker.csv`)
- Converts job entries into embeddings using SentenceTransformers
- Stores data in ChromaDB with file persistence
- Supports semantic and analytical queries

**Example queries:**
- "Which jobs did I apply for?"
- "How many interviews did I get?"
- "Which site is more successful?"

---

### 🔹 Service 3: LinkedIn MCP Server

- Connects to LinkedIn via MCP server
- Uses dynamic tool discovery
- Requires Browserbase for execution
BROWSERBASE_API_KEY
BROWSERBASE_CDP_URL

**Example queries:**
- "Find Project Manager jobs on LinkedIn"
- "Search Program Manager jobs in Toronto"


---

## 🤖 GPT Integration

GPT is used to:

- Convert raw API output into natural language
- Improve readability of responses
- Maintain consistent conversational tone

---

## 🛡️ Guardrails

The system prevents:

- Access to system prompts
- Prompt injection attacks
- Responses on restricted topics:
  - Cats or dogs
  - Horoscopes / zodiac signs
  - Taylor Swift

----

## 📁 Project Structure
assignment_chat/
│
├── data/
│ └── Job_Search_Tracker.csv
│
├── chroma_job_tracker/
│
├── final_job_assistant3.py
├── gradio_job_assistant.py
├── requirements.txt
├── .env
└── README.md


---

## 🔐 Environment Variables

Create a `.env` file:

OPENAI_API_KEY=your_openai_key
MARKETSTACK_API_KEY=your_marketstack_key
BROWSERBASE_API_KEY=your_browserbase_key
BROWSERBASE_CDP_URL=your_browserbase_url

## ▶️ How to Run

### 1. Install dependencies


pip install -r requirements.txt

### 2. Run the chat UI


python gradio_job_assistant_frontend.py

### 3. Open in browser


http://127.0.0.1:7860


---

## Example Demo Flow

1. API Service  
   → "Show me Apple stock price"

2. Semantic Search  
   → "Which jobs did I apply for?"

3. MCP  
   → "Find Project Manager jobs on LinkedIn"

4. Guardrails  
   → "Tell me about cats"

---

## Implementation Decisions

### Why ChromaDB?
- Lightweight and supports persistence
- Easy to integrate for semantic search

### Why SentenceTransformers?
- Efficient embedding generation
- Good balance of speed and accuracy

### Why Gradio?
- Simple chat UI
- Supports memory
- Quick to deploy

### Why GPT formatting?
- Improves readability
- Meets assignment requirement to transform API output

### Why MCP?
- Demonstrates tool-based AI architecture
- Enables dynamic external integrations

---

## Limitations

- LinkedIn MCP depends on Browserbase configuration
- API rate limits may apply
- Semantic search quality depends on dataset quality

---

## Summary

This project demonstrates:

- Multi-service AI system design
- Semantic search implementation
- API integration
- MCP server usage
- Safe and structured conversational AI

---

## Author

Hema Dawonauth  
