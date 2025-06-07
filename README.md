[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)  
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Mac-blue)](#)
---

# 🤖 BrowserAgent2: Automated Browser Agent with Gemini & MCP

**BrowserAgent2** is an automated agent that executes user instructions on a Chrome browser using Playwright, orchestrated by a Browser MCP Server (Model Context Protocol, SSE-based) and powered by Gemini for intelligent tool selection and reasoning.

---

## 🚀 Features
- Natural language command execution in a real browser
- Gemini LLM for instruction analysis and tool selection
- Modular MCP server (SSE protocol)
- Multi-turn conversation with context
- Cross-platform: Windows & Mac (Linux coming soon)

---

## 📁 Directory Structure
- `main.py` — Main entry point for the agent
- `browserMCP/browser_mcp_sse.py` — MCP server (must be started separately)
- `config/` — Configuration files (MCP server, models, profiles)
- `requirements.txt` — Python dependencies
- `action/`, `agent/`, `memory/`, etc. — Core modules

---

## ⚡ Quickstart

### 1. Clone the Repository
```cmd
# In your terminal
cd BrowserAgent2
```

### 2. Install [uv](https://github.com/astral-sh/uv) (if not already installed)
```cmd
pip install uv
```

### 3. Install Python Dependencies
```cmd
uv pip install -r requirements.txt
```

### 4. Download the spaCy Model
```cmd
uv python -m spacy download en_core_web_sm
```

### 5. Set Up Environment Variables
Create a `.env` file in the project root with your Gemini API key:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```

---

## 🖥️ Running the Application

### 1. Start the Browser MCP Server (in a new terminal)
```cmd
uv python browserMCP/browser_mcp_sse.py
```

### 2. Start the Agent (in another terminal)
```cmd
uv python main.py
```

---

## 💡 Example Query
```
Open https://inkers.ai and click on the Demo Button
```

---

## ❓ FAQ
- **Q:** Can I use this on Linux?  
  **A:** Linux support is coming soon!
- **Q:** What browser does it use?  
  **A:** Chrome via Playwright.
- **Q:** How do I stop the agent?  
  **A:** Type `exit` or `quit` at any prompt.

---

## 📄 License
MIT License @ Saish Shetty 2025


