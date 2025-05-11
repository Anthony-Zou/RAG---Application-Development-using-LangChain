# RAG & LangChain Notebooks

This repository contains a collection of Jupyter notebooks demonstrating various techniques for working with Large Language Models (LLMs) using LangChain, with a focus on Retrieval-Augmented Generation (RAG) and other advanced LLM techniques.

## Overview

These notebooks provide practical examples and implementations of:

- Creating and using prompt templates
- Working with agents and tools
- Implementing output parsers
- Building advanced agent systems

## Technologies Used

- Python 3.x
- LangChain
- Google Gemini AI
- OpenAI
- Various NLP and search tools

## Prerequisites

- Python 3.8+
- API keys for:
  - OpenAI
  - Google (for Gemini AI)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/RAG_LangChain.git
cd RAG_LangChain
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file in the root directory):

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Notebook Descriptions

### 1. Prompts Demo (`1.prompts-demo.ipynb`)

- Introduction to prompt templates and chat prompt templates
- Demonstrates various ways to format prompts
- Examples of few-shot learning with prompts

### 2. Agents and Tools (`2.agents-tools.ipynb`)

- Introduction to LangChain agents
- Using tools like calculators, search engines
- Creating custom tools
- Zero-shot ReAct agent implementation

### 3. Advanced Agents (`3.agents-deep.ipynb`)

- Deeper exploration of agent capabilities
- Conversational ReAct agents with memory
- DocStore agents for knowledge retrieval
- Self-ask with search implementation

### 5. Output Parsers (`5.output_parsers.ipynb`)

- Transform LLM outputs into structured formats
- JSON output parsing
- Comma-separated list parsing
- Structured output parsing

## Usage Examples

Each notebook contains detailed examples and step-by-step explanations. To run a notebook:

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open any of the notebooks and run the cells sequentially.

## Contributing

Contributions, issues, and feature requests are welcome!

## License

[Your License Choice]
