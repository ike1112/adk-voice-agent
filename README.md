# adk-voice-agent

## Overview

An AI-powered podcast generation system that researches the latest AI news for NASDAQ-listed US companies and automatically generates a multi-speaker podcast episode.

## What It Does

This agent orchestrates a complete workflow:

1. **Researches AI News** - Searches for the latest AI-related news about US-listed companies on NASDAQ
2. **Enriches with Financial Data** - Fetches current stock prices and daily changes for relevant companies
3. **Analyzes Sentiment** - Evaluates news headlines using keyword-based sentiment analysis
4. **Structures Findings** - Organizes all information into a well-formatted report with source attribution
5. **Generates Report** - Saves findings to `ai_research_report.md` with sourcing notes
6. **Creates Podcast Script** - Converts the report into a conversational dialogue between two hosts
7. **Produces Audio** - Generates multi-speaker podcast audio with distinct voices (Joe and Jane)

## Key Features

- **Smart Filtering**: Enforces quality by blocking searches to non-news sources (Wikipedia, Reddit, Medium, etc.)
- **Data Freshness**: Automatically filters search results to show only recent news (last week)
- **Resilient Processing**: Continues workflow even if some data points are missing (e.g., unavailable stock tickers)
- **Process Transparency**: Logs all callback actions and data sourcing decisions in the report
- **Multi-Speaker Audio**: Generates podcast with distinct voices using Gemini's text-to-speech API
- **Structured Output**: Uses Pydantic models for validated data structures and consistent formatting

## Technologies Used

- **Google ADK (Agent Development Kit)** - Multi-agent framework with callbacks
- **Gemini API** - LLM reasoning and text-to-speech audio generation
- **yfinance** - Real-time stock price data
- **Google Search** - News research capabilities
- **Pydantic** - Data validation and schema definitions

## Main Components

- `AINewsReport` - Structured schema for the research report
- `NewsStory` - Individual news item with context and financial data
- `root_agent` (ai_news_researcher) - Main orchestrator agent
- `podcaster_agent` - Audio generation specialist
- Callbacks for policy enforcement and response enrichment

## Setup Instructions

### Create a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Install Dependencies

```bash
# Install required packages from requirements.txt
pip install -r requirements.txt
```

### Configure API Keys

Set up your Google API credentials:

Update your GOOGLE_API_KEY inside the .env file
GOOGLE_API_KEY="your-api-key-here"


### Run the Agent

```bash
# Run the main agent
adk web --port 8000
```