from typing import Dict, List
import pathlib
import wave

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search, ToolContext
from google import genai
from google.genai import types
import yfinance as yf
from pydantic import BaseModel, Field


class NewsStory(BaseModel):
    """A single news story with its context."""
    company: str = Field(description="Company name associated with the story (e.g., 'Nvidia', 'OpenAI'). Use 'N/A' if not applicable.")
    ticker: str = Field(description="Stock ticker for the company (e.g., 'NVDA'). Use 'N/A' if private or not found.")
    summary: str = Field(description="A brief, one-sentence summary of the news story.")
    why_it_matters: str = Field(description="A concise explanation of the story's significance or impact.")
    financial_context: str = Field(description="Current stock price and change, e.g., '$950.00 (+1.5%)'. Use 'No financial data' if not applicable.")
    source_domain: str = Field(description="The source domain of the news, e.g., 'techcrunch.com'.")
    process_log: str = Field(description="populate the `process_log` field in the schema with the `process_log` list from the `google_search` tool's output." ) 

class AINewsReport(BaseModel):
    """A structured report of the latest AI news."""
    title: str = Field(default="AI Research Report", description="The main title of the report.")
    report_summary: str = Field(description="A brief, high-level summary of the key findings in the report.")
    stories: List[NewsStory] = Field(description="A list of the individual news stories found.")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "AI Research Report",
                    "report_summary": "Summary of AI news",
                    "stories": [
                        {
                            "company": "OpenAI",
                            "ticker": "PRIVATE",
                            "summary": "News summary",
                            "why_it_matters": "Impact explanation",
                            "financial_context": "Not Available",
                            "source_domain": "techcrunch.com",
                            "process_log": "Processing log"
                        }
                    ]
                }
            ]
        }
    }



def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Helper function to save audio data as a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
        

async def generate_podcast_audio(podcast_script: str, tool_context: ToolContext, filename: str = "'ai_today_podcast") -> Dict[str, str]:
    """
    Generates audio from a podcast script using Gemini API and saves it as a WAV file.

    Args:
        podcast_script: The conversational script to be converted to audio.
        tool_context: The ADK tool context.
        filename: Base filename for the audio file (without extension).

    Returns:
        Dictionary with status and file information.
    """
    try:
        client = genai.Client()
        prompt = f"TTS the following conversation between Joe and Jane:\n\n{podcast_script}"

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(speaker='Joe', 
                                                     voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore'))),
                            types.SpeakerVoiceConfig(speaker='Jane', 
                                                     voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')))
                        ]
                    )
                )
            )
        )

        data = response.candidates[0].content.parts[0].inline_data.data

        if not filename.endswith(".wav"):
            filename += ".wav"

        # ** BUG FIX **: This logic now runs for all cases, not just when the extension is added.
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / filename
        wave_file(str(file_path), data)

        return {
            "status": "success",
            "message": f"Successfully generated and saved podcast audio to {file_path.resolve()}",
            "file_path": str(file_path.resolve()),
            "file_size": len(data)
        }

    except Exception as e:
        error_msg = str(e)[:200]
        return {"status": "error", "message": f"Audio generation failed: {error_msg}"}



WHITELIST_DOMAINS = ["techcrunch.com", "venturebeat.com", "theverge.com", "technologyreview.com", "arstechnica.com"]

def filter_news_sources_callback(tool, args, tool_context):
    """Callback to enforce that google_search queries only use whitelisted domains."""
    if tool.name == "google_search":
        original_query = args.get("query", "")
        if any(f"site:{domain}" in original_query.lower() for domain in WHITELIST_DOMAINS):
            return None
        whitelist_query_part = " OR ".join([f"site:{domain}" for domain in WHITELIST_DOMAINS])
        args['query'] = f"{original_query} {whitelist_query_part}"
        print(f"MODIFIED query to enforce whitelist: '{args['query']}'")
    return None

def enforce_data_freshness_callback(tool, args, tool_context):
    """Callback to add a time filter to search queries to get recent news."""
    if tool.name == "google_search":
        query = args.get("query", "")
        # Adds a Google search parameter to filter results from the last week.
        if "tbs=qdr:w" not in query:
            args['query'] = f"{query} tbs=qdr:w"
            print(f"MODIFIED query for freshness: '{args['query']}'")
    return None

    
def get_financial_context(tickers: List[str]) -> Dict[str, str]:
    """
    Fetches the current stock price and daily change for a list of stock tickers
    using the yfinance library.

    Args:
        tickers: A list of stock market tickers (e.g., ["GOOG", "NVDA"]).

    Returns:
        A dictionary mapping each ticker to its formatted financial data string.
    """
    
    financial_data: Dict[str, str] = {}
    for ticker_symbol in tickers:
        try:
            # Create a Ticker object
            stock = yf.Ticker(ticker_symbol)

            # Fetch the info dictionary
            info = stock.info

            # Safely access the required data points
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            change_percent = info.get("regularMarketChangePercent")

            if price is not None and change_percent is not None:
                # Format the percentage and the final string
                change_str = f"{change_percent * 100:+.2f}%"
                financial_data[ticker_symbol] = f"${price:.2f} ({change_str})"
            else:
                # Handle cases where the ticker is valid but data is missing
                financial_data[ticker_symbol] = "Price data not available."

        except Exception:
            # This handles invalid tickers or other yfinance errors gracefully
            financial_data[ticker_symbol] = "Invalid Ticker or Data Error"

    return financial_data


def analyze_news_sentiment(headlines: List[str]) -> Dict[str, str]:
    """
    Analyzes the sentiment of news headlines and returns a sentiment label
    for each headline (Positive, Negative, or Neutral).

    This function uses a keyword-based approach to determine sentiment by
    counting positive and negative words in each headline.

    Args:
        headlines: A list of news headline strings to analyze.

    Returns:
        A dictionary mapping each headline to its sentiment label
        (Positive, Negative, or Neutral).

    Example:
        >>> headlines = ["Stock soars on breakthrough AI discovery",
        ...              "Company faces lawsuit over data breach"]
        >>> analyze_news_sentiment(headlines)
        {'Stock soars on breakthrough AI discovery': 'Positive',
         'Company faces lawsuit over data breach': 'Negative'}
    """

    # Define keyword lists for sentiment analysis
    positive_keywords = {
        'surge', 'soar', 'gain', 'rise', 'jump', 'rally', 'breakthrough',
        'success', 'wins', 'beats', 'exceeds', 'strong', 'growth', 'up',
        'profit', 'boost', 'advance', 'innovation', 'partnership', 'deal',
        'expand', 'launch', 'record', 'high', 'positive', 'outperform',
        'announces', 'unveils', 'milestone', 'achievement', 'revolutionary'
    }

    negative_keywords = {
        'fall', 'drop', 'plunge', 'decline', 'loss', 'losses', 'down',
        'crash', 'tumble', 'slump', 'weak', 'miss', 'misses', 'lawsuit',
        'scandal', 'crisis', 'failure', 'fails', 'probe', 'investigation',
        'concern', 'worry', 'threat', 'risk', 'warning', 'cut', 'layoff',
        'bankruptcy', 'debt', 'struggle', 'controversy', 'breach'
    }

    sentiment_results: Dict[str, str] = {}

    for headline in headlines:
        try:
            # Convert headline to lowercase for case-insensitive matching
            headline_lower = headline.lower()

            # Remove punctuation for better word matching
            headline_clean = re.sub(r'[^\w\s]', ' ', headline_lower)
            words = headline_clean.split()

            # Count positive and negative keywords
            positive_count = sum(1 for word in words if word in positive_keywords)
            negative_count = sum(1 for word in words if word in negative_keywords)

            # Determine sentiment based on keyword counts
            if positive_count > negative_count:
                sentiment = "Positive"
            elif negative_count > positive_count:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            sentiment_results[headline] = sentiment

        except Exception:
            # Handle any unexpected errors gracefully
            sentiment_results[headline] = "Neutral"

    return sentiment_results


def save_news_to_markdown(filename: str, content: str) -> Dict[str, str]:
    """
    Saves the given content to a Markdown file in the current directory.

    Args:
        filename: The name of the file to save (e.g., 'ai_news.md').
        content: The Markdown-formatted string to write to the file.

    Returns:
        A dictionary with the status of the operation.
    """
    try:
        if not filename.endswith(".md"):
            filename += ".md"
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / filename
        file_path.write_text(content, encoding="utf-8")
        return {
            "status": "success",
            "message": f"Successfully saved news to {file_path.resolve()}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {str(e)}"}


"""
Callbacks are Python functions that run at specific checkpoints in an agent's lifecycle, providing programmatic control over behavior.

Understanding ADK Callback types
ADK provides several callback points:

before_agent_callback: Runs before agent execution starts
after_agent_callback: Runs after agent execution completes
before_tool_callback: Runs before any tool is executed
after_tool_callback: Runs after any tool completes
before_model_callback: Runs before LLM calls
after_model_callback: Runs after LLM responses

"""


"""
Callback 1: Source filtering Callback (Before Tool Callback)
This callback demonstrates programmatic policy enforcement. It automatically blocks search queries that 
require the agent to fetch news from certain sources.

How It Works:
Interception: Runs before every google_search tool call
Query analysis: Examines the search query for blocked domains
Policy enforcement: Blocks searches targeting certain sources like Wikipedia, Reddit, Medium
Error response: Returns structured error messages when domains are blocked
Transparency: Logs allowed/blocked decisions for debugging
"""
BLOCKED_DOMAINS = [
    "wikipedia.org",      # General info, not latest news
    "reddit.com",         # Discussion forums, not primary news
    "youtube.com",        # Video content not useful for text processing
    "medium.com",         # Blog platform with variable quality
    "investopedia.com",   # Financial definitions, not tech news
    "quora.com",          # Q&A site, opinions not reports
]

def filter_news_sources_callback(tool, args, tool_context):
    """
    Callback: Blocks search requests that target certain domains which are not necessarily news sources.
    Demonstrates content quality enforcement through request blocking.
    """
    if tool.name == "google_search":
        query = args.get("query", "").lower()

        # Check if query explicitly targets blocked domains
        for domain in BLOCKED_DOMAINS:
            if f"site:{domain}" in query or domain.replace(".org", "").replace(".com", "") in query:
                print(f"BLOCKED: Domains from blocked list detected: '{query}'")
                return {
                    "error": "blocked_source",
                    "reason": f"Searches targeting {domain} or similar are not allowed. Please search for professional news sources."
                }

        print(f"ALLOWED: Professional source query: '{query}'")
        return None


"""
Callback 2: Response enhancement (After Tool Callback)
The next callback demonstrates a sophisticated pattern: response enhancement. Instead of blocking requests,
this callback enriches tool responses with additional metadata.


How this callback works
When this callback is triggered after the tool execution (google_search), the following actions are taken:

Callback trigger: Monitors when google_search tools finish execution.
Domain extraction: Automatically parses URLs from search results to identify source domains.
State management: Maintains a persistent log across multiple tool calls using tool_context.state.
Response transformation: Converts simple string responses into structured data with metadata.
Write to the report: Makes callback actions visible to the LLM through process logs. This log is written to the generated markdown report.
This pattern transforms your agent from a "black box" into a transparent, auditable system suitable for production deployment.
"""


def initialize_process_log(tool_context: ToolContext):
    """Helper to ensure the process_log list exists in the state."""
    if 'process_log' not in tool_context.state:
        tool_context.state['process_log'] = []

def inject_process_log_after_search(tool, args, tool_context, tool_response):
    """
    Callback: After a successful search, this injects the process_log into the response
    and adds a specific note about which domains were sourced. This makes the callbacks'
    actions visible to the LLM.
    """
    if tool.name == "google_search" and isinstance(tool_response, str):
        # Extract source domains from the search results
        urls = re.findall(r'https?://[^\s/]+', tool_response)
        unique_domains = sorted(list(set(urlparse(url).netloc for url in urls)))
        
        if unique_domains:
            sourcing_log = f"Action: Sourced news from the following domains: {', '.join(unique_domains)}."
            # Prepend the new log to the existing one for better readability in the report
            current_log = tool_context.state.get('process_log', [])
            tool_context.state['process_log'] = [sourcing_log] + current_log

        final_log = tool_context.state.get('process_log', [])
        print(f"CALLBACK LOG: Injecting process log into tool response: {final_log}")
        return {
            "search_results": tool_response,
            "process_log": final_log
        }
    return tool_response





"""
Notice how these instructions implement several best practices:

1. Structured Workflow: 5-step process from clarification to detailed discussion

2. Tool Citation Requirements: Agent must cite google_search and get_financial_context usage

3. Interactive Design: Prompts user for input at each stage rather than providing monologues

4. Scope Boundaries: Clear rules about staying focused on AI news for US-listed companies

5. Error Handling: Graceful responses when asked about off-topic subjects

This creates a much more reliable and user-friendly conversational experience.
"""


podcaster_agent = Agent(
    name="podcaster_agent",
    model="gemini-2.0-flash",
    instruction="""
    You are an Audio Generation Specialist. Your single task is to take a provided text script
    and convert it into a multi-speaker audio file using the `generate_podcast_audio` tool.

    Workflow:
    1. Receive the text script from the user or another agent.
    2. Immediately call the `generate_podcast_audio` tool with the provided script and the filename of 'ai_today_podcast'
    3. Report the result of the audio generation back to the user.
    """,
    tools=[generate_podcast_audio],
)

root_agent = Agent(
    name="ai_news_researcher",
    model="gemini-2.5-flash-native-audio-preview-09-2025", 
    instruction="""
    **Your Core Identity:**
    You are an AI News Podcast Producer. Your job is to orchestrate a complete workflow: find the latest AI news for US-listed
    companies on the NASDAQ, compile a report, write a script, and generate a podcast audio file, all while keeping the user informed.

    **Crucial Rules:**
    1.  **Resilience is Key:** If you encounter an error or cannot find specific information for one item (like fetching a stock ticker), you MUST NOT halt the entire process. Use a placeholder value like "Not Available", and continue to the next step. Your primary goal is to deliver the final report and podcast, even if some data points are missing.
    2.  **Scope Limitation:** Your research is strictly limited to US-listed companies on the NASDAQ exchange. All search queries and analysis must adhere to this constraint.
    3.  **User-Facing Communication:** Your interaction has only two user-facing messages: the initial acknowledgment and the final confirmation. All complex work must happen silently in the background between these two messages.

    **Understanding Callback-Modified Tool Outputs:**
    The `google_search` tool is enhanced by callbacks. Its final output is a JSON object with two keys:
    1.  `search_results`: A string containing the actual search results.
    2.  `process_log`: A list of strings describing the filtering actions performed.

    **Required Conversational Workflow:**
    1.  **Acknowledge and Inform:** The VERY FIRST thing you do is respond to the user with: "Okay, I'll start researching the latest AI news for NASDAQ-listed US companies. I will enrich the findings with financial data where available and compile a report for you. This might take a moment."
    2.  **Search (Background Step):** Immediately after acknowledging, use the `google_search` tool to find relevant news. Your query must be specifically tailored to find news about "AI" and "NASDAQ-listed US companies".
    3.  **Analyze & Extract Tickers (Internal Step):** Process search results to identify company names and their stock tickers. If a company is not on NASDAQ or a ticker cannot be found, use 'N/A'.
    4.  **Get Financial Data (Background Step):** Call the `get_financial_context` tool with the extracted tickers. If the tool returns "Not Available" for any ticker, you will accept this and proceed. Do not stop or report an error.
    5.  **Structure the Report (Internal Step):** Use the `AINewsReport` schema to structure all gathered information. If financial data was not found for a story, you MUST use "Not Available" in the `financial_context` field. You MUST also populate the `process_log` field in the schema with the `process_log` list from the `google_search` tool's output.
    6.  **Format for Markdown (Internal Step):** Convert the structured `AINewsReport` data into a well-formatted Markdown string. This MUST include a section at the end called "## Data Sourcing Notes" where you list the items from the `process_log`.
    7.  **Save the Report (Background Step):** Save the Markdown string using `save_news_to_markdown` with the filename `ai_research_report.md`.
    8.  **Create Podcast Script (Internal Step):** After saving the report, you MUST convert the structured `AINewsReport` data into a natural, conversational podcast script between two hosts, 'Joe' (enthusiastic) and 'Jane' (analytical).
    9.  **Generate Audio (Background Step):** Call the `podcaster_agent` tool, passing the complete conversational script you just created to it.
    10. **Final Confirmation:** After the audio is successfully generated, your final response to the user MUST be: "All done. I've compiled the research report, saved it to `ai_research_report.md`, and generated the podcast audio file for you."
    
    **CRITICAL: Output Format Requirements**
    When returning the final AINewsReport, you MUST output it as a JSON object with this exact structure:
    {
        "title": "AI Research Report",
        "report_summary": "...",
        "stories": [
            {
                "company": "Company Name",
                "ticker": "TICKER",
                "summary": "...",
                "why_it_matters": "...",
                "financial_context": "...",
                "source_domain": "domain.com",
                "process_log": "..."
            }
        ]
    }
    
    The stories array MUST contain dictionaries, NOT strings. Each story object must have all required fields.
    """,
    tools=[
        google_search,
        get_financial_context,
        save_news_to_markdown,
        AgentTool(agent=podcaster_agent) 
    ],
    output_schema=AINewsReport,
    before_tool_callback=[
        filter_news_sources_callback,
        enforce_data_freshness_callback
    ],
    after_tool_callback=[
        inject_process_log_after_search,
    ]
)