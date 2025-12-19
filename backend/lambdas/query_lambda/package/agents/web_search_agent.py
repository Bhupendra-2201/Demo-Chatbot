import requests
from os import getenv
import logging
from datetime import datetime, timedelta
# Use local core definitions
try:
    from .core import Agent, tool
except ImportError:
    from core import Agent, tool

LOG = logging.getLogger("web_search_agent")
LOG.setLevel(logging.INFO)

# Configuration
LLM_MODEL_ID = getenv("LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

# Define http_request locally as it was missing
@tool
def http_request(url: str):
    """
    Makes an HTTP GET request to the specified URL.
    Args:
        url: The URL to fetch.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text[:2000] # Limit response size
    except Exception as e:
        return f"Error fetching URL: {e}"

WEB_SEARCH_SYSTEM_PROMPT = """
You are a web search agent. Your task is to search the web for information based on the user query.

You have access to the following tools:
- search_ddg: To search the DuckDuckGo for instant answers
- search_wiki: To search Wikipedia for information on a specific topic
- search_yahoo_finance: To search Yahoo Finance for stock market information
- rewrite_user_query: To rewrite the user query
- http_request: To make HTTP requests to the web incase other tools are not able to provide the information

Key Responsibilities:
- Rewrite the user query if needed
- Search the web for information using available tools
- Once you have sufficient information, SUMMARIZE the findings and answer the user's question directly.
- Do NOT call a summarization tool; do it yourself.
"""

date = datetime.now()
next_date = datetime.now() + timedelta(days=30)
year = date.year
month = date.month
month_label = date.strftime('%B')
next_month_label = next_date.strftime('%B')
day = date.day


def callback_handler(**kwargs):
    tool_use_ids = []
    if "data" in kwargs:
        # Log the streamed data chunks
        print(kwargs["data"])
    elif "current_tool_use" in kwargs:
        tool = kwargs["current_tool_use"]
        if tool["toolUseId"] not in tool_use_ids:
            # Log the tool use
            print(f"\n[Using tool: {tool.get('name')}]")
            tool_use_ids.append(tool["toolUseId"])


@tool
def web_search_agent(user_query):
    # Pass model_id explicitly instead of object
    agent = Agent(name="WebSearch", instructions=WEB_SEARCH_SYSTEM_PROMPT, model_id=LLM_MODEL_ID,
                tools=[ rewrite_user_query, search_ddg, search_wiki, search_yahoo_finance, http_request ]
                )
    # The 'run' method in core.py takes (message, history=...).
    # We adapt to that signature.
    agent_response = agent.run(user_query, history=[])
    return agent_response

@tool
def search_ddg(query):
    # Use the official library for better stability and rate limit handling
    try:
        from duckduckgo_search import DDGS
        import random
        import time
        
        # Add random delay to mimic human behavior
        time.sleep(random.uniform(2, 5))
        
        with DDGS() as ddgs:
            # text() returns an iterator, convert to list
            search_results = list(ddgs.text(query, max_results=5))
            
        if not search_results:
            return "No results found."
            
        return search_results
    except ImportError:
        return "duckduckgo-search library not installed."
    except Exception as e:
        return f"DuckDuckGo search failed: {e}"

@tool
def search_wiki(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "format": "json", "list": "search", "srsearch": query}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error performing Wiki search: {e}"

@tool
def search_yahoo_finance(query):
    url = "https://query2.finance.yahoo.com/v8/finance/chart/"
    params = {"q": query, "interval": "1d", "range": "1d"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error performing Yahoo Finance search: {e}"


@tool
def summarize_search_results(search_data, user_query):
    summarizer_system_prompt = f""" You are a search results summarizer. Given the search results and a user query,
                        Your task is to provide a concise summary of the search results based on the user query.
                        Remember todays year is {year} and the month is {month_label} and day is {day}.
                        The summary should be no more than 60 sentences.
                        Do not include any other text or tags in the response.
                        The search results are available in the <web_search_results> tags.
                        Your summarized output should be placed within the <summarize> tags
                        """
    # Create a clear prompt for the summarizer
    user_message = f"""<web_search_results>
{search_data}
</web_search_results>

User Query: {user_query}"""

    summarizer_agent = Agent(name="Summarizer", instructions=summarizer_system_prompt, model_id=LLM_MODEL_ID)
    summarized = summarizer_agent.run(user_message, history=[])
    
    return summarized


@tool
def rewrite_user_query(chat_history):
    print(f'In Query rewrite = {chat_history}')
    system_prompt = f""" You are a query rewriter. Given a user query, Your task is to step back and paraphrase a question to a more generic 
                        step-back question, which is easier to answer. 
                        Remember todays year is {year} and the month is {month_label} and day is {day}.
                        <instructions>
                        The entire chat history is provided to you
                        you should identify the user query from the provided chat history
                        You should then rewrite the user query so we get accurate search results.
                        The rewritten user query should be wrapped in <user-query></user-query> tags.
                        Do not include any other text or tags in the response.
                        </instructions>
                        """
    
    agent = Agent(name="Rewriter", instructions=system_prompt, model_id=LLM_MODEL_ID)
    rewritten_query = agent.run(chat_history, history=[])
    print(f'reformatted search_query text = {rewritten_query}')
    return rewritten_query


# if __name__ == '__main__':
#     print(web_search_agent(' Amazon '))
