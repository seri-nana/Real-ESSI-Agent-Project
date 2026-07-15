from datetime import datetime

from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper


@tool
def save_tool(data: str) -> str:
    """Save structured research data to a text file."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = (
        f"--- Research Output ---\n"
        f"Timestamp: {timestamp}\n\n"
        f"{data}\n\n"
    )

    with open("research_output.txt", "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return "Data successfully saved."


search = DuckDuckGoSearchRun()

@tool
def search_tool(query: str) -> str:
    """Search the web for information."""
    return search.run(query)


api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=100,
)

wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
