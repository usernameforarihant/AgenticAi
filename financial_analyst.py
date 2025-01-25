from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

web_search_agent=Agent(name="WebSearchAgent",role="Search web for information",
model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
tools=[DuckDuckGo()],
instructions=['Always include sources'],
show_tools_calls=True, 
markdown=True
)

# #Financial Agent
# finance_agent=Agent(
#     name='financeagent',
#     model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
#     tools=[YFinanceTools(stock_price=True,analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
#     instructiions=['Use tables to display data'],
#     show_tools_calls=True,
#     markdown=True)

#Financial Agent
finance_agent=Agent(
    name='financeagent',
    model=Groq(id='llama3-70b-8192', api_key="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"),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructiions=['Use tables to display data'],
    show_tools_calls=True,
    markdown=True)

multi_ai_agent=Agent(team=[web_search_agent,finance_agent],
                     instructions=['Always include sources','use tables to show data' ],
                     show_tool_calls=True,
                     markdown=True)

multi_ai_agent.print_response("Summarize analyst recommendation and share latest  news for NVIDIA",stream=True)