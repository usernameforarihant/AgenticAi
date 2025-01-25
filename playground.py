import os
from dotenv import load_dotenv
import phi
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.playground import Playground,serve_playground_app
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

load_dotenv()

phi.api.key = os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name="WebSearchAgent",
    role="Search web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=['Always include sources'],
    show_tools_calls=True, 
    markdown=True
)

finance_agent = Agent(
    name='financeagent',
    model=Groq(id='llama3-70b-8192'),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=['Use tables to display data'],
    show_tools_calls=True,
    markdown=True
)

app = Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__ == '__main__':
    try:
        print("Starting Playground Server...")
        serve_playground_app("playground:app", host='localhost', port=7777, reload=True)
    except Exception as e:
        print(f"Server startup error: {e}")