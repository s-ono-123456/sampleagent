# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4.1-nano")

async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["sample_services/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8000
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
            "playwright": {
                # make sure to start your playwright server
                "command": "npx",
                "args": [
                    "@playwright/mcp@latest"
                ],
                "transport": "stdio"
            }
        }
    ) as client:
        tools = client.get_tools()
        agent = create_react_agent(model, tools)
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
        print("Math Response:")
        for message in math_response["messages"]:
            print(f"Type: {message.type}, Content: {message.content}")
        print("Weather Response:")
        for message in weather_response["messages"]:
            print(f"Type: {message.type}, Content: {message.content}")

if __name__ == "__main__":
    asyncio.run(main())
