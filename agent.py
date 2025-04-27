from openai import OpenAI
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from vector_store_loader import load_vector_stores
import operator
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing import Any
from langchain_core.runnables import ConfigurableField
from langgraph.graph import END, START
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage, AIMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import InjectedState
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing import TypedDict
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# 環境変数の設定
# os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="sampleagent"
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

MODEL_NAME = "gpt-4.1-nano"
embedding_model_name = "pkshatech/GLuCoSE-base-ja-v2"

llm = init_chat_model(MODEL_NAME)

tool = TavilySearch(max_results=2)
tools = [tool]

# ツールは直接呼び出すこともできる。
# tool.invoke("What's a 'node' in LangGraph?")

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    # メッセージタイプを管理するための状態クラス
    # `add_messages` 関数はこの状態キーの更新方法を定義します
    # (この場合、メッセージを上書きせずリストに追加します)
    messages: Annotated[list, add_messages]

# chatbot関数の定義
# この関数は現在の状態(メッセージ履歴)を受け取り、LLMの応答を含む新しい状態を返します
# state: 現在の会話の状態（メッセージリストを含む）
# 戻り値: LLMからの応答メッセージを含む更新された状態
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


class BasicToolNode:
    """最後のAIMessageでリクエストされたツールを実行するノード。"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

def route_tools(
    state: State,
):
    """
    最後のメッセージにツール呼び出しがある場合はToolNodeにルーティングするために条件付きエッジで使用します。
    そうでない場合はENDへルーティングします。
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

def gragh_build():
    # メインのグラフ構築
    # ステートグラフは状態遷移を管理します
    # llm変数はanthropicのClaude 3.5 Sonnetモデルを初期化しています
    graph_builder = StateGraph(State)

    # chatbotノード関数が現在のStateを入力として受け取り
    # 「messages」キーの下に更新されたメッセージリストを含む辞書を返すことに注目してください。
    # これはすべてのLangGraphノード関数の基本的なパターンです。

    tool_node = BasicToolNode(tools=[tool])

    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph_builder.add_edge("tools", "chatbot")

    # `tools_condition` 関数は、チャットボットがツールの使用を要求する場合は "tools" を返し、
    # 直接応答する場合は "END" を返します。この条件付きルーティングはメインのエージェントループを定義します。
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # 以下の辞書を使用すると、条件の出力を特定のノードとして解釈するようにグラフに指示できます
        # デフォルトでは恒等関数ですが、"tools"以外の名前のノードを使用したい場合は、
        # 辞書の値を他のものに更新できます
        # 例："tools": "my_tools"
        {"tools": "tools", END: END},
    )
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    

    return graph

graph = gragh_build()

# グラフ実行
user_input = "受注処理の詳細を教えてください。"
config = {"configurable": {"thread_id": "1"}}

# The config is the **second positional argument** to stream() or invoke()!
# Streamだとストリーミングモードで返却される。
# Invokeだと一度に全てのメッセージが返却される。
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()


user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
