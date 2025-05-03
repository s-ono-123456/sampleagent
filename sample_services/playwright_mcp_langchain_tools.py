import json
import re
import os
import sys
import time  # 時間計測用に追加
import operator
import datetime  # 日付時刻の処理に必要
import base64  # base64エンコードされた画像の処理に必要
import asyncio
from typing_extensions import TypedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, TypedDict, Annotated
from mcp.types import ImageContent

from langchain_openai import ChatOpenAI  # OpenAIのインポートを追加
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage, AIMessage

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_mcp_adapters.client import MultiServerMCPClient

google_api_key = os.getenv("GOOGLE_APIKEY")
openai_api_key = os.getenv("OPENAI_API_KEY")  # OpenAIのAPIキーを環境変数から取得

# スクリーンショット保存用のディレクトリを作成
SCREENSHOTS_DIR = "screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    # system_message: Optional[SystemMessage]
    human_message: Optional[HumanMessage]


# スクリーンショットを処理する関数
def process_screenshot(message):
    """
    ツールのレスポンスからスクリーンショットを抽出して保存する
    """
    # ツールメッセージを処理
    if isinstance(message, ToolMessage):
        try:
            # ツールメッセージから画像データを抽出
            content = message.artifact
            if content is not None:
                base64_data = content[0].data
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(SCREENSHOTS_DIR, f"screenshot_{timestamp}.png")
                
                with open(screenshot_path, "wb") as img_file:
                    img_file.write(base64.b64decode(base64_data))
                print(f"ツールからスクリーンショットを保存しました: {screenshot_path}")
                return True
        except Exception as e:
            print(f"ツールレスポンスからのスクリーンショット保存に失敗しました: {e}")
    return False

def process_message(messages):
    
    return messages


def create_graph(state: GraphState, tools, model_chain):
    def should_continue(state):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state):
        messages = state["messages"]
        # 直前はtoolsのメッセージであるため、最後のメッセージを取得し、画像を保存する。
        last_message = messages[-1]
        bool = process_screenshot(last_message)
        if bool:
            # スクリーンショットを保存した場合、メッセージを追加する
            last_message.artifact = []
            messages[-1] = last_message
            state["messages"] = messages

        # model_chain.invokeの実行時間を計測
        messages_limited = process_message(messages)
        start_time = time.time()
        response = model_chain.invoke(messages_limited)
        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"model_chain.invokeの実行時間: {execution_time:.2f}秒")
        print(f"last_message: {last_message}")
        print(f"response: {response}")

        return {"messages": [response]}


    tool_node = ToolNode(tools)
    
    workflow = StateGraph(state)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

async def main(graph_config = {"configurable": {"thread_id": "12345"}}, query = None):
    # モデル設定の読み込み
    with open("config/mcp_config.json", "r") as f:
        mcp_config = json.load(f)
    
    model = ChatOpenAI(
        model="gpt-4.1-mini",
        openai_api_key=openai_api_key,
        temperature=0.1,
    )
    print("OpenAIモデルを使用します")

    # messageを作成する
    message = [
        SystemMessage(content= """
あなたは役にたつAIアシスタントです。日本語で回答し、考えた過程を結論より前に出力してください。
あなたは、「PlayWright」というブラウザを操作するtoolを利用することができます。適切に利用してユーザからの質問に回答してください。
ツールを利用する場合は、必ずツールから得られた情報のみを利用して回答してください。

まず、ユーザの質問からツールをどういう意図で何回利用しないといけないのかを判断し、必要なら複数回toolを利用して情報収集をしたのち、すべての情報が取得できたら、その情報を元に返答してください。
また、会話履歴は一部が省略されている場合があります。
ブラウザ操作後は１回の操作後に必ずbrowser_take_screenshot toolを使用してスクリーンショットを取得してください。

また、ブラウザの操作4回ごとに今までの操作をまとめ、回答メッセージにSUMMARYの文字を入れてください。
"""),
        MessagesPlaceholder("messages"),
    ]

    # messageからプロンプトを作成
    prompt = ChatPromptTemplate.from_messages(message)

    async with MultiServerMCPClient(mcp_config["mcpServers"]) as mcp_client:
        tools = mcp_client.get_tools()

        model_with_tools = prompt | model.bind_tools(tools)
        if query is None:
            # ユーザからの入力を取得
            query = input("入力してください:exitで終了: ")

        if query.lower() in ["exit", "quit"]:
            print("終了します。")
            return

        input_query = HumanMessage(
                [
                    {
                        "type": "text",
                        "text": f"{query}"
                    },
                ]
            )
        
        initial_state = {
            "messages": [input_query],
            # "system_message": query,
            "human_message": input_query
        }

        graph = create_graph(
            GraphState,
            tools,
            model_with_tools
        )

        response = await graph.ainvoke(initial_state, graph_config)

        # デバック用
        # print("response: ", response)

        # 最終的な回答
        print("=================================")
        print(response["messages"][-1].content)


if __name__ == "__main__":
    
    # イベントループを明示的に取得
    # Windows環境での非同期処理のためにProactorEventLoopを使用
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("プログラムが中断されました")
    finally:
        # 保留中のタスクをキャンセル
        pending_tasks = asyncio.all_tasks(loop)
        for task in pending_tasks:
            task.cancel()
            
        # タスクがキャンセルされるのを待つ
        if pending_tasks:
            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            
        # イベントループを閉じる
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

