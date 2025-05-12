# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import operator, os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage, AIMessage, RemoveMessage
from typing import Any, Callable, Dict, Iterable, List, Optional
from langgraph.graph import StateGraph, START, END, MessagesState, add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
import asyncio
from langgraph.checkpoint.memory import MemorySaver

# 共通化したファイルのインポート
from services.config.settings import setup_environment, SCREENSHOTS_DIR
from services.utils.screenshot_utils import process_screenshot
from services.utils.graph_types import GraphState, TestPlan, TEST_AGENT_TEMPLATE, PLANNING_SYSTEM_PROMPT
from langchain.prompts import ChatPromptTemplate

# 環境変数の設定
setup_environment()

model = ChatOpenAI(model="gpt-4.1-mini")

###########################################################################
# サブグラフノードの定義
###########################################################################
def create_testagent(state: GraphState, model, tools) -> GraphState:
    """エージェントノードを作成"""
    if state["subplan_index"] > 0:
        return {
            "completed_plans": True,
            "subplan_index": 0,
        }
    
    plan_prompt = ChatPromptTemplate.from_template(TEST_AGENT_TEMPLATE)
    model_with_tools = model.bind_tools(tools)
    plan = state["plans"]
    current_plan_index = state["current_plan_index"]
    current_plan = plan[current_plan_index]
    # print(f"メッセージ件数: {len(state['messages'])}")
    last_message = state["messages"][-1] if state["messages"] else None
    # 直前のツールの出力からデータを取得する。
    last_content = ""
    if last_message is not None:
        last_content = last_message.content
        # print(f"直前のツールの出力: {last_message}")

    testagent_chain = plan_prompt | model_with_tools
    response = testagent_chain.invoke({
        "current_plan": current_plan,
        "last_content": last_content,
    })
    
    return {
        "messages": [response],
        "completed_plans": False,
        "subplan_index": state["subplan_index"] + 1,
    }

def should_complete_subplan(state: GraphState) -> bool:
    """サブプランが完了しているかを判断"""
    # フラグがTrueの場合、サブプランは完了している
    return state["completed_plans"]

def end_subgraph(state: GraphState) -> GraphState:
    """サブグラフの終了ノード"""
    # ここでは何もしない
    last_message = state["messages"][-1] if state["messages"] else None
    # 直前のツールの出力からデータを取得する。
    last_content = ""
    if last_message is not None:
        last_content = last_message.content
        process_screenshot(last_message)
    return {
        "current_plan_index": state["current_plan_index"] + 1,
        "completed_plans": False,
        "subplan_index": 0,
    }

def build_subgraph(model, tools, memory) -> StateGraph:
    """サブグラフを構築するノード"""
    build_subgraph = StateGraph(GraphState)
    tool_node = ToolNode(tools)

    build_subgraph.add_node("testagent", lambda state: create_testagent(state, model, tools))
    build_subgraph.add_node("end_node", end_subgraph)
    build_subgraph.add_node("tool_node", tool_node)
    
    build_subgraph.add_edge(START, "testagent")
    build_subgraph.add_conditional_edges(
        "testagent", should_complete_subplan,
        path_map={
            True: "end_node",
            False: "tool_node"
        }
    )
    build_subgraph.add_edge("end_node", END)
    build_subgraph.add_edge("tool_node", "testagent")

    return build_subgraph.compile(name="subgraph", checkpointer=memory)

###########################################################################
# 親グラフノードの定義
###########################################################################

def planning(state: GraphState, model, tools) -> GraphState:
    """クエリから具体的なテスト計画を作成するノード"""
    query = state["query"]
    
    # with_structured_outputを使用して配列形式の出力を強制する
    structured_model = model.bind_tools(tools)
    structured_model = structured_model.with_structured_output(TestPlan)
    
    messages = [
        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
        HumanMessage(content=f"以下のテスト指示に対する具体的な実行手順を作成してください。各手順は1行で簡潔に記述してください。\n\n{query}")
    ]
    
    # 構造化された応答を直接取得
    response: TestPlan = structured_model.invoke(messages)
    
    # 計画を取得
    plans = response.steps
    print(f"Generated plans:")
    for i, plan in enumerate(plans):
        print(f"Step {i + 1}: {plan}")
    
    # 状態を更新
    return {
        "plans": plans,
        "current_plan_index": 0,
        "completed": False,
        "subplan_index": 0,
    }

def should_continue(state: GraphState) -> bool:
    """エージェントが続行するかどうかを判断する条件"""
    plans = state["plans"]
    current_plan_index = state["current_plan_index"]
    # すべての計画が完了した場合、Flaseを返す
    if current_plan_index is None or current_plan_index >= len(plans):
        return False
    # まだ計画が残っている場合、Trueを返す
    return True

def create_graph(model, tools):
    """メインのグラフを作成"""
    
    workflow = StateGraph(GraphState)
    
    memory = MemorySaver()
    # ノードの追加
    workflow.add_node("planning", lambda state: planning(state, model, tools))
    workflow.add_node("agent", build_subgraph(model, tools, memory=memory))
    
    # エッジの追加
    workflow.add_edge(START, "planning")
    workflow.add_edge("planning", "agent")
    workflow.add_conditional_edges(
        "agent", should_continue,
        path_map={
            True: "agent",
            False: END
        }
    )
    
    # グラフのコンパイル
    app = workflow.compile(checkpointer=memory, name="test_agent")
    
    return app
    
async def main(user_input=None, graph_config=None):
    async with MultiServerMCPClient(
        {
            "playwright": {
                # make sure you start your playwright server on port 8931
                # 以下コマンドを実行してPlaywrightサーバーを起動してください。
                # npx @playwright/mcp@latest --port 8931
                "url": "http://localhost:8931/sse",
                "transport": "sse",
            },
        }
    ) as client:
        if graph_config is None:
            # デフォルトのグラフ設定を使用する場合は、以下のように指定します。
            graph_config = {"configurable": {"thread_id": "12345"}, "recursion_limit": 50}
        tools = client.get_tools()
        agent = create_graph(model, tools)
        print(agent.get_graph().draw_mermaid())
        if user_input is None:
            user_input = """
1.http://localhost:8080/testにアクセスして、スクリーンショットを取得してください。
2.受注番号に1234を入力して、
受注年月日に2023-10-01を入力してください。
発送日に2023-10-02を入力してください。
受注金額に2020円を入力してください。
3.ここまで行ったらスクリーンショットを取得してください。
4.登録ボタンを押下して、スクリーンショットを取得してください。
5.戻るボタンを押下して、スクリーンショットを取得してください。
6.受注番号に1235を入力して、
受注年月日に2023-10-02を入力してください。
発送日に2023-10-03を入力してください。
受注金額に10250円を入力してください。
7.ここまで行ったらスクリーンショットを取得してください。
8.登録ボタンを押下して、スクリーンショットを取得してください。
9.戻るボタンを押下して、スクリーンショットを取得してください。
            """
        
        # 初期状態
        initial_state = {
            "query": user_input,
            "plans": None,
            "current_plan_index": None,
            "completed": False,
            "messages": [],
            "completed_plans": False,
        }
        
        response = await agent.ainvoke(initial_state, graph_config)
        return response

if __name__ == "__main__":
    asyncio.run(main())
