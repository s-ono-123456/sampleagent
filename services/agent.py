"""
RAGベースのエージェントモジュール
設計書から情報を抽出し、ユーザーの質問に回答するエージェントです
"""
import json
import os
import datetime
import base64

# 共通ユーティリティのインポート
from services.config.settings import setup_environment
from services.utils.model_utils import init_chat_model, MODEL_NAME
from services.utils.state_utils import State, handle_screenshot
from services.utils.agent_utils import build_agent_graph

# 環境変数の設定
setup_environment()

# グラフのビルド
def gragh_build():
    """
    エージェントの処理フローを定義するグラフを構築する
    
    Returns:
        graph: コンパイルされたStateGraph
    """
    return build_agent_graph()


if __name__ == "__main__":
    # グラフのビルド
    graph = gragh_build()

    # グラフ実行
    user_input = "受注テーブルを利用している箇所を洗い出してください。"
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
