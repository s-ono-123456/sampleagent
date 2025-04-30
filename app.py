import streamlit as st
import os
from services.agent import gragh_build
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from streamlit_mermaid import st_mermaid

# ページタイトルと説明
st.set_page_config(page_title="RAGエージェントアプリ", page_icon="🤖", layout="wide")
st.title("RAGエージェントアプリ")
st.markdown("""
このアプリケーションは、質問に対して関連する設計書を検索し、回答を生成します。
バッチ設計や画面設計に関する質問に回答します。
""")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.thread_id = "1"  # 固定のスレッドID

# グラフの構築
@st.cache_resource
def get_graph():
    return gragh_build()

graph = get_graph()


# サイドバーのオプション
with st.sidebar:
    st.subheader("使用可能な質問例:")
    st.markdown("""
    - 受注テーブルを利用している箇所を洗い出してください
    - 在庫管理に関する画面の機能を教えてください
    - 発送ラベル生成バッチの処理内容を説明してください
    - 受注確定バッチと発送バッチの連携について教えてください
    """)
# ユーザー入力
user_input = st.text_input("質問を入力してください", key="user_input", placeholder="質問を入力してください...")


if user_input:
    # タブの作成
    tab1, tab2, tab3, tab4 = st.tabs(["回答", "フローチャート", "関連文書", "回答評価"])
    error_message = None
    with tab1:
        # 処理中の表示
        with st.status("処理中...", expanded=True) as status:
            try:
                # 状態を保存するための準備
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                # 各エージェントの処理結果を保存するための辞書
                # Graphを使用しつつ、詳細情報を収集
                processing_results = {
                    "user_query": user_input,
                    "analyzed_questions": [],
                    "search_results": [],
                    "evaluation_results": None,
                    "final_response": None,
                    "check_result": None,
                }
                endflg = False
                for chunk in graph.stream(
                    {"messages": [HumanMessage(content=user_input)],
                        "last_node": ""},
                    config,
                    stream_mode="values",
                ):
                    # イベントをストリーミングして処理状況を更新
                    # ノード名を取得して状態を更新
                    node_name = chunk['last_node']
                    if node_name == "query_analyzer":
                        status.update(label="質問の分析中...", state="running")
                    elif node_name == "search":
                        status.update(label="関連情報の検索中...", state="running")
                    elif node_name == "information_evaluator":
                        status.update(label="情報の評価中...", state="running")
                    elif node_name == "information_completer":
                        status.update(label="情報の補完中...", state="running")
                    elif node_name == "response_generator":
                        status.update(label="回答の生成中...", state="running")
                    elif node_name == "response_evaluate":
                        status.update(label="回答の評価中...", state="running")
                        endflg = True
                    elif node_name == "":
                        status.update(label="呼び出し中...", state="running")
                    
                    # 最終的な回答を取得
                    if endflg and chunk.get("messages") and len(chunk["messages"]) > 0:
                        useful_documents = chunk['useful_documents']
                        processing_results["final_response"] = chunk['final_response']
                        processing_results["check_result"] = chunk['check_result']
                        status.update(label="処理完了", state="complete")
            
                # チャット履歴に回答を追加
                st.session_state.messages.append({"role": "assistant", "content": processing_results["final_response"] if processing_results["final_response"] else "回答を生成できませんでした。"})
            
            except Exception as e:
                error_message = f"エラーが発生しました: {str(e)}"
                st.error(error_message)
                import traceback
                st.error(traceback.format_exc())
    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("処理中です...")
            if error_message:
                message_placeholder.markdown(error_message)
            else:
                # 最終回答
                message_placeholder.markdown(processing_results["final_response"])

    with tab2:
        st.header("エージェント処理フロー")
        st.write("RAGエージェントの処理フローを示すダイアグラムです。各ノードは異なるエージェントの役割を表しています。")
        
        # フローチャートを表示
        try:
            # LangGraphのメモリーダイドを取得して表示
            mermaid_code = gragh_build().get_graph().draw_mermaid()
            st_mermaid(mermaid_code)
            st.code(mermaid_code, language="mermaid")

        except Exception as e:
            st.error(f"フローチャートの表示中にエラーが発生しました: {str(e)}")
            st.info("グラフ構造を表示するには、まずクエリを実行してください。")

    with tab3:
        # 検索結果の表示
        st.write("【関連文書】")
        for i, doc in enumerate(useful_documents):
            with st.expander(f"文書 {i+1}: {doc['metadata'].get('source', '不明')}"):
                st.write(doc['content'])

    with tab4:
        st.write("【回答評価】")
        st.write(processing_results["check_result"])        