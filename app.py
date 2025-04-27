import streamlit as st
import os
from agent import gragh_build, query_analyzer_agent, search_agent, information_evaluator_agent, information_completer_agent, response_generator_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

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
    st.header("オプション")
    show_raw_response = st.checkbox("詳細な処理結果を表示", value=False)
    st.divider()
    st.subheader("使用可能な質問例:")
    st.markdown("""
    - 受注テーブルを利用している箇所を洗い出してください
    - 在庫管理に関する画面の機能を教えてください
    - 発送ラベル生成バッチの処理内容を説明してください
    - 受注確定バッチと発送バッチの連携について教えてください
    """)

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
user_input = st.chat_input("質問を入力してください...")

if user_input:
    # ユーザーの質問をチャット履歴に追加
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 処理中の表示
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("処理中です...")
        
        # 状態を保存するための準備
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        try:
            # 各エージェントの処理結果を保存するための辞書
            processing_results = {
                "user_query": user_input,
                "analyzed_questions": None,
                "search_results": None,
                "evaluation_results": None,
                "final_response": None
            }
            
            # 詳細表示モードのための処理
            if show_raw_response:
                # 手動でエージェントをステップバイステップで実行
                with st.status("質問の分析中...", expanded=False) as status:
                    # 質問分析
                    state = {"messages": [HumanMessage(content=user_input)]}
                    analysis_result = query_analyzer_agent(state)
                    processing_results["analyzed_questions"] = analysis_result["questions"]
                    status.update(label="質問分析完了", state="complete")
                
                with st.status("関連情報の検索中...", expanded=False) as status:
                    # 検索
                    state.update(analysis_result)
                    search_result = search_agent(state)
                    processing_results["search_results"] = search_result["relevant_documents"]
                    status.update(label="検索完了", state="complete")
                
                with st.status("情報の評価中...", expanded=False) as status:
                    # 情報評価
                    state.update(search_result)
                    evaluation_result = information_evaluator_agent(state)
                    processing_results["evaluation_results"] = evaluation_result
                    status.update(label="評価完了", state="complete")
                
                with st.status("情報の補完中...", expanded=False) as status:
                    # 情報補完（必要な場合）
                    if evaluation_result.get("has_information_gap", False):
                        state.update(evaluation_result)
                        completion_result = information_completer_agent(state)
                        state.update(completion_result)
                        status.update(label="情報補完完了", state="complete")
                    else:
                        state.update(evaluation_result)
                        status.update(label="補完不要", state="complete")
                
                with st.status("回答の生成中...", expanded=False) as status:
                    # 回答生成
                    response_result = response_generator_agent(state)
                    final_response = response_result.get("final_response", "回答を生成できませんでした。")
                    processing_results["final_response"] = final_response
                    status.update(label="回答生成完了", state="complete")
                
                # 詳細情報の表示
                st.subheader("処理詳細")
                
                # 分析された質問のカテゴリーと検索クエリの表示
                st.write("【分析された質問】")
                for i, q in enumerate(processing_results["analyzed_questions"]):
                    st.write(f"質問 {i+1}: カテゴリ `{q.question_category}`, 検索クエリ: `{q.search_query}`")
                
                # 検索結果の表示
                st.write("【関連文書】")
                for i, doc in enumerate(processing_results["search_results"][:3]):  # 最初の3件のみ表示
                    st.write(f"文書 {i+1}: {doc['metadata'].get('source', '不明')}")
                    with st.expander(f"内容を表示"):
                        st.write(doc['content'])
                        st.write(f"スコア: {doc['score']}")
                
                # 最終回答
                st.write("【最終回答】")
                message_placeholder.markdown(final_response)
            else:
                # グラフを使った実行（通常モード）
                events = graph.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config,
                    stream_mode="values",
                )
                
                # 最後のメッセージを取得して表示
                final_response = None
                for event in events:
                    if event.get("messages") and len(event["messages"]) > 0:
                        final_message = event["messages"][-1]
                        if hasattr(final_message, "content"):
                            final_response = final_message.content
                
                if final_response:
                    message_placeholder.markdown(final_response)
                else:
                    message_placeholder.markdown("回答を生成できませんでした。")
            
            # チャット履歴に回答を追加
            st.session_state.messages.append({"role": "assistant", "content": final_response if final_response else "回答を生成できませんでした。"})
            
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            message_placeholder.markdown(error_message)
            st.error(error_message)
            import traceback
            st.error(traceback.format_exc())

# フッター
st.divider()
st.markdown("© 2025 RAGエージェントアプリ")