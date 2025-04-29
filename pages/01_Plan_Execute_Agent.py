import streamlit as st
import os
from plan_execute_agent import PlanExecuteAgent, AgentState, build_agent_graph
from typing import Dict, Any, List

# ページタイトルと説明
st.set_page_config(page_title="Plan and Executeエージェント", page_icon="🧠", layout="wide")
st.title("Plan and Execute型 設計書調査エージェント")
st.markdown("""
このページでは、Plan and Execute型のAIエージェントを使用して、設計書に関する質問に回答します。
エージェントは質問を複数のステップに分解し、段階的に情報を収集・分析して回答を生成します。
複数の検索クエリを自動生成して幅広い情報収集を行います。

できること:
- 計画を立てて、計画をもとに検索、分析、整理すること
- インデックス登録を行った設計書を確認すること
できないこと（今後の拡張を検討）:
- 設計書の一覧を確認すること
- Web上の情報を検索すること
""")

# セッション状態の初期化
if "plan_execute_messages" not in st.session_state:
    st.session_state.plan_execute_messages = []

if "plan_execute_steps" not in st.session_state:
    st.session_state.plan_execute_steps = []

if "plan_execute_results" not in st.session_state:
    st.session_state.plan_execute_results = []

# エージェントの初期化
@st.cache_resource
def get_agent():
    return PlanExecuteAgent()

# サイドバーのオプション
with st.sidebar:
    st.header("Plan and Execute型エージェント")
    st.divider()
    st.subheader("使用可能な質問例:")
    st.markdown("""
    - 受注データ取込バッチと受注確定バッチの違いを教えてください
    - 発送ラベル生成バッチと発送データ作成バッチの関係について
    - 在庫管理画面の主な機能を説明してください
    - 在庫自動発注バッチの処理内容を要約してください
    """)

    # モデル選択
    model = st.selectbox(
        "使用するモデル",
        ["gpt-4.1-nano"], 
        index=0
    )
    
    # 温度設定
    temperature = st.slider(
        "応答の多様性 (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0に近いほど決定的な回答、1に近いほど創造的な回答になります"
    )
    
# タブの作成
tab1, tab2 = st.tabs(["チャット", "実行計画"])

with tab1:
    # チャット履歴の表示
    for message in st.session_state.plan_execute_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力
    user_input = st.chat_input("設計書に関する質問を入力してください...")

    if user_input:
        # ユーザーの質問をチャット履歴に追加
        st.session_state.plan_execute_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 処理中の表示
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("計画を立案・実行中です...")

            processing_results = {
                    "user_query": user_input,
                    "final_response": None,
                }
            
            try:
                # エージェントの初期化とモデル設定
                agent = get_agent()
                agent.model_name = model
                agent.temperature = temperature
                
                # エージェントの実行
                with st.status("処理中...", expanded=False) as status:
                    status.update(label="計画を立案中...", state="running")
                    # 初期状態の作成
                    initial_state = AgentState(query=user_input, last_substep="", last_plan_description="")
                    agent_graph = build_agent_graph()
                    endflg = False
                    
                    # ステータス表示用のプレースホルダを作成
                    status_placeholder = st.empty()
                    status_messages = []
                    
                    # グラフの実行
                    for chunk in agent_graph.stream(initial_state, stream_mode="values", subgraphs=True):
                        
                        node_name = chunk[1]['last_substep']
                        last_plan_description = chunk[1]['last_plan_description']
                        
                        # ステータスメッセージを追加
                        if node_name == "search":
                            status.update(label="必要な情報を収集中...", state="running")
                            status_messages.append("> 必要な情報を収集中...")
                        elif node_name == "analyze":
                            status.update(label="情報の分析中...", state="running")
                            status_messages.append("> 情報の分析中...")
                        elif node_name == "synthesize":
                            status.update(label="情報の整理中...", state="running")
                            status_messages.append("> 情報の整理中...")
                        elif node_name == "unknown":
                            status.update(label="エラーを検知しました。", state="running")
                            status_messages.append("> エラーを検知しました。")
                        elif node_name == "plan":
                            status.update(label="計画を立案中...", state="running")
                            if last_plan_description != "":
                                status_messages.append("> 計画を立案完了")
                                status_messages.append(f"> 計画: {last_plan_description}")
                        elif node_name == "revise":
                            status.update(label="再計画中...", state="running")
                            status_messages.append("> 再計画中...")
                        elif node_name == "assessment":
                            status.update(label="収集した情報の評価中...", state="running")
                            status_messages.append("> 収集した情報を評価中...")
                        elif node_name == "generate_answer":
                            status.update(label="最終回答の生成中...", state="running")
                            status_messages.append("> 最終回答の生成中...")
                            endflg = True
                        elif node_name == "":
                            status.update(label="呼び出し中...", state="running")
                        
                        # HTMLを使って行間を制御し、蓄積されたメッセージを表示
                        html_content = "<div style='line-height: 1.2; margin-bottom: 0.5rem;'>"
                        html_content += "<br>".join(status_messages)
                        html_content += "</div>"
                        status_placeholder.markdown(html_content, unsafe_allow_html=True)
                        
                        if endflg and chunk[1]['final_answer'] is not None:
                            processing_results["final_response"] = chunk[1]['final_answer']
                            status.update(label="処理完了", state="complete")
                        
                    message_placeholder.markdown(processing_results["final_response"])
                    
            
            except Exception as e:
                error_message = f"エラーが発生しました: {str(e)}"
                message_placeholder.markdown(error_message)
                st.error(error_message)
                import traceback
                st.error(traceback.format_exc())

with tab2:
    st.header("Plan and Executeアプローチ")
    
    # 最新の計画を表示
    if st.session_state.plan_execute_steps:
        st.write("### 実行計画")
        
        # 各ステップの詳細な説明
        st.write("### 計画ステップの詳細")
        for step in st.session_state.plan_execute_steps:
            st.write(f"**ステップ {step['step_number']}**: {step['description']}")
            st.write(f"アクションタイプ: `{step['action_type']}`")
            st.divider()
        
        # 実行結果が存在する場合は表示
        if st.session_state.plan_execute_results:
            st.write("### 実行結果の詳細")
            for result in st.session_state.plan_execute_results:
                success_emoji = "✅" if result["success"] else "❌"
                step = result["step"]
                st.markdown(f"{success_emoji} **ステップ {step['step_number']}**: {step['description']} ({step['action_type']})")
                
                # 検索ステップの場合、生成された検索クエリを表示
                if step['action_type'] == 'search' and 'search_queries' in result:
                    st.markdown("**生成された検索クエリ:**")
                    for q_idx, query in enumerate(result['search_queries']):
                        st.markdown(f"- クエリ {q_idx+1}: `{query}`")
    else:
        st.info("質問を入力すると、ここに実行計画と結果が表示されます。")
