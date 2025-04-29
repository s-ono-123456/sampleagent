import streamlit as st
import os
from plan_execute_agent import PlanExecuteAgent
from typing import Dict, Any, List

# ページタイトルと説明
st.set_page_config(page_title="Plan and Executeエージェント", page_icon="🧠", layout="wide")
st.title("Plan and Execute型 設計書調査エージェント")
st.markdown("""
このページでは、Plan and Execute型のAIエージェントを使用して、設計書に関する質問に回答します。
エージェントは質問を複数のステップに分解し、段階的に情報を収集・分析して回答を生成します。
複数の検索クエリを自動生成して幅広い情報収集を行います。
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
    
    # デバッグモード
    debug_mode = st.checkbox("デバッグモード", value=False, help="チェックすると計画と実行の詳細ステップが表示されます")

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
            
            try:
                # エージェントの初期化とモデル設定
                agent = get_agent()
                agent.model_name = model
                agent.temperature = temperature
                
                # エージェントの実行
                with st.status("処理中...", expanded=True) as status:
                    status.update(label="計画を立案中...", state="running")
                    result = agent.run(user_input)
                    
                    if result["status"] == "success":
                        # 計画と実行結果の保存
                        st.session_state.plan_execute_steps = result["plan"]
                        st.session_state.plan_execute_results = result["execution_results"]
                        
                        # 計画の表示
                        status.update(label="計画を実行中...", state="running")
                        
                        # 各実行ステップを表示
                        for i, exec_result in enumerate(result["execution_results"]):
                            step = exec_result["step"]
                            status.update(label=f"ステップ {step['step_number']} 実行中: {step['description']}", state="running")
                        
                        status.update(label="処理完了", state="complete")
                        
                        # 最終回答を表示
                        message_placeholder.markdown(result["answer"])
                        
                        # デバッグモードの場合、実行ステップの詳細を表示
                        if debug_mode:
                            st.subheader("処理詳細")
                            st.write("【実行された計画】")
                            for step in result["plan"]:
                                st.write(f"ステップ {step['step_number']}: {step['description']} ({step['action_type']})")
                            
                            # ステップごとの実行結果
                            for i, exec_result in enumerate(result["execution_results"]):
                                step = exec_result["step"]
                                st.markdown(f"**ステップ {step['step_number']}: {step['description']} ({step['action_type']})**")
                                
                                # 検索ステップの場合、生成された検索クエリを表示
                                if step['action_type'] == 'search' and 'search_queries' in exec_result:
                                    st.markdown("**生成された検索クエリ:**")
                                    for q_idx, query in enumerate(exec_result['search_queries']):
                                        st.markdown(f"- クエリ {q_idx+1}: `{query}`")
                                    
                                st.markdown("**実行結果:**")
                                st.markdown(exec_result['result'])
                                st.divider()
                        
                        # チャット履歴に回答を追加
                        st.session_state.plan_execute_messages.append({"role": "assistant", "content": result["answer"]})
                    else:
                        error_message = f"エラーが発生しました: {result['error']}"
                        message_placeholder.markdown(error_message)
                        st.error(error_message)
            
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
