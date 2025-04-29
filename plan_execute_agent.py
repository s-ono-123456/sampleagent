import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple, TypedDict

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from vector_store_loader import load_vector_stores


# 環境変数の設定
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "Plan and Execute agent"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 定数の定義
MAX_REVISION_COUNT = 2  # 再計画の最大回数
MODEL_NAME = "gpt-4.1-nano"  # 使用するモデル名

###########################################################################
# プロンプトテンプレートの定義
###########################################################################

# 計画作成用プロンプト
PLAN_PROMPT_TEMPLATE = """\
あなたは設計書に関する調査・質問対応を行うAIアシスタントです。
ユーザーからの以下の依頼事項に答えるための調査計画を作成してください。

依頼事項: {query}

与えられた依頼事項に対して、単純なステップバイステップな計画を作成してください。
この計画は正しく実行すれば正しい答えを導き出せるような個々のタスクで構成されている必要があります。
余計な手順は追加しないでください。
最終ステップの結果が最終的な答えになるようにしてください。
各ステップには必要な情報をすべて含めて、スキップしないようにしてください。

また、各ステップの中でどのような作業を行うべきかを詳細に説明してください。
できるだけ簡潔な内容として、複数の項目を含まないようにしてください。
"""

# 計画作成用プロンプト
SUBPLAN_PROMPT_TEMPLATE = """\
あなたは設計書に関する調査・質問対応を行うAIアシスタントです。
ユーザーからの以下の依頼事項に答えるため、以下の計画を作成しました。
これから以下のアクションを行ってもらいます。

依頼事項: {query}
計画: {main_plan}
実施アクション: {action}

与えられた依頼事項に対して、単純なステップバイステップな計画を作成してください。
この計画は正しく実行すれば正しい答えを導き出せるような個々のタスクで構成されている必要があります。
余計な手順は追加しないでください。
最終ステップの結果が最終的な答えになるようにしてください。
各ステップには必要な情報をすべて含めて、スキップしないようにしてください。

計画の中では以下のステップを利用することができます。
1回のステップの中に複数回のアクションを含むこともできます。
- 必要な情報を特定するための検索ステップ (step_type: search)
- 収集した情報を分析するためのステップ (step_type: analyze)
- 最終的な回答を生成するための統合ステップ (step_type: synthesize)

また、各ステップの中でどのような作業を行うべきかを詳細に説明してください。
できるだけ簡潔な内容として、複数の項目を含まないようにしてください。
"""

# 検索クエリ生成用プロンプト
SEARCH_PROMPT_TEMPLATE = """\
以下のステップを実行するために適切な検索クエリを複数作成してください:

ステップ: {step_description}
元の質問: {query}

ベクトルデータベースを検索するための検索クエリを3つ作成してください。
各クエリは異なる視点や表現を用いて、情報を広く収集できるようにしてください。
"""

# 分析用プロンプト
ANALYZE_PROMPT_TEMPLATE = """\
以下の情報を分析し、ステップに関する洞察を提供してください:

ステップ: {step_description}
元の質問: {query}

これまでに収集した情報:
{collected_info}

分析結果:
"""

# 計画修正用プロンプト
REVISION_PROMPT_TEMPLATE = """\
次の計画を実行中に問題が発生しました。より効果的な計画を提案してください。

元の質問: {query}

元の計画: 
{original_plan}

実行結果:
{execution_results}

修正された計画を作成してください。各ステップには番号、説明、アクションタイプ（search/analyze/synthesize）を含めてください。
"""

# 情報充足度評価用プロンプト
ASSESSMENT_PROMPT_TEMPLATE = """
あなたは質問に回答するために得られた洞察の充足度を評価するAIアシスタントです。
ユーザーの質問に対して、以下の洞察が十分であるかどうかを評価してください。

質問: {query}

得られた洞察:
{all_insights}

以下の評価基準に基づいて得られた洞察の充足度を評価してください:
1. **網羅性**: 質問に関連するすべての重要な側面がカバーされているか
2. **一貫性**: 矛盾する情報がないか
3. **適切性**: 洞察が質問に直接関連しているか
4. **詳細度**: 質問に答えるために十分な詳細情報が含まれているか
5. **最新性**: 情報が最新かどうか（もし判断できる場合）

充足度スコアは1〜5の整数で評価してください（1が最も不十分、5が最も十分）。
不足している情報がある場合は具体的に列挙してください。
再調査が必要かどうかを判断し（スコアが4未満の場合は必要）、その理由を説明してください。
"""

# 回答生成用プロンプト
ANSWER_PROMPT_TEMPLATE = """
収集したすべての情報に基づいて、ユーザーの質問に対する最終的な回答を作成してください。

質問: {query}

収集した情報:
{all_results}

回答は明確で簡潔、かつ情報に富んだものにしてください。
設計書の内容に基づいて事実を述べ、根拠となる情報を含めてください。

回答:
"""

###########################################################################
# Pydanticモデルの定義
###########################################################################

# エージェントの状態を定義するPydanticモデル
class AgentState(BaseModel):
    """LangGraphで使用するエージェントの状態"""
    query: str = Field(description="ユーザーからの質問")
    main_plan: List[Dict[str, Any]] = Field(default_factory=list, description="解決のためのメイン計画ステップ")
    sub_plan: List[Dict[str, Any]] = Field(default_factory=list, description="解決のためのサブ計画ステップ")
    current_step_index: int = Field(default=0, description="現在実行中のステップのインデックス")
    current_substep_index: int = Field(default=0, description="現在実行中のサブステップのインデックス")
    execution_results: List[Dict[str, Any]] = Field(default_factory=list, description="実行結果のリスト")
    need_plan_revision: bool = Field(default=False, description="計画修正が必要かどうか")
    information_sufficient: bool = Field(default=True, description="収集した情報が十分かどうか")
    final_answer: Optional[str] = Field(default=None, description="最終的な回答")
    next_step: str = Field(default=None, description="次に実行すべきステップ")
    revision_count: int = Field(default=0, description="計画修正の回数")

# 計画のステップを表すモデル
class MainPlanStep(BaseModel):
    step_number: int = Field(description="ステップの番号")
    description: str = Field(description="ステップの説明")
    step_action: str = Field(description="ステップの詳細な作業内容")

# 計画のステップを表すモデル
class SubPlanStep(BaseModel):
    step_number: int = Field(description="ステップの番号")
    description: str = Field(description="ステップの説明")
    step_type: str = Field(description="ステップのタイプ（search, analyze, synthesize）")
    step_action: str = Field(description="ステップの詳細な作業内容")

# 実行計画全体を表すモデル
class MainPlan(BaseModel):
    steps: List[MainPlanStep] = Field(description="計画のステップリスト")

# 実行計画全体を表すモデル
class SubPlan(BaseModel):
    steps: List[SubPlanStep] = Field(description="計画のステップリスト")

# 検索クエリを表すモデル
class SearchQuery(BaseModel):
    query: str = Field(description="検索に使用するクエリ")

# 複数の検索クエリを表すモデル
class MultiSearchQuery(BaseModel):
    queries: List[str] = Field(description="複数の検索クエリリスト")

# 情報充足度評価の結果を表すモデル
class InformationSufficiencyAssessment(BaseModel):
    sufficiency_score: int = Field(description="情報の充足度スコア (1-5)")
    missing_info: List[str] = Field(default_factory=list, description="不足している情報のリスト")
    need_more_research: bool = Field(description="再調査が必要かどうか")
    reason: str = Field(description="再調査が必要な理由や十分と判断した理由")

###########################################################################
# サブグラフノードの定義
###########################################################################

# サブ計画作成ノード
def create_plan(state: AgentState) -> AgentState:
    """ユーザーの質問から実行計画を作成するノード"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # プロンプトテンプレートを使用
    plan_prompt = ChatPromptTemplate.from_template(SUBPLAN_PROMPT_TEMPLATE)
    
    # with_structured_outputを使用してSubPlan形式での出力を強制
    structured_llm = llm.with_structured_output(SubPlan)

    # 現在のステップを取得
    current_step = state.main_plan[state.current_step_index]
    step_description = current_step.get("description", "")
    step_infomation = current_step.get("step_action", "")
    descriptions = []
    for step in state.main_plan:
        descriptions.append(step.get("description", ""))
    main_plan ="\n- ".join(descriptions)
    action =f"""
    概要: {step_description}
    詳細: {step_infomation}
    """
    
    # チェーンを実行して計画を生成
    plan_chain = plan_prompt | structured_llm
    plan_result = plan_chain.invoke({"query": state.query, "main_plan": main_plan, "action": action})
    
    # Planオブジェクトをリストに変換
    steps = [step.model_dump() for step in plan_result.steps]
    
    # 状態を更新して返す
    return {
        "sub_plan": steps
    }

# ベクトル検索関数
def search_documents(step_description: str, query: str) -> str:
    """ベクトルストアを使用して関連文書を検索"""
    # LLMの初期化
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # プロンプトテンプレートを使用
    search_prompt = ChatPromptTemplate.from_template(SEARCH_PROMPT_TEMPLATE)
    
    # with_structured_outputを使用してMultiSearchQuery形式での出力を強制
    structured_llm = llm.with_structured_output(MultiSearchQuery)
    
    # チェーンを実行して検索クエリを生成
    search_chain = search_prompt | structured_llm
    search_result = search_chain.invoke({
        "step_description": step_description, 
        "query": query
    })
    
    # 構造化された結果から検索クエリを取得
    search_queries = search_result.queries
    
    # ベクトルストアを読み込み
    try:
        vector_store = load_vector_stores()
    except Exception as e:
        return f"ベクトルストアの読み込みに失敗しました: {str(e)}"
    
    # 重複除去のためのセット
    all_docs_set = set()
    unique_docs = []
    
    # 各クエリで検索を実行し重複を除去
    for query_text in search_queries:
        try:
            # 検索の実行
            docs = vector_store.similarity_search(query_text, k=3)
            
            for doc in docs:
                # ドキュメントのソースと内容をキーとして使用して重複を検出
                doc_key = f"{doc.metadata.get('source', 'unknown')}::{doc.page_content[:100]}"
                
                if doc_key not in all_docs_set:
                    all_docs_set.add(doc_key)
                    unique_docs.append(doc)
        except Exception as e:
            print(f"クエリ '{query_text}' の検索中にエラーが発生しました: {str(e)}")
            continue
    
    # 関連ドキュメントが見つからなかった場合
    if not unique_docs:
        return "関連する情報が見つかりませんでした。"
    
    # 検索結果のフォーマット
    results = []
    for doc in unique_docs:
        source = doc.metadata.get("source", "不明")
        results.append(f"ドキュメント: {source}\n内容: {doc.page_content}")
    
    # 最終的な結果文字列
    result_text = "\n\n".join(results)
    
    # UI表示用に検索クエリを返す
    return result_text, search_queries

# サブグラフの各ノード（ステップタイプごとの処理関数）
def execute_search_step(state: AgentState) -> AgentState:
    """検索ステップの実行"""
    # 現在のステップを取得
    current_step = state.main_plan[state.current_substep_index]
    step_description = current_step.get("description", "")
    
    try:
        # 検索の実行（結果と検索クエリを取得）
        search_result = search_documents(step_description, state.query)
        
        # 検索結果とクエリを取得
        result, search_queries = search_result
        # 検索クエリを実行結果に含める
        execution_result = {
            "step": current_step,
            "result": result,
            "success": bool(result) and "情報が見つかりません" not in result,
            "search_queries": search_queries
        }
        
    except Exception as e:
        result = f"検索中にエラーが発生しました: {str(e)}"
        execution_result = {
            "step": current_step,
            "result": result,
            "success": False
        }
    
    # 実行結果を記録
    execution_results = state.execution_results.copy()
    execution_results.append(execution_result)
    
    # 次のステップへ、または計画修正が必要かどうかを判断
    return {
        "current_substep_index": state.current_substep_index + 1,
        "execution_results": execution_results,
        "need_plan_revision": not execution_result["success"]
    }

def execute_analyze_step(state: AgentState) -> AgentState:
    """分析ステップの実行"""
    # 現在のステップを取得
    current_step = state.main_plan[state.current_substep_index]
    step_description = current_step.get("description", "")
    
    # LLMの初期化
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # これまでの情報を分析
    collected_info = "\n\n".join([
        f"検索結果 {i+1}: {r.get('result', '')}" 
        for i, r in enumerate(state.execution_results)
        if r.get("step", {}).get("step_type", "") == "search"
    ])
    
    try:
        if not collected_info:
            result = "分析する情報がありません。検索ステップが失敗したか、実行されていません。"
            success = False
        else:
            # プロンプトテンプレートを使用
            analyze_prompt = ChatPromptTemplate.from_template(ANALYZE_PROMPT_TEMPLATE)
            
            analyze_chain = analyze_prompt | llm
            analyze_result = analyze_chain.invoke({
                "step_description": step_description,
                "query": state.query,
                "collected_info": collected_info
            })
            
            result = analyze_result.content if hasattr(analyze_result, 'content') else str(analyze_result)
            success = True
        
        execution_result = {
            "step": current_step,
            "result": result,
            "success": success
        }
            
    except Exception as e:
        result = f"分析中にエラーが発生しました: {str(e)}"
        execution_result = {
            "step": current_step,
            "result": result,
            "success": False
        }
    
    # 実行結果を記録
    execution_results = state.execution_results.copy()
    execution_results.append(execution_result)
    
    # 次のステップへ、または計画修正が必要かどうかを判断
    return {
        "current_substep_index": state.current_substep_index + 1,
        "execution_results": execution_results,
        "need_plan_revision": not execution_result["success"]
    }

def execute_synthesize_step(state: AgentState) -> AgentState:
    """統合ステップの実行"""
    # 現在のステップを取得
    current_step = state.main_plan[state.current_substep_index]
    
    # 統合ステップは特別な処理をせず、後で実行される
    execution_result = {
        "step": current_step,
        "result": "統合ステップは後で実行されます",
        "success": True
    }
    
    # 実行結果を記録
    execution_results = state.execution_results.copy()
    execution_results.append(execution_result)
    
    # 次のステップへ
    return {
        "current_substep_index": state.current_substep_index + 1,
        "execution_results": execution_results,
        "need_plan_revision": False
    }

def execute_unknown_step(state: AgentState) -> AgentState:
    """不明なステップタイプの実行（エラー処理）"""
    # 現在のステップを取得
    current_step = state.main_plan[state.current_substep_index]
    step_type = current_step.get("step_type", "")
    
    # 不明なアクションタイプ
    execution_result = {
        "step": current_step,
        "result": f"不明なアクションタイプ: {step_type}",
        "success": False
    }
    
    # 実行結果を記録
    execution_results = state.execution_results.copy()
    execution_results.append(execution_result)
    
    # 次のステップへ、修正が必要
    return {
        "current_substep_index": state.current_substep_index + 1,
        "execution_results": execution_results,
        "need_plan_revision": True
    }

# 実行するサブステップタイプを判断する関数
def determine_substep_type(state: AgentState) -> str:
    """現在のステップタイプを判断し、対応するノード名を返す"""
    # 全ステップが完了していたら終了
    if state.current_substep_index >= len(state.main_plan):
        state.current_step_index = state.current_step_index + 1
        return "completed"
    
    # 現在のステップタイプを取得
    current_step = state.main_plan[state.current_substep_index]
    step_type = current_step.get("step_type", "")
    
    # ステップタイプに基づいてノード名を返す
    if step_type == "search":
        return "execute_search"
    elif step_type == "analyze":
        return "execute_analyze"
    elif step_type == "synthesize":
        return "execute_synthesize"
    else:
        return "execute_unknown"

# サブグラフ構築関数
def build_execution_subgraph():
    """ステップ実行のためのサブグラフを構築"""
    # サブグラフの作成
    execution_graph = StateGraph(AgentState)
    
    # ノードの追加
    execution_graph.add_node("create_plan", create_plan)
    execution_graph.add_node("execute_search", execute_search_step)
    execution_graph.add_node("execute_analyze", execute_analyze_step)
    execution_graph.add_node("execute_synthesize", execute_synthesize_step)
    execution_graph.add_node("execute_unknown", execute_unknown_step)

    # 条件分岐の設定
    path_map = {
            "execute_search": "execute_search",
            "execute_analyze": "execute_analyze",
            "execute_synthesize": "execute_synthesize",
            "execute_unknown": "execute_unknown",
            "completed": END
        }
    # エントリーポイントからの条件分岐
    execution_graph.add_edge(START, "create_plan")
    execution_graph.add_conditional_edges("create_plan", determine_substep_type, path_map=path_map)
    execution_graph.add_conditional_edges("execute_search", determine_substep_type, path_map=path_map)
    execution_graph.add_conditional_edges("execute_analyze", determine_substep_type, path_map=path_map)
    execution_graph.add_conditional_edges("execute_synthesize", determine_substep_type, path_map=path_map)
    # 不明なタイプの場合はサブグラフを終わらせて再計画を促す必要がある。
    execution_graph.add_edge("execute_unknown", END)
    
    # サブグラフをコンパイルして返す
    return execution_graph.compile(name='SubAgentGraph')

###########################################################################
# メイングラフノードの定義
###########################################################################

# 計画作成ノード
def create_plan(state: AgentState) -> AgentState:
    """ユーザーの質問から実行計画を作成するノード"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # プロンプトテンプレートを使用
    plan_prompt = ChatPromptTemplate.from_template(PLAN_PROMPT_TEMPLATE)
    
    # with_structured_outputを使用してMainPlan形式での出力を強制
    structured_llm = llm.with_structured_output(MainPlan)
    
    # チェーンを実行して計画を生成
    plan_chain = plan_prompt | structured_llm
    plan_result = plan_chain.invoke({"query": state.query})
    
    # Planオブジェクトをリストに変換
    steps = [step.model_dump() for step in plan_result.steps]
    
    # 状態を更新して返す
    return {
        "main_plan": steps
    }
# タスク実行ノード (メインの実行ノード、サブグラフを呼び出す)
def execute_step(state: AgentState) -> AgentState:
    """計画の各ステップを実行するノード (サブグラフを使用)"""
    # サブグラフの構築
    execution_subgraph = build_execution_subgraph()
    mermaid_code = execution_subgraph.get_graph().draw_mermaid()
    print(f"サブエージェントのグラフ:\n{mermaid_code}")
    
    # サブグラフを実行
    result = execution_subgraph.invoke(state)
    
    # サブグラフの結果を返す
    return result

# 計画評価ノード
def evaluate_plan(state: AgentState) -> str:
    """計画の状態を評価し、次に実行すべきノードを決定"""
    if state.need_plan_revision:
        return "revise_plan"
    elif state.current_step_index < len(state.main_plan):
        return "execute_step"
    else:
        return "assess_information_sufficiency"

# 計画修正ノード
def revise_plan(state: AgentState) -> AgentState:
    """実行結果に基づいて計画を修正"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # 実行結果のフォーマット
    execution_results_text = "\n".join([\
        f"ステップ {r.get('step', {}).get('step_number', i+1)} " +\
        f"({r.get('step', {}).get('step_type', 'unknown')}): {r.get('result', '')}"\
        for i, r in enumerate(state.execution_results)\
    ])
    
    # 計画修正プロンプト
    revision_prompt = ChatPromptTemplate.from_template(REVISION_PROMPT_TEMPLATE)
    
    # with_structured_outputを使用してPlan形式での出力を強制
    structured_llm = llm.with_structured_output(MainPlan)
    
    try:
        # チェーンを実行して修正計画を生成
        revision_chain = revision_prompt | structured_llm
        revision_result = revision_chain.invoke({
            "query": state.query,
            "original_plan": json.dumps(state.main_plan, ensure_ascii=False),
            "execution_results": execution_results_text
        })
        
        # Planオブジェクトをリストに変換
        revised_steps = [step.model_dump() for step in revision_result.steps]
        
    except Exception as e:
        print(f"修正計画の生成中にエラーが発生しました: {str(e)}")
        # フォールバック計画
        revised_steps = [
            {"step_number": 1, "description": f"「{state.query}」に関する別の情報源を検索", "step_type": "search"},
            {"step_number": 2, "description": "新しく収集した情報を分析", "step_type": "analyze"},
            {"step_number": 3, "description": "最終的な回答を生成", "step_type": "synthesize"}
        ]
    
    # 状態を更新して返す
    return {
        "current_step_index": 0,
        "current_substep_index": 0,
        "main_plan": revised_steps,
        "need_plan_revision": False
    }

# 情報充足度評価ノード
def assess_information_sufficiency(state: AgentState) -> AgentState:
    """収集した情報の充足度を評価し、再計画の必要性を判断する"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # 分析ステップ（step_type: analyze）の結果のみを抽出
    analyze_results = []
    
    # 現在の計画のanalyzeステップの結果
    current_analyze_results = [
        f"ステップ {r.get('step', {}).get('step_number', i+1)} (洞察): {r.get('result', '')}"
        for i, r in enumerate(state.execution_results)
        if r.get('step', {}).get('step_type', '') == 'analyze'
    ]
    
    analyze_results.extend(current_analyze_results)
    
    # 以前の計画のanalyzeステップの結果も抽出
    # 現在の計画のanalyzeステップではない古い実行結果を取得
    previous_results_slice = state.execution_results[:-len(current_analyze_results)] if current_analyze_results else state.execution_results
    previous_analyze_results = [
        f"以前の計画からの洞察 {i+1}: {r.get('result', '')}"
        for i, r in enumerate(previous_results_slice)
        if r.get('step', {}).get('step_type', '') == 'analyze'
    ]
    
    if previous_analyze_results:
        analyze_results.extend(previous_analyze_results)
    
    # 分析結果がない場合は適切なメッセージを設定
    if not analyze_results:
        analyze_results = ["洞察が得られていません。検索結果からの分析が実行されていないか失敗しました。"]
    
    # 洞察結果のフォーマット
    all_insights = "\n\n".join(analyze_results)
    
    # 情報充足度評価プロンプト
    assessment_prompt = ChatPromptTemplate.from_template(ASSESSMENT_PROMPT_TEMPLATE)
    
    # with_structured_outputを使用してInformationSufficiencyAssessment形式での出力を強制
    structured_llm = llm.with_structured_output(InformationSufficiencyAssessment)
    
    # チェーンを実行して評価結果を生成
    assessment_chain = assessment_prompt | structured_llm
    assessment_result = assessment_chain.invoke({
        "query": state.query,
        "all_insights": all_insights
    })
    
    # 構造化された評価結果からデータを取得
    sufficiency_score = assessment_result.sufficiency_score
    missing_info = assessment_result.missing_info
    need_more_research = assessment_result.need_more_research
    reason = assessment_result.reason
    
    # 評価結果の詳細情報
    assessment_details = {
        "sufficiency_score": sufficiency_score,
        "missing_info": missing_info,
        "need_more_research": need_more_research,
        "reason": reason
    }
    
    # 再計画回数を更新
    revision_count = state.revision_count
    
    # 判断結果に基づいて次のステップを設定
    if need_more_research and revision_count < MAX_REVISION_COUNT:
        # 再計画が必要かつ最大回数未満の場合
        next_step = "revise_plan"
        information_sufficient = False
        revision_count += 1
        reason_message = reason
    elif need_more_research and revision_count >= MAX_REVISION_COUNT:
        # 再計画が必要だが最大回数に達した場合、強制的に回答生成へ
        next_step = "generate_answer"
        information_sufficient = True
        reason_message = f"再計画の最大回数({MAX_REVISION_COUNT}回)に達したため、現在の情報で回答を生成します。\n" + reason
    else:
        # 情報が十分な場合
        next_step = "generate_answer"
        information_sufficient = True
        reason_message = reason
    
    # 状態を更新して返す
    return {
        "next_step": next_step,
        "information_sufficient": information_sufficient,
        "revision_count": revision_count,
        "execution_results": state.execution_results + [{
            "step": {"step_number": len(state.execution_results) + 1, "description": "情報充足度評価", "step_type": "evaluate"},
            "result": f"充足度評価: {sufficiency_score}/5 - {'十分' if information_sufficient else '不十分'}\n理由: {reason_message}",
            "success": True,
            "assessment_details": assessment_details
        }]
    }

# 回答生成ノード
def generate_answer(state: AgentState) -> AgentState:
    """実行結果から最終回答を生成"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # 実行結果のフォーマット
    all_results = "\n\n".join([
        f"ステップ {r.get('step', {}).get('step_number', i+1)} " +
        f"({r.get('step', {}).get('step_type', 'unknown')}): {r.get('result', '')}"
        for i, r in enumerate(state.execution_results)
    ])
    
    # 回答生成プロンプト
    answer_prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)
    
    # 回答の生成
    answer_chain = answer_prompt | llm
    answer_result = answer_chain.invoke({
        "query": state.query,
        "all_results": all_results
    })
    
    # 回答テキストを抽出
    final_answer = answer_result.content if hasattr(answer_result, 'content') else str(answer_result)
    
    # 状態を更新して返す
    return {
        "final_answer": final_answer
    }

# グラフの構築
def build_agent_graph():
    """LangGraphを使用したエージェントのグラフを構築"""
    # 状態グラフの作成
    workflow = StateGraph(AgentState)
    
    # ノードの追加
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("execute_step", execute_step)
    workflow.add_node("revise_plan", revise_plan)
    workflow.add_node("assess_information_sufficiency", assess_information_sufficiency)
    workflow.add_node("generate_answer", generate_answer)
    
    # エッジの追加（ノード間の接続）
    workflow.add_edge("create_plan", "execute_step")
    
    # execute_stepノードから直接条件分岐
    workflow.add_conditional_edges(
        "execute_step", evaluate_plan,
        path_map = {
            "revise_plan": "revise_plan",
            "execute_step": "execute_step",
            "assess_information_sufficiency": "assess_information_sufficiency"
        }
    )
    
    # 情報充足度評価ノードからの条件分岐
    workflow.add_conditional_edges(
        "assess_information_sufficiency", 
        lambda state: state.next_step,
        path_map = {
            "revise_plan": "revise_plan",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_edge("revise_plan", "execute_step")
    workflow.add_edge("generate_answer", END)
    
    # 開始ノードの設定
    workflow.set_entry_point("create_plan")
    
    # コンパイルしてグラフを返す
    return workflow.compile(name='MainAgentGraph')

class PlanExecuteAgent:
    """
    LangGraphを使用したPlan and Execute型のAIエージェント
    """
    def __init__(self, model_name: str = MODEL_NAME, temperature: float = 0.0):
        """
        AIエージェントの初期化
        
        Args:
            model_name: 使用するOpenAIモデル名
            temperature: 生成時の温度パラメータ
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # グラフの構築
        self.agent_graph = build_agent_graph()
        mermaid_code = self.agent_graph.get_graph().draw_mermaid()
        print(f"エージェントのグラフ:\n{mermaid_code}")
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        ユーザークエリに対して回答を生成
        
        Args:
            query: ユーザーからの質問
            
        Returns:
            結果を含む辞書
        """
        # 初期状態の作成
        initial_state = AgentState(query=query)
        
        # グラフの実行
        result = self.agent_graph.invoke(initial_state, {"recursion_limit": 25})
        
        # 結果の作成
        return {
            "status": "success",
            "query": query,
            "main_plan": result['main_plan'],
            "execution_results": result['execution_results'],
            "answer": result['final_answer']
        }
            


# サンプル実行用コード
if __name__ == "__main__":
    
    # エージェントの初期化
    agent = PlanExecuteAgent()
    
    # サンプル質問の実行
    query = "受注データ取込バッチと受注確定バッチの違いを教えてください"
    result = agent.run(query)
    
    # 結果の表示
    if result["status"] == "success":
        # 計画の表示
        print("【計画】")
        for i, step in enumerate(result["main_plan"]):
            print(f"ステップ {step.get('step_number', i+1)}: {step.get('description', '')} ({step.get('step_type', 'unknown')})")
        print("\n")
        
        # 最終回答の表示
        print("【回答】")
        print(result["answer"])
    else:
        print(f"エラーが発生しました: {result['error']}")