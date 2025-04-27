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

# モデルの初期化
def init_chat_model(model_name: str) -> ChatOpenAI:
    """
    モデルを初期化する関数
    
    Args:
        model_name: モデル名
        
    Returns:
        初期化されたモデル
    """
    # OpenAIのAPIキーを取得
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # モデルの初期化
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0.0,
        openai_api_key=openai_api_key,
        streaming=True,
        verbose=True,
        max_tokens=2000,
        request_timeout=60,
    )
    # ツールの初期化
    tool = TavilySearch(max_results=2)
    tools = []

    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools

llm = init_chat_model(MODEL_NAME)

class Question(BaseModel):
    question_category: str
    search_query: str

class Questions(BaseModel):
    questions: list[Question]

class Evaluate(BaseModel):
    name: str
    relation: int
    usefulness: int
    appendix: str

class Evaluates(BaseModel):
    evaluates: list[Evaluate]


class State(TypedDict):
    # メッセージタイプを管理するための状態クラス
    # `add_messages` 関数はこの状態キーの更新方法を定義します
    # (この場合、メッセージを上書きせずリストに追加します)
    messages: Annotated[list, add_messages]
    # 質問リスト
    questions: Questions
    # 関連設計書リスト
    relevant_documents: list
    # 有用設計書リスト
    useful_documents: list
    # 評価値
    evaluates: Evaluates
    # 情報ギャップの有無
    has_information_gap: bool = False
    # ループ回数
    loop_count: int = 0
    # 最終回答
    final_response: str
    # 最後のノード情報
    last_node: str = ""
    # 最終回答のチェック
    check_result: str

# 各専門エージェントの実装

def query_analyzer_agent(state: State):
    """
    質問分析エージェント
    ユーザーからの質問を分析し、必要な情報を特定する
    
    Args:
        state: 現在の状態
        
    Returns:
        更新された状態（質問カテゴリと検索クエリを追加）
    """
    messages = state["messages"]
    user_query = messages[-1].content if isinstance(messages[-1].content, str) else messages[-1].content[0].text
    
    # LLMを使用して質問を分析する
    has_gap = state.get("has_information_gap", False)
    loop_count = state.get("loop_count", 0)
    loop_count += 1
    state["loop_count"] = loop_count


    if has_gap: 
        # 情報ギャップがある場合は、質問を再分析する
        system_message = """
        あなたは情報補完の専門家です。ユーザーの質問と現在の情報を分析し、不足している情報を特定してください。
        以下の情報の組み合わせを抽出してください：
        1. 質問のカテゴリ：「batch_design」（バッチ設計に関する質問）または「screen_design」（画面設計に関する質問）
        2. 重要キーワード：質問から抽出された不足している情報を検索するために役立つ新規の質問文
        
        """
    else:
        system_message = """
        あなたは質問分析の専門家です。ユーザーからの質問を分析し、以下の情報の組み合わせを抽出してください：
        1. 質問のカテゴリ：「batch_design」（バッチ設計に関する質問）または「screen_design」（画面設計に関する質問）
        2. 重要キーワード：質問から抽出された検索に役立つ新規の質問文
        
        """
    
    analysis_prompt = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"以下の質問を分析してください。質問の組み合わせを３つ以上回答してください。：\n{user_query}")
    ]
    
    analysis_result = llm.with_structured_output(Questions).invoke(analysis_prompt)
    
    # JSON形式の結果をパース
    try:
        # result_dict = json.loads(analysis_result.content)
        questions = analysis_result.questions
    except:
        # JSONパースに失敗した場合は元の質問をそのまま使用
        questions = [{"question_category": "unknown", "search_query": user_query}]
    # 状態を更新
    return {
        "questions": questions,
        "last_node": "query_analyzer",
    }

def search_agent(state: State):
    """
    検索エージェント
    適切な設計書を検索・取得する
    
    Args:
        state: 現在の状態
        
    Returns:
        更新された状態（関連設計書リストを追加）
    """
    # 結果を記載するリスト変数

    results = []
    questions = state["questions"]
    for question in questions:
        question_category = question.question_category
        search_query = question.search_query
        
        # ベクトルストアを読み込む
        vector_store = load_vector_stores(embedding_model_name)
        
        # 検索を実行
        search_results = vector_store.similarity_search_with_score(search_query, k=5)
        
        for doc, score in search_results:
            if doc.page_content in results:
                # 既に結果に含まれている場合はスキップ
                continue
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
    
    # 状態を更新
    return {
        "relevant_documents": results,
        "last_node": "search",
    }

def information_evaluator_agent(state: State):
    """
    情報評価エージェント
    検索結果から関連性の高い情報を評価・選択する
    
    Args:
        state: 現在の状態
        
    Returns:
        更新された状態（評価済み情報リストを追加）
    """
    messages = state["messages"]
    user_query = messages[-1].content if isinstance(messages[-1].content, str) else messages[-1].content[0].text
    relevant_documents = state["relevant_documents"]

    useful_documents = state.get("useful_documents", [])
    
    if not relevant_documents:
        return {
            "evaluated_information": [],
            "has_information_gap": True,
            "last_node": "information_evaluator",
        }
    
    # LLMを使用して情報の関連性を評価
    system_message = """
    あなたは情報評価の専門家です。ユーザーの質問と検索結果を分析し、関連性の高い情報を選択してください。
    各検索結果に対して、以下の評価を行ってください：
    0. name: 文書1～文書XX
    1. relations: 関連度（0～10）：質問に対する情報の関連性
    2. usefulness: 有用性（0～10）：質問に答えるためにどれだけ役立つか
    3. appendix: 補足として、資料に対するコメントを追加してください。
    
    JSON形式で回答してください。
    """
    
    # 検索結果を文字列として連結
    docs_text = "\n\n".join([f"文書{i+1}:\n{doc['content']}" for i, doc in enumerate(relevant_documents)])
    
    evaluation_prompt = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"以下のテキストを確認して評価してください。各文書ごとに対して評価してください。質問: {user_query}\n\n検索結果:\n{docs_text}")
    ]
    
    evaluation_result = llm.with_structured_output(Evaluates).invoke(evaluation_prompt)
    
    
    # LLMの回答を構造化
    # 実際の実装では、適切なJSON解析が必要
    has_gap = False
    evaluates = evaluation_result.evaluates

    for evaluate, relevant_document in zip(evaluates, relevant_documents):
        usefulness = evaluate.usefulness
        relation = evaluate.relation
        appendix = evaluate.appendix

        if usefulness > 7:
            if relevant_document in useful_documents:
                # 既に有用文書リストに含まれている場合はスキップ
                continue
            # 有用性が高い場合はリストに追加
            useful_documents.append(relevant_document)

    if len(useful_documents) < 6:
        has_gap = True
    
    # 状態を更新
    return {
        "evaluates": evaluation_result,
        "useful_documents": useful_documents,
        "has_information_gap": has_gap,
        "last_node": "information_evaluator",
    }

def response_generator_agent(state: State):
    """
    回答生成エージェント
    収集した情報から回答を生成する
    
    Args:
        state: 現在の状態
        
    Returns:
        更新された状態（最終回答を追加）
    """
    messages = state["messages"]
    user_query = messages[-1].content if isinstance(messages[-1].content, str) else messages[-1].content[0].text
    useful_documents = state["useful_documents"]
    
    # 関連文書を文字列として連結
    docs_text = "\n\n".join([f"文書{i+1}:\n{doc['content']}" for i, doc in enumerate(useful_documents)])
    
    # LLMを使用して回答を生成
    system_message = """
    あなたは回答生成の専門家です。ユーザーの質問と収集された情報を分析し、正確で包括的な回答を生成してください。
    回答には以下の要素を含めてください：
    1. 質問への直接的な回答
    2. 関連する重要情報の要約
    3. 情報の出典（可能な場合）
    
    回答は日本語で、わかりやすく構造化してください。
    """
    
    response_prompt = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"質問: {user_query}\n\n利用可能な情報:\n{docs_text}")
    ]
    
    response_result = llm.invoke(response_prompt)
    
    # 回答を抽出
    final_response = response_result.content
    
    # 状態を更新
    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
        "last_node": "response_generator",
    }

def response_evaluate_agent(state: State):
    """
    回答精度判断エージェント
    収集した回答が質問に答えられているかを判断する。
    
    Args:
        state: 現在の状態
        
    Returns:
        更新された状態（最終回答を追加）
    """
    messages = state["messages"]
    user_query = messages[-1].content if isinstance(messages[-1].content, str) else messages[-1].content[0].text
    final_response = state["final_response"]
    
    # 関連文書を文字列として連結
    
    # LLMを使用して回答を生成
    system_message = """
    あなたは回答内容をチェックする専門家です。
    ユーザーの質問と回答内容を分析し、適切に質問に回答出来ているかをチェックしてください。
    チェック内容には以下の要素を含めてください：
    1. 質問で求められている内容を満たせているか
    2. 回答内容に不足している内容はないか
    3. 情報の出典が正しく記載されているか
    
    回答は日本語で、わかりやすく構造化してください。
    """
    
    check_prompt = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"質問: {user_query}\n\n回答:\n{final_response}")
    ]
    
    check_result = llm.invoke(check_prompt)
    
    # 回答を抽出
    check_result = check_result.content
    
    # 状態を更新
    return {
        "check_result": check_result,
        "messages": [AIMessage(content=check_result)],
        "last_node": "response_evaluate",
    }

def gragh_build():
    """
    エージェントの処理フローを定義するグラフを構築する
    
    Returns:
        graph: コンパイルされたStateGraph
    """
    # ステートグラフの作成
    graph_builder = StateGraph(State)
    
    # 各エージェントをノードとして追加
    graph_builder.add_node("query_analyzer", query_analyzer_agent)
    graph_builder.add_node("search", search_agent)
    graph_builder.add_node("information_evaluator", information_evaluator_agent)
    graph_builder.add_node("response_generator", response_generator_agent)
    graph_builder.add_node("response_evaluate", response_evaluate_agent)
    
    # エッジの追加：基本的なフロー
    graph_builder.add_edge(START, "query_analyzer")
    graph_builder.add_edge("query_analyzer", "search")
    graph_builder.add_edge("search", "information_evaluator")
    graph_builder.add_edge("response_generator", "response_evaluate")
    graph_builder.add_edge("response_generator", END)

    def check_condition(state: State) -> str:
        # 情報ギャップがある場合は再度質問分析を行う
        return_value = ""
        if state.get("loop_count", 0) > 2:
            return_value = "query_analyzer"
        else:
            if state.get("has_information_gap", False):
                return_value = "query_analyzer"
            # 情報ギャップがない場合は、回答生成エージェントに進む
            return_value = "response_generator"
        return return_value
    
    # コントローラーによる条件付きルーティング
    graph_builder.add_conditional_edges(
        "information_evaluator", check_condition, 
        path_map={
            "query_analyzer": "query_analyzer",
            "response_generator": "response_generator"
        }
    )
    
    # メモリの設定
    memory = MemorySaver()
    
    # グラフのコンパイル
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


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
