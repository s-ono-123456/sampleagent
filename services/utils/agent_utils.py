from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from services.utils.state_utils import State
from services.utils.model_utils import init_chat_model, MODEL_NAME

# モデルのインスタンスを作成
llm = init_chat_model(MODEL_NAME)

def query_analyzer_agent(state: State):
    """
    質問分析エージェント
    ユーザーからの質問を分析し、必要な情報を特定する
    
    Args:
        state: 現在の状態
        
    Returns:
        更新された状態（質問カテゴリと検索クエリを追加）
    """
    from services.utils.state_utils import Questions
    
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
    
    # 結果を構造化
    try:
        questions = analysis_result.questions
    except:
        # 解析に失敗した場合は元の質問をそのまま使用
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
    from services.vector_store_loader import load_vector_stores
    from services.utils.model_utils import EMBEDDING_MODEL_NAME
    
    # 結果を記載するリスト変数
    results = []
    questions = state["questions"]
    for question in questions:
        question_category = question.question_category
        search_query = question.search_query
        
        # ベクトルストアを読み込む
        vector_store = load_vector_stores(EMBEDDING_MODEL_NAME)
        
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
    from services.utils.state_utils import Evaluates
    
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

def build_agent_graph():
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