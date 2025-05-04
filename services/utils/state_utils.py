from pydantic import BaseModel, Field
from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import AnyMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages

class Question(BaseModel):
    """質問を解析した結果を表すモデル"""
    question_category: str
    search_query: str

class Questions(BaseModel):
    """複数の質問を格納するモデル"""
    questions: list[Question]

class Evaluate(BaseModel):
    """文書評価の結果を表すモデル"""
    name: str
    relation: int
    usefulness: int
    appendix: str

class Evaluates(BaseModel):
    """複数の評価結果を格納するモデル"""
    evaluates: list[Evaluate]

class State(TypedDict):
    """
    エージェントの状態を管理するクラス
    """
    # メッセージタイプを管理するための状態キー
    # `add_messages` 関数はこの状態キーの更新方法を定義します
    # (この場合、メッセージを上書きせずリストに追加します)
    messages: Annotated[list, add_messages]
    # 質問リスト
    questions: Optional[Questions] = None
    # 関連設計書リスト
    relevant_documents: Optional[list] = None
    # 有用設計書リスト
    useful_documents: Optional[list] = None
    # 評価値
    evaluates: Optional[Evaluates] = None
    # 情報ギャップの有無
    has_information_gap: bool = False
    # ループ回数
    loop_count: int = 0
    # 最終回答
    final_response: Optional[str] = None
    # 最後のノード情報
    last_node: str = ""
    # 最終回答のチェック
    check_result: Optional[str] = None

# スクリーンショット処理を行うヘルパー関数
def handle_screenshot(message: AnyMessage) -> None:
    """
    ツールメッセージからスクリーンショットを処理する
    
    Args:
        message: 処理するメッセージ
    """
    if isinstance(message, ToolMessage):
        from services.utils.screenshot_utils import process_screenshot
        process_screenshot(message)