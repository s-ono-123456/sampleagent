from typing import List, Optional, TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

# GraphState型定義
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    plans: Optional[List[str]]
    current_plan_index: Optional[int]
    completed: Optional[bool]
    last_message: Optional[AnyMessage]
    completed_plans: Optional[bool]
    subplan_index: Optional[int]

# テスト計画のPydanticモデル
class TestPlan(BaseModel):
    steps: List[str] = Field(description="テスト実行の具体的な手順のリスト")

# プロンプトテンプレート
TEST_AGENT_TEMPLATE = """
あなたはテスト自動化の専門家です。
あなたは、Playwrightというブラウザを操作するツールを使用してテストを実行します。
直前にツールを使用している場合、その出力を利用して必要な情報を抽出してください。

与えられていない場合は、始めての実行として考えてください。

次の手順を実行してください: {current_plan}
直前のツールの出力: {last_content}
"""

PLANNING_SYSTEM_PROMPT = """あなたはテスト自動化の専門家です。
後続のテスト実行ノードでは、Playwrightというブラウザを操作するツールを使用してテストを実行します。
ユーザーからの指示に基づいて、ブラウザテストの具体的な実行手順を作成してください。
各ステップは明確で具体的な1つの操作に分解してください。
"""