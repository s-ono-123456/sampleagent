from openai import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_tavily import TavilySearch

# デフォルトモデル設定
MODEL_NAME = "gpt-4.1-nano"
EMBEDDING_MODEL_NAME = "pkshatech/GLuCoSE-base-ja-v2"

def init_chat_model(model_name: str = MODEL_NAME) -> ChatOpenAI:
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