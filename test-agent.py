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
from langgraph.graph import END
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage, AIMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import InjectedState


# 環境変数の設定
# os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="sampleagent"
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

MODEL_NAME = "gpt-4.1-nano"
embedding_model_name = "pkshatech/GLuCoSE-base-ja-v2"

# モデルの設定
model = ChatOpenAI(model=MODEL_NAME, temperature=0)

# HuggingFaceのGLuCoSE埋め込みモデルをロード
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ベクトルストアを読み込む
db = load_vector_stores(embedding_model_name)

retriever = db.as_retriever()

planning_prompt = ChatPromptTemplate.from_template('''\
以下の質問に回答する計画を考えてください。複数のステップに分けて考えてください。
また、各ステップでは、関連する文書を検索することが出来、基本的にその文書を用いて回答を考えます。

各ステップで、関連する文書を利用して何を考えれば良いかまで考えてください。

質問: {question}
''')

class PlanningOutput(BaseModel):
     plans: list[str] = Field(..., description="計画のリスト")

plan_generation_chain = (
    planning_prompt
    | model.with_structured_output(PlanningOutput)
    | (lambda x: x.plans)
)

question = "受注処理について教えてください。"

plan_output = plan_generation_chain.invoke(
    {
        "question": question
    }
)

# プランニングの結果を表示
# print("Plan Output:", plan_output)

retriever_prompt = ChatPromptTemplate.from_template('''\
以下の計画を遂行するために、関連する文書を検索するクエリを3つ考えてください。

計画: {plan}
''')

prompt = ChatPromptTemplate.from_template('''\
以下はユーザからの質問に対して回答するための一ステップです。
以下の計画を遂行するために関連文書を検索しました。
関連文書を考慮して計画を遂行してください。
なお、関連文書に記載がない場合は回答ができない、と答えてください。

質問: {question}

計画: {plan}

関連文書: """
{context}
"""
''')

class RetrieverOutput(BaseModel):
     querys: list[str] = Field(..., description="クエリのリスト")

retriever_chain = (
    retriever_prompt
    | model.with_structured_output(RetrieverOutput)
    | (lambda x: x.querys)
)

multi_query_rag_chain = {
    "question": RunnablePassthrough(),
    "plan": RunnablePassthrough(),
    "context": retriever_chain | retriever.map(),
} | prompt | model | StrOutputParser()

retriever_outputs = []

for plan in plan_output:
    retriever_output = multi_query_rag_chain.invoke(
        {
            "plan": plan,
            "question": question
        }
    )
    retriever_outputs.append(retriever_output)


responce_prompt = ChatPromptTemplate.from_template('''\
以下はユーザからの質問に対して回答するため、事前に計画を立てて、ステップごとに分けて考えた結果です。
各ステップで考えた結果を考慮して、ユーザの質問に対する最終的な回答を考えてください。

質問: {question}

計画: {plans}

結果："""
{context}
"""
''')

responce_rag_chain = {
    "question": RunnablePassthrough(),
    "plans": RunnablePassthrough(),
    "context": RunnablePassthrough(),
} | responce_prompt | model | StrOutputParser()

responce = responce_rag_chain.invoke(
    {
        "question": question,
        "plans": plan_output,
        "context": retriever_outputs
    }
)

print("Final Response:", responce)