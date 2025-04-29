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

MODEL_NAME = "gpt-4.1-nano"
embedding_model_name = "pkshatech/GLuCoSE-base-ja-v2"

# プロンプトテンプレートの定義
# RAG用テンプレート
RAG_PROMPT_TEMPLATE = '''\
以下の文脈だけを踏まえて質問に回答してください。
文脈: """
{context}
"""

質問: {question}
'''

# クエリ生成用テンプレート
QUERY_GENERATION_TEMPLATE = """\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}
"""

# Hypothetical RAGのテンプレート (コメントアウト部分も定義)
HYPOTHETICAL_PROMPT_TEMPLATE = """\
次の質問に回答する一文を書いてください。

質問: {question}
"""

model = ChatOpenAI(model=MODEL_NAME, temperature=0)

# HuggingFaceのGLuCoSE埋め込みモデルをロード
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# ベクトルストアを読み込む
db = load_vector_stores(embedding_model_name)

retriever = db.as_retriever()

class QueryGenerationOutput(BaseModel):
     queries: list[str] = Field(..., description="検索クエリのリスト")

query_generation_prompt = ChatPromptTemplate.from_template(QUERY_GENERATION_TEMPLATE)

query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
)

multi_query_rag_chain = {
    "question": RunnablePassthrough(),
    "context": query_generation_chain | retriever.map(),
} | prompt | model | StrOutputParser()

output = multi_query_rag_chain.invoke("受注処理の概要を教えて")

print(output)



####################################################################################
# Hypothetical RAG Chainの例
####################################################################################

# hypothetical_prompt = ChatPromptTemplate.from_template(HYPOTHETICAL_PROMPT_TEMPLATE)

# hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

# hyde_rag_chain = {
#     "question": RunnablePassthrough(),
#     "context": hypothetical_chain | retriever,
# } | prompt | model | StrOutputParser()

# output = hyde_rag_chain.invoke("受注処理の概要を教えて")
