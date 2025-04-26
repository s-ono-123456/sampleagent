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


# 環境変数の設定
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="sampleagent"
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4.1-nano"
embedding_model_name = "pkshatech/GLuCoSE-base-ja-v2"

model = ChatOpenAI(model=MODEL_NAME, temperature=0)

# HuggingFaceのGLuCoSE埋め込みモデルをロード
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。
文脈: """
{context}
"""

質問: {question}
''')

# ベクトルストアを読み込む
db = load_vector_stores(embedding_model_name)

retriever = db.as_retriever()

class QueryGenerationOutput(BaseModel):
     queries: list[str] = Field(..., description="検索クエリのリスト")

query_generation_prompt = ChatPromptTemplate.from_template("""\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}
""")

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

# hypothetical_prompt = ChatPromptTemplate.from_template("""\
# 次の質問に回答する一文を書いてください。

# 質問: {question}
# """)

# hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

# hyde_rag_chain = {
#     "question": RunnablePassthrough(),
#     "context": hypothetical_chain | retriever,
# } | prompt | model | StrOutputParser()

# output = hyde_rag_chain.invoke("受注処理の概要を教えて")
