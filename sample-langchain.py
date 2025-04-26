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

MODEL_NAME = "gpt-4.1-nano"

model = ChatOpenAI(model=MODEL_NAME, temperature=0)

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")
output_parser = PydanticOutputParser(pydantic_object=Recipe)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
    response_format={"type": "json_object"}
)

# ここで、構造化出力を指定する
chain = prompt | model.with_structured_output(Recipe)
ai_message = chain.invoke({"dish": "カレー"})
print(type(ai_message))
print(ai_message)
