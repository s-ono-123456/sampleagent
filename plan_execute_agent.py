#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
from langchain.chains import LLMChain
from vector_store_loader import load_vector_stores

class PlanStep(BaseModel):
    """計画の各ステップを表すPydanticモデル"""
    step_number: int = Field(description="ステップ番号")
    description: str = Field(description="ステップの説明")
    action_type: str = Field(description="アクションタイプ (search, analyze, synthesize)")

class Plan(BaseModel):
    """計画全体を表すPydanticモデル"""
    steps: List[PlanStep] = Field(description="計画のステップリスト")

class ExecutionResult(BaseModel):
    """ステップ実行結果を表すPydanticモデル"""
    step: PlanStep = Field(description="実行されたステップ")
    result: str = Field(description="実行結果")
    success: bool = Field(description="実行が成功したかどうか")

class Planner:
    """
    ユーザーの質問を分析し、解決のための計画を立てるクラス
    """
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        
        self.plan_prompt_template = PromptTemplate(
            template="""
            あなたは設計書に関する調査・質問対応を行うAIアシスタントです。
            ユーザーからの以下の質問に答えるための調査計画を作成してください。
            
            質問: {query}
            
            計画は以下のステップに分解してください:
            1. 必要な情報を特定するための検索ステップ (action_type: search)
            2. 収集した情報を分析するためのステップ (action_type: analyze)
            3. 最終的な回答を生成するための統合ステップ (action_type: synthesize)
            
            以下の形式でJSON形式の計画を返してください:
            {{
              "steps": [
                {{
                  "step_number": 1,
                  "description": "検索するステップの説明",
                  "action_type": "search"
                }},
                {{
                  "step_number": 2,
                  "description": "分析するステップの説明",
                  "action_type": "analyze"
                }},
                {{
                  "step_number": 3,
                  "description": "統合するステップの説明",
                  "action_type": "synthesize"
                }}
              ]
            }}
            """,
            input_variables=["query"]
        )
        
    def create_plan(self, query: str) -> Plan:
        """ユーザークエリから計画を生成する"""
        plan_chain = self.plan_prompt_template | self.llm | self.parser
        
        try:
            # 構造化出力を解析
            plan_output = plan_chain.invoke({"query": query})
            steps_data = plan_output.get("steps", [])
            
            # Pydanticモデルに変換
            plan_steps = []
            for step_data in steps_data:
                step = PlanStep(
                    step_number=step_data.get("step_number", 1),
                    description=step_data.get("description", ""),
                    action_type=step_data.get("action_type", "search")
                )
                plan_steps.append(step)
            
            return Plan(steps=plan_steps)
        except Exception as e:
            print(f"計画生成中にエラーが発生しました: {str(e)}")
            # フォールバック計画を作成
            return self._create_fallback_plan(query)
            
    def _create_fallback_plan(self, query: str) -> Plan:
        """エラー発生時のフォールバック計画を作成"""
        return Plan(steps=[
            PlanStep(step_number=1, description=f"「{query}」に関連する情報を検索", action_type="search"),
            PlanStep(step_number=2, description="収集した情報を分析", action_type="analyze"),
            PlanStep(step_number=3, description="最終的な回答を生成", action_type="synthesize")
        ])

class Executor:
    """
    計画の各ステップを実行するクラス
    """
    def __init__(self, llm: ChatOpenAI, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.execution_results: List[ExecutionResult] = []
        
        # 検索用のプロンプトテンプレート
        self.search_prompt = PromptTemplate(
            template="""
            以下のステップを実行するために適切な検索クエリを作成してください:
            
            ステップ: {step_description}
            
            元の質問: {query}
            
            検索クエリ (キーワードのみで、簡潔に):
            """,
            input_variables=["step_description", "query"]
        )
        
        # 分析用のプロンプトテンプレート
        self.analyze_prompt = PromptTemplate(
            template="""
            以下の情報を分析し、ステップに関する洞察を提供してください:
            
            ステップ: {step_description}
            
            元の質問: {query}
            
            これまでに収集した情報:
            {collected_info}
            
            分析結果:
            """,
            input_variables=["step_description", "query", "collected_info"]
        )
        
        # 統合用のプロンプトテンプレート
        self.synthesize_prompt = PromptTemplate(
            template="""
            収集したすべての情報に基づいて、ユーザーの質問に対する最終的な回答を作成してください。
            
            質問: {query}
            
            収集した情報:
            {all_results}
            
            回答は明確で簡潔、かつ情報に富んだものにしてください。
            設計書の内容に基づいて事実を述べ、根拠となる情報を含めてください。
            
            回答:
            """,
            input_variables=["query", "all_results"]
        )
    
    def execute_plan(self, plan: Plan, query: str) -> str:
        """計画を実行し、最終的な回答を返す"""
        for step in plan.steps:
            result = self._execute_step(step, query)
            self.execution_results.append(result)
            
            # 実行に失敗した場合、計画を修正する
            if not result.success:
                self._revise_plan(plan, query)
                break
        
        # 最終回答の生成
        return self._generate_final_answer(query)
    
    def _execute_step(self, step: PlanStep, query: str) -> ExecutionResult:
        """ステップの内容に基づいて適切なアクションを実行"""
        try:
            if step.action_type == "search":
                result = self._search_documents(step, query)
                return ExecutionResult(step=step, result=result, success=bool(result))
            
            elif step.action_type == "analyze":
                result = self._analyze_information(step, query)
                return ExecutionResult(step=step, result=result, success=True)
            
            elif step.action_type == "synthesize":
                # 統合ステップは最後に実行するので、ここでは仮の結果を返す
                return ExecutionResult(step=step, result="統合ステップは後で実行します", success=True)
            
            else:
                # 不明なアクションタイプの場合
                return ExecutionResult(
                    step=step,
                    result=f"不明なアクションタイプ: {step.action_type}",
                    success=False
                )
        
        except Exception as e:
            # 例外が発生した場合
            return ExecutionResult(
                step=step,
                result=f"実行中にエラーが発生しました: {str(e)}",
                success=False
            )
    
    def _search_documents(self, step: PlanStep, query: str) -> str:
        """ベクトルストアを使用して関連文書を検索"""
        # 検索クエリの生成
        search_chain = self.search_prompt | self.llm
        search_query = search_chain.invoke({"step_description": step.description, "query": query})
        
        # ベクトルストアで検索
        docs = self.vector_store.similarity_search(search_query, k=3)
        
        if not docs:
            return "関連する情報が見つかりませんでした。"
        
        # 検索結果のフォーマット
        results = []
        for doc in docs:
            source = doc.metadata.get("source", "不明")
            results.append(f"ドキュメント: {source}\n内容: {doc.page_content}")
        
        return "\n\n".join(results)
    
    def _analyze_information(self, step: PlanStep, query: str) -> str:
        """収集した情報を分析"""
        # これまでに収集した情報を取得
        collected_info = "\n\n".join([
            f"ステップ {r.step.step_number}: {r.result}" for r in self.execution_results
            if r.step.action_type == "search"
        ])
        
        if not collected_info:
            return "分析する情報がありません。まず検索ステップを実行してください。"
        
        # 分析の実行
        analyze_chain = self.analyze_prompt | self.llm
        analysis = analyze_chain.invoke({
            "step_description": step.description,
            "query": query,
            "collected_info": collected_info
        })
        
        return analysis
    
    def _revise_plan(self, plan: Plan, query: str) -> None:
        """計画の修正（現在の実装では単に警告を出すだけ）"""
        print(f"警告: 計画の実行中に問題が発生しました。ステップ {self.execution_results[-1].step.step_number}")
    
    def _generate_final_answer(self, query: str) -> str:
        """すべての結果を統合して最終回答を生成"""
        # これまでに収集したすべての結果を取得
        all_results = "\n\n".join([
            f"ステップ {r.step.step_number} ({r.step.action_type}): {r.result}" 
            for r in self.execution_results
        ])
        
        # 最終回答の生成
        synthesize_chain = self.synthesize_prompt | self.llm
        final_answer = synthesize_chain.invoke({
            "query": query, 
            "all_results": all_results
        })
        
        return final_answer

class PlanExecuteAgent:
    """
    Plan and Execute型のAIエージェント
    """
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        """
        AIエージェントの初期化
        
        Args:
            model_name: 使用するOpenAIモデル名
            temperature: 生成時の温度パラメータ
        """
        # LLMの初期化
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # ベクトルストアの読み込み
        self.vector_store = load_vector_stores()
        
        # プランナーとエグゼキュータの初期化
        self.planner = Planner(self.llm)
        self.executor = Executor(self.llm, self.vector_store)
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        ユーザークエリに対して回答を生成
        
        Args:
            query: ユーザーからの質問
            
        Returns:
            結果を含む辞書
        """
        try:
            # 計画の作成
            plan = self.planner.create_plan(query)
            
            # 計画の実行
            answer = self.executor.execute_plan(plan, query)
            
            # 結果の作成
            return {
                "status": "success",
                "query": query,
                "plan": plan,
                "execution_results": self.executor.execution_results,
                "answer": answer
            }
            
        except Exception as e:
            return {
                "status": "error",
                "query": query,
                "error": str(e)
            }

# サンプル実行用コード
if __name__ == "__main__":
    import os
    # OpenAI APIキーの設定
    # os.environ["OPENAI_API_KEY"] = "あなたのOpenAI APIキー"
    
    # エージェントの初期化
    agent = PlanExecuteAgent()
    
    # サンプル質問の実行
    query = "受注データ取込バッチと受注確定バッチの違いを教えてください"
    result = agent.run(query)
    
    # 結果の表示
    if result["status"] == "success":
        # 計画の表示
        print("【計画】")
        for step in result["plan"].steps:
            print(f"ステップ {step.step_number}: {step.description} ({step.action_type})")
        print("\n")
        
        # 最終回答の表示
        print("【回答】")
        print(result["answer"])
    else:
        print(f"エラーが発生しました: {result['error']}")