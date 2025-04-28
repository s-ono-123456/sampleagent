# 設計書調査・質問対応AIエージェント - Plan and Execute型

## 1. 概要

このAIエージェントは、設計書に関する調査や質問に回答・対応するために設計されたPlan and Execute型のシステムです。設計書の内容を理解し、ユーザーからの複雑な質問に対して段階的な計画を立てながら回答を導き出します。

## 2. アーキテクチャ

### 2.1 全体構成

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  ユーザー入力    │───▶│  プランナー     │───▶│  エグゼキュータ  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              ▲                        │
                              │                        ▼
                        ┌──────────────────┐    ┌─────────────────┐
                        │   ベクトルDB    │◀───│    回答生成     │
                        └──────────────────┘    └─────────────────┘
```

### 2.2 主要コンポーネント

1. **ユーザーインターフェース**
   - ユーザーからの質問を受け付け
   - 回答を表示

2. **プランナー**
   - ユーザーの質問を分析
   - 解決に必要なステップを計画
   - サブタスクへの分解

3. **エグゼキュータ**
   - プランに基づいて各ステップを実行
   - ベクトルDBからの情報取得
   - 中間結果の評価と計画の修正

4. **ベクトルデータベース**
   - 設計書のベクトル化されたデータを格納
   - 類似度検索による関連情報の抽出

5. **回答生成**
   - 収集した情報を統合
   - ユーザーに理解しやすい形で回答を生成

## 3. 実装詳細

### 3.1 プランナーの実装

```python
class Planner:
    def __init__(self, llm):
        self.llm = llm
    
    def create_plan(self, query):
        # ユーザークエリから計画を生成
        plan_prompt = f"""
        以下の質問に対する調査計画を作成してください：
        {query}
        
        計画は以下の形式で具体的なステップを含めてください：
        1. [ステップ1の説明]
        2. [ステップ2の説明]
        ...
        """
        plan = self.llm.predict(plan_prompt)
        return self._parse_plan(plan)
    
    def _parse_plan(self, plan_text):
        # 計画をステップごとに分割して構造化
        steps = []
        for line in plan_text.strip().split('\n'):
            if line.strip() and line[0].isdigit():
                steps.append(line.strip())
        return steps
```

### 3.2 エグゼキュータの実装

```python
class Executor:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.execution_results = []
    
    def execute_plan(self, plan, query):
        for step in plan:
            # 各ステップを実行
            result = self._execute_step(step, query)
            self.execution_results.append({"step": step, "result": result})
            
            # ステップの結果に基づいて計画を調整するかどうか判断
            if self._should_revise_plan(result):
                return self._revise_plan(plan, query)
        
        # 最終回答の生成
        return self._generate_final_answer(query)
    
    def _execute_step(self, step, query):
        # ステップの内容に基づいて適切なアクションを実行
        if "検索" in step or "調査" in step:
            return self._search_documents(step, query)
        elif "比較" in step or "分析" in step:
            return self._analyze_information(step)
        else:
            return self._default_execution(step, query)
    
    def _search_documents(self, step, query):
        # ベクトルストアを使用して関連文書を検索
        search_query = self._extract_search_terms(step, query)
        documents = self.vector_store.search(search_query)
        return documents
    
    def _analyze_information(self, step):
        # これまでに収集した情報を分析
        collected_info = [result["result"] for result in self.execution_results]
        analysis_prompt = f"""
        以下の情報を分析し、{step}に関する洞察を提供してください：
        {collected_info}
        """
        return self.llm.predict(analysis_prompt)
    
    def _default_execution(self, step, query):
        # その他のタイプのステップの実行
        execution_prompt = f"""
        次のステップを実行してください：
        {step}
        
        元の質問：{query}
        """
        return self.llm.predict(execution_prompt)
    
    def _should_revise_plan(self, result):
        # 結果に基づいて計画を見直す必要があるか判断
        if not result or "情報が見つかりません" in result:
            return True
        return False
    
    def _revise_plan(self, original_plan, query):
        # 計画の修正
        revision_prompt = f"""
        次の計画を実行中に問題が発生しました。
        元の質問: {query}
        元の計画: {original_plan}
        これまでの結果: {self.execution_results}
        
        より効果的な新しい計画を提案してください。
        """
        revised_plan = self.llm.predict(revision_prompt)
        return self._parse_revised_plan(revised_plan)
    
    def _generate_final_answer(self, query):
        # すべての結果を統合して最終回答を生成
        results_summary = "\n".join([f"ステップ: {r['step']}\n結果: {r['result']}" for r in self.execution_results])
        answer_prompt = f"""
        次の質問に対する回答を、収集したすべての情報に基づいて作成してください：
        質問: {query}
        
        収集した情報:
        {results_summary}
        
        明確で簡潔、かつ情報に富んだ回答を提供してください。
        """
        return self.llm.predict(answer_prompt)
```

### 3.3 ベクトルデータベース接続

```python
class VectorStore:
    def __init__(self, index_path):
        self.index = self._load_index(index_path)
    
    def _load_index(self, index_path):
        # FAISSインデックスの読み込み
        return faiss.read_index(index_path)
    
    def search(self, query, top_k=5):
        # クエリベクトルの生成
        query_vector = self._vectorize_query(query)
        
        # 類似度検索の実行
        scores, indices = self.index.search(query_vector, top_k)
        
        # 結果の取得と整形
        results = self._format_results(scores, indices)
        return results
    
    def _vectorize_query(self, query):
        # クエリをベクトル化
        # 実装はベクトル化手法による
        pass
    
    def _format_results(self, scores, indices):
        # 検索結果を整形
        # インデックスから対応する文書情報を取得
        pass
```

## 4. データフロー

1. ユーザーが設計書に関する質問を入力
2. プランナーが質問を分析し、解決のための計画を立てる
3. エグゼキュータが計画の各ステップを実行
   a. 関連情報をベクトルDBから検索
   b. 情報の分析・統合
   c. 必要に応じて計画を修正
4. 収集した情報から最終的な回答を生成
5. ユーザーに回答を提示

## 5. エージェントの特徴

### 5.1 長所

- **段階的アプローチ**: 複雑な質問を小さなタスクに分解して解決
- **適応性**: 実行結果に基づいて計画を動的に調整
- **透明性**: 回答の導出過程をユーザーに提示可能
- **スケーラビリティ**: 新しい設計書データを追加しやすい構造

### 5.2 課題と対策

| 課題 | 対策 |
|------|------|
| 検索精度の限界 | ベクトル化手法の改善、定期的な再学習 |
| 計画の複雑さ | 適切な抽象化レベルでの計画立案 |
| 実行時間の長さ | キャッシング、並列処理の導入 |
| 誤った情報の混入 | 信頼性スコアの導入、ソース引用の徹底 |

## 6. 実装ロードマップ

1. **フェーズ1**: 基本的なPlan and Execute機能の実装
   - プランナーとエグゼキュータの基本機能
   - シンプルなベクトル検索

2. **フェーズ2**: 高度な検索と分析機能
   - 複数のベクトルストアの統合
   - コンテキスト認識の向上

3. **フェーズ3**: 自己改善と適応機能
   - フィードバックに基づく学習
   - 計画戦略の最適化

## 7. 評価方法

- **正確性**: 生成された回答の正確さ
- **関連性**: 質問に対する回答の関連度
- **完全性**: 回答が質問のすべての側面をカバーしているか
- **効率性**: 回答生成までの時間と計算リソース
- **ユーザー満足度**: フィードバックに基づく評価

## 8. 使用技術

- **LLM**: OpenAI GPT-4 または同等のモデル
- **ベクトルDB**: FAISS
- **埋め込み**: OpenAI Ada または同等のモデル
- **開発言語**: Python
- **フレームワーク**: LangChain

## 9. 設計書データの管理

- 設計書を適切な単位で分割
- メタデータ（作成日、カテゴリなど）の付与
- 定期的な更新プロセスの確立
- ベクトルインデックスの最適化と管理