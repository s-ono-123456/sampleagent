import os
from typing import List, Dict, Any, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 自作モジュールのインポート
from services.vector_store_loader import load_vector_stores, load_vector_stores_category

# 定数の定義
DEFAULT_MODEL_NAME = "gpt-4.1-nano"
DEFAULT_EMBEDDING_MODEL_NAME = "pkshatech/GLuCoSE-base-ja-v2"
MAX_RESULTS = 5

# プロンプトテンプレート
RAG_PROMPT_TEMPLATE = '''
あなたは設計書検索アシスタントです。以下の文脈に基づいて、ユーザーの質問に対して明確かつ正確に回答してください。
ユーザーは設計書の内容について質問しています。文脈として提供される検索結果だけを情報源として回答を生成してください。

文脈:
"""
{context}
"""

質問: {question}

回答を生成する際には:
1. 設計書からの情報を引用して回答を構成してください
2. 文脈に情報がない場合は、情報がないことを正直に伝えてください
3. 文脈に基づいた事実だけを回答し、推測や一般的な知識に基づく情報の追加は避けてください
4. 回答は簡潔に、ポイントを明確にして提供してください
'''

# クエリ生成用プロンプト
QUERY_GENERATION_TEMPLATE = '''
以下の質問に対して、設計書を検索するための3つの異なる検索クエリを生成してください。
元の質問の言い換えや、より具体的なキーワードの組み合わせなど、多様な検索クエリを提案してください。

質問: {question}
'''


class QueryGenerationOutput(BaseModel):
    """検索クエリ生成の出力クラス"""
    queries: List[str] = Field(..., description="検索クエリのリスト")


class RAGQueryProcessor:
    """クエリ処理を担当するクラス"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """初期化"""
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.query_generation_prompt = ChatPromptTemplate.from_template(QUERY_GENERATION_TEMPLATE)
    
    def create_search_queries(self, question: str) -> List[str]:
        """ユーザーの質問から複数の検索クエリを生成する"""
        query_generation_chain = (
            self.query_generation_prompt
            | self.model.with_structured_output(QueryGenerationOutput)
            | (lambda x: x.queries)
        )
        
        try:
            queries = query_generation_chain.invoke({"question": question})
            return queries
        except Exception as e:
            print(f"検索クエリの生成中にエラーが発生しました: {str(e)}")
            # エラーの場合は元の質問をそのまま返す
            return [question]
    
    def determine_category(self, question: str) -> Optional[str]:
        """質問からカテゴリを判定する（キーワードベース）"""
        # 簡易的なキーワードマッチングによるカテゴリ判定
        batch_keywords = ["バッチ", "処理", "確定", "集計", "発注", "在庫", "棚卸"]
        screen_keywords = ["画面", "入力", "表示", "管理", "ユーザー", "インターフェース"]
        
        # 質問内の単語をチェック
        lower_question = question.lower()
        
        batch_match = any(keyword in lower_question for keyword in batch_keywords)
        screen_match = any(keyword in lower_question for keyword in screen_keywords)
        
        if batch_match and not screen_match:
            return "batch_design"
        elif screen_match and not batch_match:
            return "screen_design"
        else:
            # 両方のキーワードを含むか、どちらも含まない場合はNoneを返す
            return None


class RAGRetriever:
    """文書検索を担当するクラス"""
    
    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME):
        """初期化"""
        self.embedding_model_name = embedding_model_name
        # 埋め込みモデルの初期化
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def search_documents(self, queries: List[str], categories: Optional[List[str]] = None, max_results: int = MAX_RESULTS) -> List[Dict[str, Any]]:
        """複数のクエリを使用して文書を検索する"""
        try:
            # ベクトルストアの読み込み（カテゴリに基づく）
            if categories and len(categories) > 0:
                # 複数のカテゴリが指定された場合は、load_vector_stores_multiple_categoriesを使用
                from services.vector_store_loader import load_vector_stores_multiple_categories
                db = load_vector_stores_multiple_categories(categories, self.embedding_model_name)
            else:
                # カテゴリが指定されなかった場合は全てのインデックスを読み込む
                from services.vector_store_loader import load_vector_stores
                db = load_vector_stores(self.embedding_model_name)
            
            # リトリーバーの設定
            retriever = db.as_retriever(search_kwargs={"k": max_results})
            
            # 各クエリで検索を実行し、結果を統合
            all_results = []
            for query in queries:
                results = retriever.invoke(query)
                all_results.extend(results)
            
            # 重複除去と結果のフォーマット
            unique_results = self._remove_duplicates(all_results)
            
            # 結果を適切な形式に変換
            formatted_results = self._format_search_results(unique_results)
            
            return formatted_results
            
        except Exception as e:
            print(f"文書検索中にエラーが発生しました: {str(e)}")
            return []
    
    def _remove_duplicates(self, results: List[Any]) -> List[Any]:
        """検索結果の重複を除去する"""
        seen = set()
        unique_results = []
        
        for result in results:
            # 文書の冒頭部分をキーとして使用
            content_key = result.page_content[:100]  # 先頭100文字を識別子として使用
            
            if content_key not in seen:
                seen.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _format_search_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """検索結果を表示用にフォーマットする"""
        formatted_results = []
        
        for result in results:
            # メタデータからファイル名を取得
            source = result.metadata.get("source", "不明")
            file_name = os.path.basename(source) if source else "不明"
            
            # スコアを取得（存在する場合）
            score = result.metadata.get("score", None)
            
            formatted_result = {
                "content": result.page_content,
                "file_name": file_name,
                "score": score,
                "source": source
            }
            
            formatted_results.append(formatted_result)
        
        # スコア順にソート（スコアが存在する場合）
        if formatted_results and formatted_results[0]["score"] is not None:
            formatted_results.sort(key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)
        
        return formatted_results


class RAGGenerator:
    """回答生成を担当するクラス"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """初期化"""
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    def generate_answer(self, question: str, search_results: List[Dict[str, Any]]) -> str:
        """検索結果に基づいて回答を生成する"""
        try:
            # 検索結果がない場合
            if not search_results:
                return "申し訳ありませんが、その質問に関連する設計書情報が見つかりませんでした。質問の言い回しを変えるか、別の質問をお試しください。"
            
            # 検索結果からコンテキストを作成
            context = self._create_context(search_results)
            
            # 回答生成チェーンを構築
            rag_chain = (
                self.prompt 
                | self.model 
                | StrOutputParser()
            )
            
            # 回答を生成
            answer = rag_chain.invoke({"question": question, "context": context})
            
            return answer
            
        except Exception as e:
            print(f"回答生成中にエラーが発生しました: {str(e)}")
            return "申し訳ありません、回答の生成中にエラーが発生しました。もう一度お試しください。"
    
    def _create_context(self, search_results: List[Dict[str, Any]]) -> str:
        """検索結果からプロンプト用のコンテキストを作成する"""
        context_parts = []
        
        for idx, result in enumerate(search_results, 1):
            content = result["content"]
            file_name = result["file_name"]
            
            # 各検索結果を番号付きでフォーマット
            formatted_result = f"【文書{idx}】\nファイル: {file_name}\n\n{content}\n"
            context_parts.append(formatted_result)
        
        # 全ての検索結果を連結
        return "\n".join(context_parts)


class RAGManager:
    """RAGプロセス全体を管理するクラス"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME):
        """初期化"""
        self.query_processor = RAGQueryProcessor(model_name)
        self.retriever = RAGRetriever(embedding_model_name)
        self.generator = RAGGenerator(model_name)
    
    def process_query(self, question: str, categories: Optional[List[str]] = None, max_results: int = MAX_RESULTS) -> Tuple[str, List[Dict[str, Any]]]:
        """クエリを処理して回答と検索結果を返す"""
        try:
            # 検索クエリの生成
            queries = self.query_processor.create_search_queries(question)
            
            # カテゴリの決定（指定がない場合、質問から推測）
            if not categories or len(categories) == 0:
                # カテゴリが指定されていない場合、質問から推測して単一カテゴリをリストに変換
                suggested_category = self.query_processor.determine_category(question)
                categories = [suggested_category] if suggested_category else None
            
            # 文書検索の実行
            search_results = self.retriever.search_documents(
                queries, 
                categories, 
                max_results
            )
            
            # 回答生成
            answer = self.generator.generate_answer(question, search_results)
            
            return answer, search_results
            
        except Exception as e:
            error_message = f"処理中にエラーが発生しました: {str(e)}"
            print(error_message)
            return error_message, []
    
    def update_model(self, model_name: str):
        """モデルを更新する"""
        self.generator = RAGGenerator(model_name)
        self.query_processor = RAGQueryProcessor(model_name)