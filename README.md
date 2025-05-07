# RAGエージェントアプリ

設計書の内容に基づいて質問に回答するための高度なRAG（Retrieval-Augmented Generation）エージェントアプリケーションです。

## 概要

このアプリケーションは、バッチ設計書や画面設計書などの技術文書から関連情報を検索し、ユーザーの質問に対して的確な回答を生成します。複数のエージェントが連携して動作し、質問分析、情報検索、評価、補完、回答生成などの処理を行います。

## 機能

- 自然言語での質問入力
- 設計書からの関連情報検索
- 複数エージェントによる段階的な処理
- 情報の評価と情報ギャップの自動補完
- 詳細な処理結果の可視化（オプション）

## システム構成

アプリケーションは以下のコンポーネントで構成されています：

1. **質問分析エージェント** - ユーザーの質問を分析し、検索クエリを生成
2. **検索エージェント** - ベクトルストアから関連設計書を検索
3. **情報評価エージェント** - 検索結果の関連性を評価
4. **情報補完エージェント** - 不足情報がある場合に追加検索を実行
5. **回答生成エージェント** - 収集した情報を元に最終回答を生成

## インストール方法

### 前提条件

- Python 3.9以上
- pip（Pythonパッケージマネージャ）

### 手順

1. リポジトリをクローンまたはダウンロードする

```bash
git clone <リポジトリURL>
cd sampleagent
```

2. 必要なパッケージをインストールする

```bash
pip install -r requirements.txt
```

3. 必要な環境変数を設定する

```bash
# OpenAIのAPIキーを設定
export OPENAI_API_KEY=your_api_key_here

# LangSmith（オプション）
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGSMITH_PROJECT=sampleagent
```

## 使用方法

1. アプリケーションを起動する

```bash
streamlit run app.py --server.fileWatcherType none
```

2. Webブラウザで `http://localhost:8501` にアクセスする

3. 質問入力欄に質問を入力し、Enterキーを押す

4. システムが処理を行い、回答が表示される

### 質問例

- 受注テーブルを利用している箇所を洗い出してください
- 在庫管理に関する画面の機能を教えてください
- 発送ラベル生成バッチの処理内容を説明してください
- 受注確定バッチと発送バッチの連携について教えてください

## 技術スタック

- **フレームワーク**: Streamlit, LangChain, LangGraph
- **言語モデル**: OpenAI GPT
- **埋め込みモデル**: GLuCoSE-base-ja-v2（日本語特化）
- **ベクトルデータベース**: FAISS
- **ユーザーインターフェース**: Streamlit

## プロジェクト構造

主要なファイルと役割：

- `app.py` - Streamlitアプリケーションのメインファイル
- `agent.py` - エージェントの実装とグラフ構築
- `vector_store_loader.py` - ベクトルストアの読み込みを担当
- `document_vectorizer.py` - 文書のベクトル化処理
- `requirements.txt` - 必要なPythonパッケージのリスト
- `sample/` - サンプル設計書ディレクトリ
  - `batch_design/` - バッチ処理の設計書
  - `screen_design/` - 画面の設計書

## 謝辞

このプロジェクトは以下のライブラリとツールに依存しています：
- OpenAI
- LangChain
- LangGraph
- FAISS
- Streamlit
- HuggingFace


## リンク
[DeepWiki プロジェクト](https://deepwiki.com/s-ono-123456/sampleagent)
