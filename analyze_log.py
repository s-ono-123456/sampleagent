# ログファイルを読み込み、LangChain経由でAIにエラー原因を問い合わせるPythonスクリプト
# 必ず日本語でコメントを記載

import datetime
import os
from dotenv import load_dotenv  # .envファイルから環境変数を読み込むためのモジュールをインポート

# LangChainの必要なモジュールをインポート
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# .envファイルを読み込む
load_dotenv()

# 今日の日付をYYYYMMDD形式で取得
today = datetime.datetime.now().strftime("%Y%m%d")
log_filename = f"bat_error_{today}.log"

# ログファイルが存在するか確認
if not os.path.exists(log_filename):
    print(f"ログファイルが見つかりません: {log_filename}")
    exit(1)

# ログファイルを読み込む
with open(log_filename, "r", encoding="utf-8") as f:
    log_content = f.read()

# ログ内容を出力
print("取得したログ内容:")
print(log_content)

# OpenAI APIキーを環境変数から取得
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("OPENAI_API_KEYが設定されていません。")
    exit(1)

# LangChainのChatOpenAIインスタンスを作成
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=openai_api_key,
    temperature=0.2,
)

# プロンプトを作成
system_message = SystemMessage(content="あなたは優秀なシステム運用エンジニアです。")
human_message = HumanMessage(
    content=f"""以下はバッチ処理のエラーログです。エラーの原因を日本語で分析してください。

{log_content}
"""
)

# LLMに問い合わせてエラー原因を取得
response = llm([system_message, human_message])

# AIの回答を取得
ai_answer = response.content

# 結果をログファイルに書き出す
result_filename = f"analysis_result_{today}.txt"
with open(result_filename, "w", encoding="utf-8") as f:
    f.write("エラーログ:\n")
    f.write(log_content)
    f.write("\n\nAIによる原因分析:\n")
    f.write(ai_answer)

print(f"AIによる原因分析結果を {result_filename} に保存しました。")
