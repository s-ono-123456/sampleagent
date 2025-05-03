import streamlit as st
import asyncio
import os
import glob
from PIL import Image
import time
import nest_asyncio
import datetime

# 既存のPlaywright MCPツールをインポート
from services.autotest_agent import main as playwright_main

# 非同期処理を有効化
nest_asyncio.apply()

# アプリケーションのタイトル
st.title("Playwright MCPブラウザ操作ツール")

# クエリ実行時のタイムスタンプを保存するための変数をセッション状態に初期化
if "query_timestamp" not in st.session_state:
    st.session_state.query_timestamp = None

# メインコンテンツ
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# メッセージ履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# スクリーンショットを表示する関数
def display_screenshots():
    screenshot_files = glob.glob("screenshots/*.png")
    if screenshot_files:
        with st.expander("スクリーンショット", expanded=True):
            # ファイルを作成日時の昇順（古い順）でソート
            sorted_screenshots = sorted(screenshot_files, key=os.path.getmtime)
            
            # クエリ実行後に作成されたファイルのみをフィルタリング
            if st.session_state.query_timestamp:
                filtered_screenshots = [
                    f for f in sorted_screenshots 
                    if os.path.getmtime(f) > st.session_state.query_timestamp
                ]
                
                if filtered_screenshots:
                    st.write(f"クエリ実行後に作成された {len(filtered_screenshots)} 件のファイルを昇順で表示します")
                    
                    for i, screenshot_path in enumerate(filtered_screenshots):
                        # ファイル名と作成日時を表示
                        filename = os.path.basename(screenshot_path)
                        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(screenshot_path))
                        img = Image.open(screenshot_path)
                        st.image(img, caption=f"スクリーンショット {i+1}: {filename} (作成時間: {file_time.strftime('%H:%M:%S')})")
                else:
                    st.info("クエリ実行後に作成されたファイルはありません")
            else:
                st.info("クエリを実行するとスクリーンショットがここに表示されます")

# 非同期関数を実行するためのヘルパー関数
def run_async(async_func, *args, **kwargs):
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_func(*args, **kwargs))
    loop.close()
    return result

# チャットインプットの処理
query = st.text_area("ブラウザで何をしますか？")

if query:
    # クエリ実行開始時のタイムスタンプを記録
    st.session_state.query_timestamp = time.time()
    
    # ユーザーメッセージを表示
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    # AIの思考中メッセージ
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("処理中...")
        
        
        # グラフ設定
        graph_config = {"configurable": {"thread_id": str(int(time.time()))}}
        
        try:
            # 非同期関数を実行
            with st.spinner('ブラウザを操作中...'):
                # 非同期関数を同期的に実行
                run_async(playwright_main, user_input=query, graph_config=graph_config)
            
            # 正常に完了したら、新しいメッセージを追加
            result_message = "処理が完了しました。スクリーンショットを確認してください。"
            st.session_state.messages.append({"role": "assistant", "content": result_message})
            message_placeholder.write(result_message)
            
            # スクリーンショットの表示
            display_screenshots()
            
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            message_placeholder.error(error_message)
