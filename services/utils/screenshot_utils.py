import os
import datetime
import base64
from langchain_core.messages import ToolMessage
from ..config.settings import SCREENSHOTS_DIR

def process_screenshot(message):
    """
    ツールのレスポンスからスクリーンショットを抽出して保存する
    """
    # ツールメッセージを処理
    if isinstance(message, ToolMessage):
        try:
            # ツールメッセージから画像データを抽出
            content = message.artifact
            if content is not None:
                base64_data = content[0].data
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(SCREENSHOTS_DIR, f"screenshot_{timestamp}.png")
                
                with open(screenshot_path, "wb") as img_file:
                    img_file.write(base64.b64decode(base64_data))
                print(f"ツールからスクリーンショットを保存しました: {screenshot_path}")
                return True
        except Exception as e:
            print(f"ツールレスポンスからのスクリーンショット保存に失敗しました: {e}")
    return False