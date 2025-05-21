"""
SSHポートフォワーディングを利用してlocalhost:2080→10.1.1.1:80に転送し、
Playwrightでlocalhost:2080にWebアクセスするサンプルスクリプト。
"""

# 必要な標準ライブラリと外部ライブラリをインポート
import asyncio  # 非同期処理用
from contextlib import asynccontextmanager  # 非同期コンテキストマネージャ
from playwright.async_api import async_playwright  # Playwrightの非同期API
import asyncssh  # SSHポートフォワード用

# SSH接続情報（テスト用）
SSH_HOST = 'your.ssh.server'  # SSHサーバのホスト名またはIPアドレス
SSH_PORT = 22                # SSHポート（通常22）
SSH_USER = 'your_username'    # SSHユーザー名
SSH_PASSWORD = 'your_password'  # SSHパスワード

LOCAL_PORT = 2080
REMOTE_HOST = '10.1.1.1'
REMOTE_PORT = 80

@asynccontextmanager
async def ssh_port_forward():
    """
    SSHトンネルを作成し、localhost:2080→10.1.1.1:80 へのポートフォワードを行う。
    """
    try:
        # SSH接続を確立し、ポートフォワードを設定
        conn = await asyncssh.connect(
            SSH_HOST,
            port=SSH_PORT,
            username=SSH_USER,
            password=SSH_PASSWORD,
            known_hosts=None
        )
        listener = await conn.forward_local_port('127.0.0.1', LOCAL_PORT, REMOTE_HOST, REMOTE_PORT)
        print(f"SSHポートフォワード開始: localhost:{LOCAL_PORT} → {REMOTE_HOST}:{REMOTE_PORT}")
        yield
    finally:
        # ポートフォワードとSSH接続をクローズ
        listener.close()
        await conn.wait_closed()
        print("SSHポートフォワード終了")

async def access_via_playwright():
    """
    Playwrightを使ってlocalhost:2080にアクセスし、ページタイトルを表示する。
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"http://localhost:{LOCAL_PORT}")
        title = await page.title()
        print(f"ページタイトル: {title}")
        await browser.close()

async def main():
    # SSHポートフォワードを張った状態でPlaywrightアクセス
    async with ssh_port_forward():
        await access_via_playwright()

if __name__ == "__main__":
    # asyncioでmainを実行
    asyncio.run(main())
