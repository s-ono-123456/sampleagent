# PowerShellスクリプトでTeratermを使ってSSH接続し、ログファイルを取得し、Pythonスクリプトを呼び出す
# 必ず日本語でコメントを記載

# .envファイルからサーバ情報・認証情報を取得するためにpython-dotenvを利用
$envPath = Join-Path $PSScriptRoot ".env"
if (Test-Path $envPath) {
    # .envファイルを1行ずつ読み込んで環境変数として設定
    Get-Content $envPath | ForEach-Object {
        if ($_ -match "^([^#=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $val = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
}

# .envから値を取得
$server = $env:SERVER
$user = $env:USER
$pass = $env:PASS
$remote_log_path = $env:REMOTE_LOG_PATH
$local_log_path = $env:LOCAL_LOG_PATH
$teraterm_path = $env:TERATERM_PATH
$python_script = $env:PYTHON_SCRIPT
$web_server_ip = $env:WEB_SERVER_IP

if (-not $server -or -not $user -or -not $pass -or -not $remote_log_path -or -not $local_log_path -or -not $teraterm_path -or -not $python_script -or -not $web_server_ip) {
    Write-Host ".envファイルにSERVER, USER, PASS, REMOTE_LOG_PATH, LOCAL_LOG_PATH, TERATERM_PATH, PYTHON_SCRIPT, WEB_SERVER_IPを設定してください。"
    exit 1
}

# 今日の日付をYYYYMMDD形式で取得
$date = Get-Date -Format "yyyyMMdd"

# ログファイルのパスを日付で置換
$remote_log = $remote_log_path -replace "\{date\}", $date
$local_log = $local_log_path -replace "\{date\}", $date

# Teratermマクロファイルを作成
$macro = @"
; Teratermマクロファイル
; 1タブ目: 踏み台サーバにSSH接続し、ポートフォワーディングを維持
connect '${server}:22 /ssh /auth=password /user=${user} /passwd=${pass} /L=10022:${web_server_ip}:22'
wait '$ '

; 2タブ目: ローカル10022経由でWebサーバにSSH接続し、ログ取得
newtab
connect 'localhost:10022 /ssh /auth=password /user=${user} /passwd=${pass}'
wait '$ '
sendln 'cat ${remote_log}'
wait '$ '
sendln 'exit'
closetab

; 1タブ目に戻って切断
tab 1
sendln 'exit'
closetab
"@

$macro_path = "get_log.ttl"
Set-Content -Path $macro_path -Value $macro -Encoding UTF8

# Teratermマクロを実行し、標準出力をログファイルに保存
& "$teraterm_path" $macro_path | Out-File -Encoding UTF8 $local_log

Write-Host "ログファイルを取得しました: $local_log"

# Pythonスクリプトを実行
Write-Host "Pythonスクリプトを実行します: $python_script"
python $python_script

Write-Host "Pythonスクリプトの実行が完了しました。"
