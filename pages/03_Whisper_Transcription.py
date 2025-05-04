import os
import sys
import tempfile
import time
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import japanize_matplotlib

# 親ディレクトリをパスに追加して、servicesモジュールをインポートできるようにする
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# 音声文字起こしサービスのインポート
from services.whisper_transcription import WhisperTranscriptionService

# ページ設定
st.set_page_config(
    page_title="Faster-Whisper 音声文字起こし",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """セッション状態の初期化"""
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'transcription_result' not in st.session_state:
        st.session_state.transcription_result = None
    if 'process_status' not in st.session_state:
        st.session_state.process_status = None
    if 'segments' not in st.session_state:
        st.session_state.segments = []
    if 'output_format' not in st.session_state:
        st.session_state.output_format = "text"
    if 'language' not in st.session_state:
        st.session_state.language = ""
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = {}
    if 'confidence_score' not in st.session_state:
        st.session_state.confidence_score = 0.0
    if 'output_file' not in st.session_state:
        st.session_state.output_file = ""

def sidebar_content():
    """サイドバーのコンテンツを表示"""
    with st.sidebar:
        st.header("設定")
        
        st.subheader("モデル設定")
        model_info = st.expander("モデル情報", expanded=False)
        with model_info:
            st.write("**モデル**: Faster-Whisper large-v3-turbo")
            st.write("**概要**: OpenAIのWhisperモデルの最適化版で、最高精度を提供します。")
            st.write("**サイズ**: 約3GB")
            st.write("**サポート言語**: 99言語")
            
        st.subheader("言語設定")
        language_options = {
            "": "自動検出",
            "ja": "日本語",
            "en": "英語",
            "zh": "中国語",
            "ko": "韓国語",
            "fr": "フランス語",
            "de": "ドイツ語",
            "es": "スペイン語",
            "it": "イタリア語",
            "ru": "ロシア語"
        }
        
        selected_language = st.selectbox(
            "言語を選択（空白の場合は自動検出）",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0
        )
        st.session_state.language = selected_language
        
        st.subheader("出力形式")
        format_options = {
            "text": "テキスト (TXT)",
            "srt": "字幕 (SRT)",
            "vtt": "Web字幕 (VTT)",
            "json": "構造化データ (JSON)"
        }
        
        selected_format = st.selectbox(
            "出力形式を選択",
            options=list(format_options.keys()),
            format_func=lambda x: format_options[x],
            index=0
        )
        st.session_state.output_format = selected_format
        
        st.subheader("詳細設定")
        with st.expander("詳細設定", expanded=False):
            st.write("**動作環境**:")
            use_gpu = torch.cuda.is_available()
            st.write(f"GPU: {'✅ 利用可能' if use_gpu else '❌ 利用不可'}")
            st.write(f"デバイス: {'CUDA' if use_gpu else 'CPU'}")
            
            if use_gpu:
                st.write(f"GPU名: {torch.cuda.get_device_name(0)}")
            
            st.divider()
            st.write("**ファイル制限**:")
            st.write("最大サイズ: 500MB")
            st.write("対応形式: WAV, MP3, FLAC, OGG")
        
        # 著者情報
        st.sidebar.divider()
        st.sidebar.caption("Faster-Whisper Transcription Service")
        st.sidebar.caption("Developed with ❤️ using Streamlit & Faster-Whisper")

def render_upload_tab():
    """アップロードタブのコンテンツを表示"""
    st.header("音声ファイルのアップロード")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "音声ファイルをアップロード (WAV, MP3, FLAC, M4A, OGG)",
        type=["wav", "mp3", "flac", "m4a", "ogg"]
    )
    
    if uploaded_file is not None:
        # 一時ファイルとして保存
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.audio_file = temp_path
        
        # ファイル情報の表示
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # MBに変換
        
        st.success(f"ファイルが正常にアップロードされました: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ファイル情報")
            st.write(f"**ファイル名**: {uploaded_file.name}")
            st.write(f"**ファイルサイズ**: {file_size:.2f} MB")
            st.write(f"**ファイル形式**: {os.path.splitext(uploaded_file.name)[1].upper()[1:]}")
            
            # 音声プレイヤーの表示
            st.audio(uploaded_file, format=f"audio/{os.path.splitext(uploaded_file.name)[1][1:]}")
        
        with col2:
            st.subheader("トランスクリプション設定")
            st.write(f"**言語**: {st.session_state.language if st.session_state.language else '自動検出'}")
            st.write(f"**出力形式**: {st.session_state.output_format.upper()}")
            
            # 実行ボタン
            start_button = st.button("文字起こしを開始", type="primary", use_container_width=True)
            
            if start_button:
                with st.spinner("文字起こしを実行中..."):
                    # 文字起こしサービスの初期化
                    service = WhisperTranscriptionService()
                    
                    # 文字起こし実行
                    start_time = time.time()
                    result = service.transcribe(
                        audio_file=st.session_state.audio_file,
                        language=st.session_state.language,
                        format_type=st.session_state.output_format
                    )
                    end_time = time.time()
                    
                    # デバッグ情報を表示（開発中のみ）
                    st.write(f"結果の型: {type(result)}")
                    
                    # LangGraphの結果は辞書のような形式でアクセスする
                    # 結果をセッション状態に保存
                    # 新しい形式（辞書形式でのアクセス）
                    st.session_state.transcription_result = result["transcript"]
                    st.session_state.process_status = result["status"]
                    st.session_state.segments = result["segments"]
                    st.session_state.processing_time = result["processing_time"]
                    st.session_state.confidence_score = result["confidence_score"]
                    st.session_state.output_file = result["output_file"]
                    
                    # 結果表示タブに切り替え
                    st.rerun()

def format_time(seconds):
    """秒数を時間形式にフォーマット"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds = seconds % 60
        return f"{minutes}分 {seconds:.2f}秒"
    else:
        hours = int(seconds / 3600)
        seconds %= 3600
        minutes = int(seconds / 60)
        seconds %= 60
        return f"{hours}時間 {minutes}分 {seconds:.2f}秒"

def get_confidence_class(confidence):
    """信頼度に基づいたCSSクラスを返す"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-med"
    else:
        return "confidence-low"

def render_result_tab():
    """結果タブのコンテンツを表示"""
    if st.session_state.transcription_result is None:
        st.info("音声ファイルをアップロードして文字起こしを開始してください。")
        return
    
    st.header("文字起こし結果")
    
    # 処理ステータスの表示
    if st.session_state.process_status == "completed":
        st.success("文字起こしが正常に完了しました")
    elif st.session_state.process_status == "error":
        st.error("文字起こし処理中にエラーが発生しました")
    
    # 処理時間と信頼度のサマリー
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("総処理時間", format_time(st.session_state.processing_time.get("total", 0)))
    
    with col2:
        confidence_percentage = int(st.session_state.confidence_score * 100)
        st.metric("信頼度スコア", f"{confidence_percentage}%")
    
    with col3:
        if st.session_state.language:
            language_name = {
                "ja": "日本語",
                "en": "英語",
                "zh": "中国語",
                "ko": "韓国語",
                "fr": "フランス語",
                "de": "ドイツ語",
                "es": "スペイン語",
                "it": "イタリア語",
                "ru": "ロシア語"
            }.get(st.session_state.language, st.session_state.language)
        else:
            language_name = "自動検出"
        st.metric("検出言語", language_name)
    
    # 処理時間の詳細グラフ
    if st.session_state.processing_time:
        with st.expander("処理時間の詳細", expanded=False):
            processing_time = {k: v for k, v in st.session_state.processing_time.items() if k != "total"}
            
            fig, ax = plt.subplots(figsize=(10, 5))
            steps = list(processing_time.keys())
            times = list(processing_time.values())
            
            ax.barh(steps, times, color='green')
            ax.set_xlabel('処理時間 (秒)')
            ax.set_title('処理ステップごとの実行時間')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
    
    # 文字起こし結果の表示
    st.subheader("文字起こしテキスト")
    
    with st.container(border=True):
        st.write(st.session_state.transcription_result)
    
    # ダウンロードボタン
    if st.session_state.output_file and os.path.exists(st.session_state.output_file):
        with open(st.session_state.output_file, "rb") as file:
            file_extension = os.path.splitext(st.session_state.output_file)[1]
            file_name = os.path.basename(st.session_state.output_file)
            
            st.download_button(
                label=f"結果をダウンロード ({file_extension.upper()[1:]}形式)",
                data=file,
                file_name=file_name,
                mime=f"text/{file_extension[1:]}"
            )
    
    # セグメント詳細の表示
    if st.session_state.segments:
        st.subheader("セグメント詳細")
        
        with st.expander("セグメント情報を表示", expanded=False):
            # セグメントをDataFrameに変換
            segments_data = []
            for segment in st.session_state.segments:
                segments_data.append({
                    "開始時間": segment["start"],
                    "終了時間": segment["end"],
                    "テキスト": segment["text"],
                    "信頼度": segment.get("probability", 0)
                })
            
            df = pd.DataFrame(segments_data)
            
            # 時間を読みやすい形式に変換
            df["開始時間"] = df["開始時間"].apply(lambda x: f"{int(x/60):02d}:{int(x%60):02d}.{int((x%1)*1000):03d}")
            df["終了時間"] = df["終了時間"].apply(lambda x: f"{int(x/60):02d}:{int(x%60):02d}.{int((x%1)*1000):03d}")
            
            # 信頼度を百分率に変換
            df["信頼度"] = df["信頼度"].apply(lambda x: f"{min(100, max(0, int((1.0 + x/10)*100)))}%")
            
            st.dataframe(
                df,
                hide_index=True,
                column_config={
                    "テキスト": st.column_config.TextColumn(width="large")
                },
                height=400
            )
        
        # タイムラインビュー
        with st.expander("セグメントタイムライン", expanded=False):
            # セグメントのタイムラインを視覚化
            max_time = max([segment["end"] for segment in st.session_state.segments])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i, segment in enumerate(st.session_state.segments):
                confidence = min(1.0, max(0.0, 1.0 + segment.get("probability", 0) / 10))
                color = plt.cm.RdYlGn(confidence)
                
                ax.barh(
                    i, 
                    segment["end"] - segment["start"], 
                    left=segment["start"], 
                    color=color, 
                    alpha=0.7,
                    height=0.5
                )
            
            ax.set_yticks(range(len(st.session_state.segments)))
            ax.set_yticklabels([f"#{i}" for i in range(len(st.session_state.segments))])
            ax.set_xlabel("時間 (秒)")
            ax.set_title("セグメントタイムライン (色は信頼度を表します)")
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)

def main():
    """メイン関数"""
    # セッション状態の初期化
    init_session_state()
    
    # サイドバーのコンテンツ表示
    sidebar_content()
    
    # タイトルと説明
    st.title("🎤 Faster-Whisper 音声文字起こしサービス")
    st.write("""
    高精度な音声文字起こしサービスです。音声ファイルをアップロードすると、
    Faster-Whisperモデル(large-v3-turbo)を使用して文字起こしを行います。
    複数の言語と出力形式をサポートしています。
    """)
    
    # タブの作成
    tab1, tab2 = st.tabs(["📤 音声ファイルのアップロード", "📝 文字起こし結果"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_result_tab()

# メイン実行
if __name__ == "__main__":
    import torch
    main()