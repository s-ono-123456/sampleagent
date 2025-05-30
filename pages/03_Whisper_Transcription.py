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
import logging  # ログ出力用

# 親ディレクトリをパスに追加して、servicesモジュールをインポートできるようにする
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# 音声文字起こしサービスのインポート
from services.whisper_transcription import WhisperTranscriptionService

# ログ設定
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'whisper_transcription.log')
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8'
)

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
    # 話者分離関連のセッション変数
    if 'enable_diarization' not in st.session_state:
        st.session_state.enable_diarization = True
    if 'speaker_count' not in st.session_state:
        st.session_state.speaker_count = 0
    if 'speaker_segments' not in st.session_state:
        st.session_state.speaker_segments = []
    # 文字起こし要約関連の状態変数
    if 'summary_result' not in st.session_state:
        st.session_state.summary_result = None
    if 'summary_status' not in st.session_state:
        st.session_state.summary_status = None
    if 'summary_processing_time' not in st.session_state:
        st.session_state.summary_processing_time = 0.0
    if 'include_speaker_info' not in st.session_state:
        st.session_state.include_speaker_info = True

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
            "vtt": "Web字幕 (VTT)",
            "text": "テキスト (TXT)",
            "srt": "字幕 (SRT)",
            "json": "構造化データ (JSON)"
        }
        
        selected_format = st.selectbox(
            "出力形式を選択",
            options=list(format_options.keys()),
            format_func=lambda x: format_options[x],
            index=0
        )
        st.session_state.output_format = selected_format
        st.subheader("話者分離設定")
        st.session_state.enable_diarization = st.toggle(
            "話者分離機能を有効にする", 
            value=True, 
            help="音声内の複数の話者を識別し、各発言に話者ラベルを付けます"
        )
        
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
            st.write(f"**話者分離**: {'有効' if st.session_state.enable_diarization else '無効'}")
            
            # 実行ボタン
            start_button = st.button("文字起こしを開始", type="primary", use_container_width=True)
            if start_button:
                with st.spinner("文字起こしを実行中..."):
                    try:
                        # 文字起こしサービスの初期化
                        service = WhisperTranscriptionService()
                        # 文字起こし実行
                        start_time = time.time()
                        result = service.transcribe(
                            audio_file=st.session_state.audio_file,
                            language=st.session_state.language,
                            format_type=st.session_state.output_format,
                            enable_diarization=st.session_state.enable_diarization
                        )
                        end_time = time.time()
                        
                        # LangGraphの結果は辞書のような形式でアクセスする
                        # 結果をセッション状態に保存
                        # 新しい形式（辞書形式でのアクセス）                    
                        st.session_state.transcription_result = result["transcript"]
                        st.session_state.process_status = result["status"]
                        st.session_state.segments = result["segments"]
                        st.session_state.processing_time = result["processing_time"]
                        st.session_state.confidence_score = result["confidence_score"]
                        st.session_state.output_file = result["output_file"]
                        
                        # 話者分離情報の保存
                        if st.session_state.enable_diarization:
                            st.session_state.speaker_count = result.get("speaker_count", 0)
                            st.session_state.speaker_segments = result.get("speaker_segments", [])
                        
                        # 文字起こし完了メッセージを表示
                        processing_time_str = format_time(result["processing_time"].get("total", 0))
                        confidence_percentage = int(result["confidence_score"] * 100)
                        st.info(f"🎉 文字起こしが完了しました！結果タブを確認してください。\n処理時間: {processing_time_str} | 信頼度: {confidence_percentage}%")
                    except Exception as e:
                        logging.error(f"文字起こし処理中にエラー: {e}", exc_info=True)
                        st.error("文字起こし処理中にエラーが発生しました。詳細は管理者にお問い合わせください。")
                    

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
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        if st.session_state.enable_diarization:
            st.metric("検出話者数", str(st.session_state.speaker_count))
    
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
            
            st.pyplot(fig)    # 文字起こし結果の表示
    st.subheader("文字起こしテキスト")
    
    with st.container(border=True):
        if st.session_state.enable_diarization and st.session_state.speaker_count > 0:
            # 話者情報を含めた表示（HTML形式で装飾）
            for segment in st.session_state.segments:
                speaker = segment.get("speaker", "不明")
                speaker_class = speaker.replace(" ", "_")  # CSSクラス名に適した形式に変換
                
                # 時間情報をフォーマット
                start_time = format_time(segment["start"]).replace("秒", "")
                
                # HTML形式でスピーカー情報と発話内容を表示
                st.markdown(f"""
                <div class="speaker-container speaker-{speaker_class}">
                    <div class="speaker-label speaker-label-{speaker_class}">【{speaker}】 {start_time}</div>
                    <div class="speaker-text">{segment['text']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # 通常の文字起こし結果
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
        
        with st.expander("セグメント情報を表示", expanded=False):            # セグメントをDataFrameに変換
            segments_data = []
            for segment in st.session_state.segments:
                segments_data.append({
                    "開始時間": segment["start"],
                    "終了時間": segment["end"],
                    "テキスト": segment["text"],
                    "話者": segment.get("speaker", "不明"),
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
        with st.expander("セグメントタイムライン", expanded=False):            # セグメントのタイムラインを視覚化
            max_time = max([segment["end"] for segment in st.session_state.segments])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 話者ごとに色を分ける
            speakers = set()
            for segment in st.session_state.segments:
                if "speaker" in segment:
                    speakers.add(segment["speaker"])
                      # 話者ごとの色を設定
            speaker_colors = {}
            # 非推奨のget_cmapの代わりにpyplot.get_cmap()を使用
            color_map = plt.get_cmap('tab10', max(10, len(speakers)))
            for i, speaker in enumerate(speakers):
                speaker_colors[speaker] = color_map(i)
            
            for i, segment in enumerate(st.session_state.segments):
                confidence = min(1.0, max(0.0, 1.0 + segment.get("probability", 0) / 10))
                
                # 話者に基づいて色を選択
                if "speaker" in segment and segment["speaker"] in speaker_colors:
                    color = speaker_colors[segment["speaker"]]
                else:
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
            ax.set_yticklabels([f"#{i} {segment.get('speaker', '')}" for i, segment in enumerate(st.session_state.segments)])
            ax.set_xlabel("時間 (秒)")
            
            # 凡例を追加
            if st.session_state.enable_diarization and speakers:
                legend_elements = [plt.Rectangle((0, 0), 1, 1, color=speaker_colors.get(speaker, 'gray'), alpha=0.7) 
                                  for speaker in speakers]
                ax.legend(legend_elements, speakers, loc="upper right", title="話者")
                ax.set_title("セグメントタイムライン (色は話者を表します)")
            else:
                ax.set_title("セグメントタイムライン (色は信頼度を表します)")
                
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            
            # 話者分析セクション
        if st.session_state.enable_diarization and st.session_state.speaker_count > 0:
            with st.expander("話者分析情報", expanded=False):
                # 各話者の発話時間を分析
                speaker_times = {}
                for segment in st.session_state.segments:
                    speaker = segment.get("speaker", "不明")
                    duration = segment["end"] - segment["start"]
                    if speaker in speaker_times:
                        speaker_times[speaker] += duration
                    else:
                        speaker_times[speaker] = duration
                
                total_duration = sum(speaker_times.values())
                
                # 話者の発話時間グラフ
                fig, ax = plt.subplots(figsize=(10, 5))
                speakers = list(speaker_times.keys())
                durations = list(speaker_times.values())
                percentages = [d/total_duration*100 for d in durations]
                
                # 色の設定
                colors = plt.cm.tab10(range(len(speakers)))
                
                # 棒グラフ
                ax.bar(speakers, durations, color=colors)
                ax.set_ylabel('発話時間 (秒)')
                ax.set_title('話者ごとの発話時間')
                
                # 各バーの上に割合を表示
                for i, (d, p) in enumerate(zip(durations, percentages)):
                    ax.annotate(f'{d:.1f}秒\n({p:.1f}%)', 
                              xy=(i, d), 
                              xytext=(0, 3),
                              textcoords="offset points", 
                              ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # 話者ごとの発話回数
                speaker_counts = {}
                for segment in st.session_state.segments:
                    speaker = segment.get("speaker", "不明")
                    if speaker in speaker_counts:
                        speaker_counts[speaker] += 1
                    else:
                        speaker_counts[speaker] = 1
                
                # 発話回数表を表示
                st.subheader("話者ごとの発話情報")
                speaker_data = []
                for speaker, count in speaker_counts.items():
                    speaker_data.append({
                        "話者": speaker,
                        "発話回数": count,
                        "合計発話時間": f"{speaker_times[speaker]:.2f}秒",
                        "発話割合": f"{speaker_times[speaker]/total_duration*100:.1f}%"
                    })
                
                st.dataframe(
                    pd.DataFrame(speaker_data),
                    hide_index=True,
                    use_container_width=True
                )

def render_summary_tab():
    """要約タブのコンテンツを表示"""
    st.header("文字起こし内容の要約")
    
    # 文字起こし結果がない場合
    if st.session_state.transcription_result is None:
        st.info("まず音声ファイルをアップロードして文字起こしを行ってください。")
        return
      # 要約のオプション
    with st.expander("要約オプション", expanded=False):
        st.write("要約の詳細度や長さを調整できます。")
        summary_length = st.select_slider(
            "要約の長さ",
            options=["短め", "標準", "詳細"],
            value="標準"
        )
        
        include_speaker_info = st.toggle(
            "話者情報を含める", 
            value=True,
            help="要約に話者（SPEAKER_01, SPEAKER_02など）の発言内容を区別して含めます"
        )
          # 要約ボタン
    if st.button("文字起こし内容を要約", type="primary", use_container_width=True):
        with st.spinner("文字起こし内容を要約中..."):
            try:
                # WhisperTranscriptionServiceのインスタンスを作成
                service = WhisperTranscriptionService()
                
                # トグルの状態をセッション状態に保存
                st.session_state.include_speaker_info = include_speaker_info
                
                # セグメント情報の準備（話者情報を含めるかどうか）
                segments_for_summary = st.session_state.segments
                if not include_speaker_info:
                    # 話者情報を含まない場合は、セグメントから話者情報を削除したコピーを作成
                    segments_for_summary = []
                    for segment in st.session_state.segments:
                        segment_copy = segment.copy()
                        if "speaker" in segment_copy:
                            del segment_copy["speaker"]
                        segments_for_summary.append(segment_copy)
                
                # サービスの要約メソッドを呼び出し
                result = service.summarize_transcription(
                    text=st.session_state.transcription_result,
                    language=st.session_state.language,
                    segments=segments_for_summary  # セグメント情報（時間情報と話者情報を含む/含まない）を渡す
                )
                
                # 結果をセッション状態に保存
                st.session_state.summary_result = result["summary"]
                st.session_state.summary_status = result["status"]
                st.session_state.summary_processing_time = result.get("processing_time", 0)
                
                # 要約完了メッセージを表示
                if result["status"] == "completed":
                    st.success(f"要約が完了しました！ 処理時間: {format_time(result.get('processing_time', 0))}")
                else:
                    st.error(f"要約中にエラーが発生しました: {result.get('error_message', '不明なエラー')}")
            except Exception as e:
                logging.error(f"要約処理中にエラー: {e}", exc_info=True)
                st.error("要約処理中にエラーが発生しました。詳細は管理者にお問い合わせください。")
    # 要約結果の表示
    if st.session_state.summary_result:
        st.subheader("要約結果")
        
        with st.container(border=True):
            st.write(st.session_state.summary_result)
        
        # 処理時間の表示
        st.caption(f"処理時間: {format_time(st.session_state.summary_processing_time)}")
        
        # 要約結果のダウンロードボタン
        # 現在の日時を取得してファイル名に含める
        from datetime import datetime
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 音声ファイル名をベースにする（session_stateにaudio_fileがある場合）
        base_filename = "summary"
        if st.session_state.audio_file:
            base_filename = os.path.splitext(os.path.basename(st.session_state.audio_file))[0] + "_summary"
        
        summary_filename = f"{base_filename}_{current_datetime}.txt"
        
        st.download_button(
            label="要約結果をダウンロード (TXT形式)",
            data=st.session_state.summary_result,
            file_name=summary_filename,
            mime="text/plain",
            key="download_summary"
        )
        
        # 元の文字起こしとの比較
        with st.expander("元の文字起こし内容と比較", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("要約")
                st.write(st.session_state.summary_result)
            
            with col2:
                st.subheader("元の文字起こし")
                st.write(st.session_state.transcription_result)

def main():
    """メイン関数"""
    # セッション状態の初期化
    init_session_state()
    
    # カスタムCSS
    st.markdown("""
    <style>
        .speaker-container {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 5px solid #4e8cff;
        }
        .speaker-label {
            font-weight: bold;
            color: #4e8cff;
            margin-bottom: 5px;
        }
        .speaker-text {
            margin-left: 10px;
        }
        .speaker-SPEAKER_00 { border-left-color: #FF4E4E; }
        .speaker-SPEAKER_01 { border-left-color: #4E8CFF; }
        .speaker-SPEAKER_02 { border-left-color: #4EFF8C; }
        .speaker-SPEAKER_03 { border-left-color: #FF8C4E; }
        .speaker-SPEAKER_04 { border-left-color: #8C4EFF; }
        .speaker-SPEAKER_05 { border-left-color: #FFFF4E; }
        .speaker-label-SPEAKER_00 { color: #FF4E4E; }
        .speaker-label-SPEAKER_01 { color: #4E8CFF; }
        .speaker-label-SPEAKER_02 { color: #4EFF8C; }
        .speaker-label-SPEAKER_03 { color: #FF8C4E; }
        .speaker-label-SPEAKER_04 { color: #8C4EFF; }
        .speaker-label-SPEAKER_05 { color: #FFFF4E; }
    </style>
    """, unsafe_allow_html=True)
    
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
    tab1, tab2, tab3 = st.tabs(["📤 音声ファイルのアップロード", "📝 文字起こし結果", "📋 文字起こし要約"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_result_tab()
    
    with tab3:
        render_summary_tab()

# メイン実行
if __name__ == "__main__":
    import torch
    main()