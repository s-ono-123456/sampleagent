import os
import tempfile
import time
from typing import Dict, List, Optional, Any, Union
import numpy as np
from pathlib import Path
import soundfile as sf  # 音声ファイル読み書き用
import torch
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field
import traceback
from pyannote.audio import Pipeline  # 話者識別用ライブラリ
from pyannote.audio.pipelines.utils.hook import ProgressHook

# LangGraph用のインポート
from langgraph.graph import StateGraph, END
# LangChain用のインポート
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import StrOutputParser

MODEL_NAME = "large-v3-turbo"

# 要約用プロンプトテンプレート定数
SUMMARY_PROMPT_JA = """
あなたは音声文字起こしの内容から議事録を作成する専門家です。
与えられたテキストを以下のルールに則って議事録を作成してください ：

# 入力テキストについて
テキストには以下の情報が含まれています。
・時系列に沿った内容の流れ（時間情報が含まれている場合）
・話者ごとの主な発言や貢献（話者情報が含まれている場合）

時間情報が提供されている場合は、それを活用して内容の流れや議論の進行を時系列で示してください。
重要な転換点や話題の変化があった時間帯を要約に含めると理解が深まります。

話者情報が提供されている場合は、各話者の識別子（SPEAKER_00、SPEAKER_01など）をもとに文脈から氏名を特定して、主要な発言内容や立場、意見を区別して要約してください。


# 注意事項
・文字起こしデータはAIによるもので、一部の書き起こしミスが含まれています。この点を考慮して、文脈を理解し、内容を整理してください
・会議の基本情報（日時、場所、出席者など）を最初に記載してください。
・会議での主要な「決定事項」を冒頭でまとめてください。
・次に、「次回への持ち帰り事項」をまとめてください。
・その後、各議題の見出しを設け、議題ごとに、議論の流れがわかるように会話形式で主要な発言を記載してください。
・文書は簡潔かつ明瞭に記述してください。
・専門用語や略語を使用する場合は、初回の使用時に定義を明記してください。
・文脈として意味が不明な箇所は、文脈的に相応しいと合理的に推測される内容に修正、または削除してください。 
・日本語で、簡潔かつ論理的に構成し、原文の意図を保持してください。
・以下の議事録サンプルに従って文章を構成してください。

# 議事録のサンプル
```
# ○○会議 議事録

## 会議基本情報

- **日時**: 2025/01/01 14:00-15:00
- **場所**: 1375会議室
- **出席者**: 鈴木部長、井上GM、佐藤、木村

---

## 決定事項

- ○○を4/1から実施する方針に決定

## 次回への持ち帰り事項

- ○○を佐藤が検討し、1/12までに鈴木部長に報告する。
- ○○を木村が確認し、1/11までに井上GMに報告する。

---

# 議事内容
## 1. 新規参画者の紹介
- 新規参画者を紹介します。木村です。（佐藤）
- 木村です。よろしくお願いいたします。（木村）

## 2.  議題１について
- 議題１について説明します。井上さんよろしくお願いいたします。（佐藤）
- 説明をいたします。（井上）


```
"""

SUMMARY_PROMPT_EN = """
You are an expert in summarizing transcribed audio content.
Please summarize the given text as follows:

1. Main topics and discussion points
2. Important facts and information
3. Decisions and next steps (if mentioned)
4. Chronological flow of the content (if time information is provided)
5. Key statements or contributions by each speaker (if speaker information is provided)

If time information is provided, use it to show the flow of content and progression of discussion chronologically.
Including key turning points or topic changes with their corresponding timestamps will enhance understanding.

If speaker information is provided, please distinguish between the different speakers (SPEAKER_00, SPEAKER_01, etc.)
in your summary, highlighting their main points or positions. For example, "SPEAKER_01 argued that..." to clearly
attribute statements to specific speakers.

The summary should be concise and logically structured, retaining the intent of the original text.
Combine bullet points and short paragraphs to make it readable.
"""

# 状態定義
class WhisperTranscriptionState(BaseModel):
    """Whisper文字起こしサービスの状態を保持するモデル"""
    audio_file: str = Field(default="", description="処理対象の音声ファイルパス")
    audio_data: Any = Field(default=None, description="前処理済み音声データ")
    language: str = Field(default="", description="検出または指定された言語コード")
    segments: List[Dict[str, Any]] = Field(default_factory=list, description="生成されたセグメント情報のリスト")
    transcript: str = Field(default="", description="生成された完全なトランスクリプトテキスト")
    confidence_score: float = Field(default=0.0, description="文字起こし全体の信頼度スコア")
    format_type: str = Field(default="text", description="出力フォーマットタイプ（text, srt, vtt, json等）")
    output_file: str = Field(default="", description="生成された出力ファイルのパス")
    status: str = Field(default="initialized", description="処理状態（processing, completed, error等）")
    error_message: str = Field(default="", description="エラー発生時のメッセージ")
    model: Any = Field(default=None, description="ロードされたWhisperモデル")
    processing_time: Dict[str, float] = Field(default_factory=dict, description="各処理ステップの所要時間")
    # 話者識別関連のフィールド
    enable_diarization: bool = Field(default=False, description="話者識別機能を有効にするかどうか")
    diarization_model: Any = Field(default=None, description="ロードされた話者識別モデル")
    speaker_count: int = Field(default=0, description="検出された話者の数")
    speaker_segments: List[Dict[str, Any]] = Field(default_factory=list, description="話者識別されたセグメント情報")
    diarization_result: Any = Field(default=None, description="話者識別の生の結果オブジェクト")

# ノード関数の実装
def validate_file(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """音声ファイルの形式と内容を検証するノード"""
    start_time = time.time()
    
    try:
        # ファイルパスのチェック
        if not state.audio_file or not os.path.exists(state.audio_file):
            raise ValueError(f"ファイルが存在しません: {state.audio_file}")
        
        # 拡張子のチェック
        supported_formats = ['.wav', '.mp3', '.flac', '.ogg']
        file_ext = os.path.splitext(state.audio_file)[1].lower()
        
        if file_ext not in supported_formats:
            raise ValueError(f"サポートされていないファイル形式です: {file_ext}。サポート形式: {', '.join(supported_formats)}")
        
        # ファイルサイズのチェック
        file_size = os.path.getsize(state.audio_file) / (1024 * 1024)  # MBに変換
        if file_size > 500:  # 500MB以上はエラー
            raise ValueError(f"ファイルサイズが大きすぎます: {file_size:.2f}MB (最大500MB)")
        
        # soundfileを使って音声メタデータを抽出
        try:
            # ファイルが開けるか確認するために短いセグメントだけ読み込む
            with sf.SoundFile(state.audio_file) as f:
                if f.frames == 0 or f.samplerate == 0:
                    raise ValueError("音声データの読み込みに失敗しました")
            
            # 状態を更新
            state.status = "file_validated"
        except Exception as e:
            raise ValueError(f"音声ファイルの解析に失敗しました。ファイルが破損している可能性があります。: {str(e)}")
    
    except Exception as e:
        state.status = "error"
        state.error_message = f"ファイル検証エラー: {str(e)}"
    
    # 処理時間を記録
    state.processing_time["validate_file"] = time.time() - start_time
    
    return state

def preprocess_audio(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """音声の前処理を行うノード"""
    start_time = time.time()
    
    try:
        if state.status == "error":
            return state
        
        # soundfileを使用して音声をロード
        audio_data, sample_rate = sf.read(state.audio_file, dtype='float32')
        
        # ステレオをモノラルに変換
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # サンプルレートを16kHzに変換
        if sample_rate != 16000:
            # scipy.signalを使用してリサンプリング
            from scipy import signal
            audio_length = len(audio_data)
            new_length = int(audio_length * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, new_length)
        
        # 音量正規化
        max_amp = np.max(np.abs(audio_data))
        if max_amp < 0.1 and max_amp > 0:
            audio_data = audio_data / max_amp * 0.9
        
        # 状態を更新
        state.audio_data = audio_data
        state.status = "audio_preprocessed"
            
    except Exception as e:
        state.status = "error"
        state.error_message = f"音声前処理エラー: {str(e)}"
        traceback.print_exc()  # デバッグ用にスタックトレースを出力
    
    # 処理時間を記録
    state.processing_time["preprocess_audio"] = time.time() - start_time
    
    return state

def load_model(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """Faster-Whisperモデルをロードするノード"""
    start_time = time.time()
    
    try:
        if state.status == "error":
            return state
        
        # GPUが利用可能かチェック
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        # モデルをロード
        model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
        
        # 状態を更新
        state.model = model
        state.status = "model_loaded"
        
    except Exception as e:
        state.status = "error"
        state.error_message = f"モデルロードエラー: {str(e)}"
    
    # 処理時間を記録
    state.processing_time["load_model"] = time.time() - start_time
    
    return state

def detect_language(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """音声から言語を検出するノード"""
    start_time = time.time()
    
    try:
        if state.status == "error" or not state.model:
            return state
        
        # 言語が既に指定されている場合はスキップ
        if state.language:
            state.status = "language_detected"
            return state
        
        # 最初の30秒のみを使用して言語検出
        audio_segment = state.audio_data[:int(16000 * 30)] if len(state.audio_data) > 16000 * 30 else state.audio_data
        
        # 言語検出の実行
        segments, info = state.model.transcribe(audio_segment, task="transcribe")
        detected_language = info.language
        
        # 状態を更新
        state.language = detected_language
        state.status = "language_detected"
        
    except Exception as e:
        state.status = "error"
        state.error_message = f"言語検出エラー: {str(e)}"
        traceback.print_exc()
    
    # 処理時間を記録
    state.processing_time["detect_language"] = time.time() - start_time
    
    return state

def transcribe_audio(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """音声の文字起こしを実行するノード"""
    start_time = time.time()
    
    try:
        if state.status == "error" or not state.model:
            return state
        
        # 文字起こしパラメータの設定
        beam_size = 5
        
        # 文字起こしの実行
        segments, info = state.model.transcribe(
            state.audio_data, 
            language=state.language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=True
        )
        
        # セグメント情報の取得
        segments_list = []
        full_text = ""
        
        for segment in segments:
            segment_dict = {
                "id": len(segments_list),
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} 
                          for word in segment.words],
                "probability": segment.avg_logprob
            }
            segments_list.append(segment_dict)
            full_text += segment.text + " "
        
        # 信頼度スコアの計算（平均対数確率を0-1の範囲にマッピング）
        if segments_list:
            avg_confidence = sum(segment["probability"] for segment in segments_list) / len(segments_list)
            # 対数確率を0-1の範囲に変換（-10を0、0を1とする近似的なマッピング）
            confidence_score = min(1.0, max(0.0, 1.0 + avg_confidence / 10))
        else:
            confidence_score = 0.0
        
        # 状態を更新
        state.segments = segments_list
        state.transcript = full_text.strip()
        state.confidence_score = confidence_score
        state.status = "transcribed"
        
    except Exception as e:
        state.status = "error"
        state.error_message = f"文字起こしエラー: {str(e)}"
    
    # 処理時間を記録
    state.processing_time["transcribe_audio"] = time.time() - start_time
    
    return state

def identify_speakers(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """音声から話者を識別（ダイアリゼーション）するノード"""
    start_time = time.time()
    
    try:
        if state.status == "error" or not state.enable_diarization:
            # 話者識別が有効でない場合はスキップ
            return state
        
        # 一時ファイルの作成（PyAnnoteはファイルパスが必要なため）
        temp_dir = tempfile.gettempdir()
        temp_wav_path = os.path.join(temp_dir, "temp_diarization.wav")
        
        # AudioSegmentを使用して音声を16kHzのWAVに変換
        audio = AudioSegment.from_file(state.audio_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_wav_path, format="wav")
        
        # 話者識別モデルの読み込み
        if state.diarization_model is None:
            try:
                # Hugging Faceからモデルを取得（アクセストークンが必要な場合があります）
                state.diarization_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.environ.get("HF_TOKEN"),  # 環境変数からトークンを取得
                )
                
                # GPUが利用可能な場合はGPUを使用
                if torch.cuda.is_available():
                    state.diarization_model = state.diarization_model.to(torch.device("cuda"))
                    
            except Exception as e:
                raise ValueError(f"話者識別モデルの読み込みに失敗しました: {str(e)}")
        
        # 話者識別の実行
        with ProgressHook() as hook:
            diarization_result = state.diarization_model(temp_wav_path, hook=hook)
        
        # 話者識別結果の解析
        speaker_segments = []
        speaker_set = set()
        
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            speaker_set.add(speaker)
            segment = {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            }
            speaker_segments.append(segment)
        
        # セグメントに話者情報を追加
        for whisper_segment in state.segments:
            # 各Whisperセグメントの中央時間を計算
            mid_time = (whisper_segment["start"] + whisper_segment["end"]) / 2
            
            # その時間に対応する話者を探す
            matching_speaker = None
            for speaker_segment in speaker_segments:
                if speaker_segment["start"] <= mid_time <= speaker_segment["end"]:
                    matching_speaker = speaker_segment["speaker"]
                    break
            
            # 話者情報をセグメントに追加
            whisper_segment["speaker"] = matching_speaker if matching_speaker else "unknown"
        
        # 状態を更新
        state.speaker_count = len(speaker_set)
        state.speaker_segments = speaker_segments
        state.diarization_result = diarization_result
        state.status = "speakers_identified"
        
        # 一時ファイルを削除
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            
    except Exception as e:
        state.status = "error"
        state.error_message = f"話者識別エラー: {str(e)}"
        traceback.print_exc()
    
    # 処理時間を記録
    state.processing_time["identify_speakers"] = time.time() - start_time
    
    return state

def postprocess_text(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """文字起こし結果のテキスト後処理を行うノード"""
    start_time = time.time()
    
    try:
        if state.status == "error" or not state.transcript:
            return state
        
        # 言語に基づいた後処理ルールの適用
        processed_text = state.transcript
        
        # 日本語の場合の後処理
        if state.language == "ja":
            # スペースの削除（日本語ではほとんどの場合不要）。
            processed_text = processed_text.replace(" ", "")
            # 句読点の調整
            processed_text = processed_text.replace("。。", "。")
            processed_text = processed_text.replace("、、", "、")
            
        # 英語の場合の後処理
        elif state.language in ["en", "english"]:
            # 単語間のスペースを確保
            processed_text = processed_text.replace("  ", " ")
            # 句読点の後にスペースを確保
            processed_text = processed_text.replace(".", ". ").replace(", ", ", ")
            processed_text = processed_text.replace("  ", " ")
        
        # 共通の後処理
        # 連続する句読点の整理
        processed_text = processed_text.replace("!!", "!").replace("??", "?")
        
        # 不要な文字の削除（制御文字など）
        processed_text = ''.join(char for char in processed_text if ord(char) >= 32)
        
        # セグメントごとにも後処理を適用
        for segment in state.segments:
            text = segment["text"]
            
            # 言語に基づいた後処理
            if state.language == "ja":
                text = text.replace(" ", "")
                text = text.replace("。。", "。")
                text = text.replace("、、", "、")
            elif state.language in ["en", "english"]:
                text = text.replace("  ", " ")
                text = text.replace(".", ". ").replace(", ", ", ")
                text = text.replace("  ", " ")
            
            # 共通の後処理
            text = text.replace("!!", "!").replace("??", "?")
            text = ''.join(char for char in text if ord(char) >= 32)
            
            segment["text"] = text
        
        # 状態を更新
        state.transcript = processed_text
        state.status = "postprocessed"
        
    except Exception as e:
        state.status = "error"
        state.error_message = f"テキスト後処理エラー: {str(e)}"
    
    # 処理時間を記録
    state.processing_time["postprocess_text"] = time.time() - start_time
    
    return state

def format_output(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """文字起こし結果を要求された形式に変換するノード"""
    start_time = time.time()
    
    try:
        if state.status == "error" or not state.transcript:
            return state
        
        # 出力ディレクトリの作成
        output_dir = os.path.join(tempfile.gettempdir(), "whisper_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # 元のファイル名をベースに出力ファイル名を生成
        base_filename = os.path.splitext(os.path.basename(state.audio_file))[0]
        
        # 形式に応じた出力ファイルの生成
        if state.format_type == "text":
            output_file = os.path.join(output_dir, f"{base_filename}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                # 話者識別が有効な場合は話者情報を含める
                if state.enable_diarization and state.speaker_count > 0:
                    for segment in state.segments:
                        speaker = segment.get("speaker", "不明")
                        f.write(f"【{speaker}】: {segment['text']}\n")
                else:
                    f.write(state.transcript)
                
        elif state.format_type == "srt":
            output_file = os.path.join(output_dir, f"{base_filename}.srt")
            with open(output_file, "w", encoding="utf-8") as f:
                for i, segment in enumerate(state.segments, 1):
                    # SRT形式: インデックス、タイムスタンプ、テキスト、空行
                    start_time_str = format_timestamp(segment["start"], msec=True)
                    end_time_str = format_timestamp(segment["end"], msec=True)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    
                    # 話者識別が有効な場合は話者情報を含める
                    if state.enable_diarization and "speaker" in segment:
                        f.write(f"【{segment['speaker']}】: {segment['text']}\n\n")
                    else:
                        f.write(f"{segment['text']}\n\n")
        
        elif state.format_type == "vtt":
            output_file = os.path.join(output_dir, f"{base_filename}.vtt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for i, segment in enumerate(state.segments):
                    # VTT形式: タイムスタンプ、テキスト、空行
                    start_time_str = format_timestamp(segment["start"], msec=True)
                    end_time_str = format_timestamp(segment["end"], msec=True)
                    
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    
                    # 話者識別が有効な場合は話者情報を含める
                    if state.enable_diarization and "speaker" in segment:
                        f.write(f"【{segment['speaker']}】: {segment['text']}\n\n")
                    else:
                        f.write(f"{segment['text']}\n\n")
        
        elif state.format_type == "json":
            import json
            output_file = os.path.join(output_dir, f"{base_filename}.json")
            
            result = {
                "transcript": state.transcript,
                "language": state.language,
                "segments": state.segments,
                "confidence_score": state.confidence_score
            }
            
            # 話者識別が有効な場合は話者情報を含める
            if state.enable_diarization:
                result["speaker_count"] = state.speaker_count
                # 整形された話者情報を追加
                formatted_speaker_segments = []
                for speaker_segment in state.speaker_segments:
                    formatted_speaker_segments.append({
                        "start": speaker_segment["start"],
                        "end": speaker_segment["end"],
                        "speaker": speaker_segment["speaker"]
                    })
                result["speaker_segments"] = formatted_speaker_segments
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        else:
            # デフォルトはテキスト
            output_file = os.path.join(output_dir, f"{base_filename}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(state.transcript)
        
        # 状態を更新
        state.output_file = output_file
        state.status = "formatted"
        
    except Exception as e:
        state.status = "error"
        state.error_message = f"出力フォーマット変換エラー: {str(e)}"
    
    # 処理時間を記録
    state.processing_time["format_output"] = time.time() - start_time
    
    return state

def evaluate_result(state: WhisperTranscriptionState) -> Dict[str, str]:
    """文字起こし結果の品質を評価し、次のノードを決定するノード"""
    start_time = time.time()
    
    try:
        if state.status == "error":
            return {"next": "finalize"}
        
        # 信頼度スコアの評価
        quality_threshold = 0.6  # 信頼度の閾値
        if state.confidence_score < quality_threshold:
            # 低品質の場合は警告を記録
            state.error_message = f"警告: 文字起こし品質が低い可能性があります（信頼度: {state.confidence_score:.2f}）"
        
        # 結果の評価
        if not state.transcript or len(state.transcript) == 0:
            state.status = "error"
            state.error_message = "エラー: 文字起こしテキストが生成されませんでした"
            return {"next": "finalize"}
        
        # セグメントの一貫性チェック
        if not state.segments or len(state.segments) == 0:
            state.status = "error"
            state.error_message = "エラー: セグメント情報が生成されませんでした"
            return {"next": "finalize"}
        
        # 成功した場合
        state.status = "evaluated"
        
    except Exception as e:
        state.status = "error"
        state.error_message = f"結果評価エラー: {str(e)}"
    
    # 処理時間を記録
    state.processing_time["evaluate_result"] = time.time() - start_time
    
    return {"next": "finalize"}

def finalize(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """処理の完了とリソース解放を行うノード"""
    start_time = time.time()
    
    try:
        # モデルリソースの解放（現在のFaster-Whisperモデルは明示的な解放は不要）
        state.model = None
        
        # 処理完了ステータスの設定
        if state.status != "error":
            state.status = "completed"
            
        # 総処理時間の計算
        total_time = sum(state.processing_time.values())
        state.processing_time["total"] = total_time
        
    except Exception as e:
        if state.status != "error":
            state.status = "error"
        state.error_message += f" 完了処理エラー: {str(e)}"
    
    # 処理時間を記録
    state.processing_time["finalize"] = time.time() - start_time
    
    return state

# ユーティリティ関数
def format_timestamp(seconds, msec=False):
    """秒数をタイムスタンプ形式に変換"""
    hours = int(seconds / 3600)
    seconds %= 3600
    minutes = int(seconds / 60)
    seconds %= 60
    
    if msec:
        mseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{mseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:.2f}"

# グラフの構築
def build_transcription_graph() -> StateGraph:
    """Whisper文字起こし処理のグラフを構築"""
    # 初期状態の設定
    workflow = StateGraph(WhisperTranscriptionState)
    
    # ノードの追加
    workflow.add_node("validate_file", validate_file)
    workflow.add_node("preprocess_audio", preprocess_audio)
    workflow.add_node("load_model", load_model)
    workflow.add_node("detect_language", detect_language)
    workflow.add_node("transcribe_audio", transcribe_audio)
    workflow.add_node("identify_speakers", identify_speakers)
    workflow.add_node("postprocess_text", postprocess_text)
    workflow.add_node("format_output", format_output)
    workflow.add_node("evaluate_result", evaluate_result)
    workflow.add_node("finalize", finalize)
    
    # エッジの設定（処理フロー）
    workflow.set_entry_point("validate_file")
    workflow.add_edge("validate_file", "preprocess_audio")
    workflow.add_edge("preprocess_audio", "load_model")
    workflow.add_edge("load_model", "detect_language")
    workflow.add_edge("detect_language", "transcribe_audio")
    workflow.add_edge("transcribe_audio", "identify_speakers")
    workflow.add_edge("identify_speakers", "postprocess_text")
    workflow.add_edge("postprocess_text", "format_output")
    workflow.add_edge("format_output", "evaluate_result")
    
    # evaluate_resultノードから条件に基づいて分岐
    # 現状、finalizeノードに直接遷移するように設定
    workflow.add_edge("evaluate_result", "finalize")
    
    # 最終ノードの設定
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# トランスクリプションを実行するメインクラス
class WhisperTranscriptionService:
    def __init__(self):
        self.graph = build_transcription_graph()
        
    def transcribe(self, 
                  audio_file: str, 
                  language: str = "",
                  format_type: str = "text",
                  enable_diarization: bool = True) -> WhisperTranscriptionState:
        """
        音声ファイルの文字起こしを実行する
        
        Args:
            audio_file: 音声ファイルのパス
            language: 言語コード（指定しない場合は自動検出）
            format_type: 出力形式（text, srt, vtt, json）
            enable_diarization: 話者識別機能を有効にするかどうか
            
        Returns:
            文字起こし処理の結果
        """        
        # 入力状態の準備
        input_state = WhisperTranscriptionState(
            audio_file=audio_file,
            language=language,
            format_type=format_type,
            enable_diarization=enable_diarization
        )
        # グラフの実行
        result = self.graph.invoke(input_state)
        
        return result
    def summarize_transcription(self, text: str, language: str = "ja", segments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        文字起こし結果を要約する
        
        Args:
            text: 要約する文字起こしテキスト
            language: 言語コード（デフォルトは日本語）
            segments: 文字起こしのセグメント情報（時間情報と話者情報を含む）
            
        Returns:
            要約結果を含む辞書
        """
        import time
        
        start_time = time.time()
        
        try:
            # タイムスタンプ情報と話者情報を含むプロンプトを作成
            time_info_text = ""
            speaker_info_text = ""
            
            # 話者情報が存在するかチェック
            has_speaker_info = False
            if segments and len(segments) > 0:
                for segment in segments:
                    if "speaker" in segment and segment["speaker"]:
                        has_speaker_info = True
                        break
            
            if segments and len(segments) > 0:
                # セグメント情報から時間情報を抽出
                time_info_text = "\n\n【時間情報】\n"
                # 重要な変化点を検出するため、一定間隔でセグメントを選択
                step = max(1, len(segments) // 10)  # 最大10個程度の時間情報を含める
                for i in range(0, len(segments), step):
                    segment = segments[i]
                    start_time_str = format_timestamp(segment["start"], msec=False)
                    
                    # 話者情報を含める場合
                    if has_speaker_info and "speaker" in segment and segment["speaker"]:
                        time_info_text += f"{start_time_str} - 【{segment['speaker']}】 {segment['text']}\n"
                    else:
                        time_info_text += f"{start_time_str} - {segment['text']}\n"
                
                # 最後のセグメントも追加
                if len(segments) > i + 1:
                    last_segment = segments[-1]
                    start_time_str = format_timestamp(last_segment["start"], msec=False)
                    
                    # 話者情報を含める場合
                    if has_speaker_info and "speaker" in last_segment and last_segment["speaker"]:
                        time_info_text += f"{start_time_str} - 【{last_segment['speaker']}】 {last_segment['text']}\n"
                    else:
                        time_info_text += f"{start_time_str} - {last_segment['text']}\n"
                
                # 話者情報のサマリーを追加
                if has_speaker_info:
                    # 各話者の発言数と重要な発言を集計
                    speaker_counts = {}
                    speaker_samples = {}
                    
                    for segment in segments:
                        if "speaker" in segment and segment["speaker"]:
                            speaker = segment["speaker"]
                            if speaker in speaker_counts:
                                speaker_counts[speaker] += 1
                                # 各話者につき3つまでのサンプル発言を保存
                                if len(speaker_samples[speaker]) < 3:
                                    speaker_samples[speaker].append(segment["text"])
                            else:
                                speaker_counts[speaker] = 1
                                speaker_samples[speaker] = [segment["text"]]
                    
                    # 話者情報のサマリーを構築
                    speaker_info_text = "\n\n【話者情報】\n"
                    for speaker, count in speaker_counts.items():
                        speaker_info_text += f"{speaker}: {count}回の発言\n"
                        speaker_info_text += f"サンプル発言:\n"
                        for sample in speaker_samples[speaker]:
                            speaker_info_text += f"- {sample}\n"
                        speaker_info_text += "\n"
            
            # 言語に応じたプロンプトの調整
            if language == "ja":
                system_prompt = SUMMARY_PROMPT_JA
                user_prompt = f"以下の文字起こし内容を要約してください:\n\n{text}{time_info_text}{speaker_info_text}"
            elif language == "en":
                system_prompt = SUMMARY_PROMPT_EN
                user_prompt = f"Please summarize the following transcription content:\n\n{text}{time_info_text}{speaker_info_text}"
            else:
                # その他の言語は日本語をデフォルトとして使用
                system_prompt = SUMMARY_PROMPT_JA
                user_prompt = f"以下の文字起こし内容を要約してください:\n\n{text}{time_info_text}{speaker_info_text}"
            
            # LangChainを使用して要約を生成
            # プロンプトテンプレートの作成
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt = HumanMessagePromptTemplate.from_template(user_prompt)
            chat_prompt = ChatPromptTemplate.from_messages([
                system_message_prompt,
                human_message_prompt
            ])
            
            # LLMの初期化
            llm = ChatOpenAI(
                model="gpt-4.1-nano",  # モデルは環境に応じて調整
                temperature=0.3        # 低い温度で一貫性のある要約
            )
            
            # チェーンの作成と実行
            chain = chat_prompt | llm | StrOutputParser()
            summary = chain.invoke({})
            
            # 処理時間を計算
            processing_time = time.time() - start_time
            
            return {
                "summary": summary,
                "status": "completed",
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "summary": None,
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            }


