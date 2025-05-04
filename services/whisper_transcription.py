import os
import tempfile
import time
from typing import Dict, List, Optional, Any, Union
import numpy as np
from pathlib import Path
import librosa  # ffmpegの代わりにlibrosaを使用
import soundfile as sf  # 音声ファイル読み書き用
import torch
from pydub import AudioSegment
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field

# LangGraph用のインポート
from langgraph.graph import StateGraph, END

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

# ノード関数の実装
def validate_file(state: WhisperTranscriptionState) -> WhisperTranscriptionState:
    """音声ファイルの形式と内容を検証するノード"""
    start_time = time.time()
    
    try:
        # ファイルパスのチェック
        if not state.audio_file or not os.path.exists(state.audio_file):
            raise ValueError(f"ファイルが存在しません: {state.audio_file}")
        
        # 拡張子のチェック
        supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        file_ext = os.path.splitext(state.audio_file)[1].lower()
        
        if file_ext not in supported_formats:
            raise ValueError(f"サポートされていないファイル形式です: {file_ext}。サポート形式: {', '.join(supported_formats)}")
        
        # ファイルサイズのチェック
        file_size = os.path.getsize(state.audio_file) / (1024 * 1024)  # MBに変換
        if file_size > 500:  # 500MB以上はエラー
            raise ValueError(f"ファイルサイズが大きすぎます: {file_size:.2f}MB (最大500MB)")
        
        # librosaを使って音声メタデータを抽出
        try:
            # ヘッダー情報のみを読み込むことで高速化
            y = None
            sr = librosa.get_samplerate(state.audio_file)
            
            # ファイルが開けるか確認するために短いセグメントだけ読み込む
            y, _ = librosa.load(state.audio_file, sr=sr, duration=0.1)
            
            if y is None or sr is None:
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
        
        # librosaを使用して音声をロード（サンプルレート16kHz、モノラルに変換）
        audio_data, sample_rate = librosa.load(state.audio_file, sr=16000, mono=True)
        
        # 音量正規化
        if np.max(np.abs(audio_data)) < 0.1:
            audio_data = librosa.util.normalize(audio_data)
        
        # 状態を更新
        state.audio_data = audio_data
        state.status = "audio_preprocessed"
            
    except Exception as e:
        state.status = "error"
        state.error_message = f"音声前処理エラー: {str(e)}"
    
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
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        
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
        segments, info = state.model.transcribe(audio_segment, task="detect_language")
        detected_language = info.language
        
        # 状態を更新
        state.language = detected_language
        state.status = "language_detected"
        
    except Exception as e:
        state.status = "error"
        state.error_message = f"言語検出エラー: {str(e)}"
    
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
    workflow.add_edge("transcribe_audio", "postprocess_text")
    workflow.add_edge("postprocess_text", "format_output")
    workflow.add_edge("format_output", "evaluate_result")
    
    # evaluate_resultノードから条件に基づいて分岐
    workflow.add_conditional_edges(
        "evaluate_result",
        lambda state, result: result["next"],
        {
            "finalize": "finalize"
        }
    )
    
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
                  format_type: str = "text") -> WhisperTranscriptionState:
        """
        音声ファイルの文字起こしを実行する
        
        Args:
            audio_file: 音声ファイルのパス
            language: 言語コード（指定しない場合は自動検出）
            format_type: 出力形式（text, srt, vtt, json）
            
        Returns:
            文字起こし処理の結果
        """
        # 入力状態の準備
        input_state = WhisperTranscriptionState(
            audio_file=audio_file,
            language=language,
            format_type=format_type
        )
        
        # グラフの実行
        result = self.graph.invoke(input_state)
        
        return result

# 使用例：
# service = WhisperTranscriptionService()
# result = service.transcribe("path/to/audio.mp3", format_type="srt")
# print(result.transcript)
# print(f"出力ファイル: {result.output_file}")