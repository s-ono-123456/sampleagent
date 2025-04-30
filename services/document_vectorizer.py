import os
import glob
from typing import List, Dict
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# GLuCoSE-base-ja-v2モデルを使用した埋め込み設定
embedding_model_name = "pkshatech/GLuCoSE-base-ja-v2"

def load_markdown_files(folder_path: str) -> List[Dict]:
    """
    フォルダ内のMarkdownファイルを読み込み、ファイルパスとコンテンツを返す
    """
    documents = []
    markdown_files = glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True)
    
    for file_path in markdown_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "file_path": file_path,
                    "content": content
                })
            print(f"読み込み完了: {file_path}")
        except Exception as e:
            print(f"ファイル読み込みエラー {file_path}: {str(e)}")
    
    return documents

def split_markdown_documents(documents: List[Dict]) -> List[Dict]:
    """
    Markdownドキュメントをチャンクに分割する
    """
    text_splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    split_documents = []
    
    for doc in documents:
        try:
            splits = text_splitter.split_text(doc["content"])
            for i, split in enumerate(splits):
                split_documents.append({
                    "file_path": doc["file_path"],
                    "content": split,
                    "chunk_id": i
                })
        except Exception as e:
            print(f"分割エラー {doc['file_path']}: {str(e)}")
    
    print(f"分割完了: 合計{len(split_documents)}チャンク")
    return split_documents

def create_faiss_index(split_documents: List[Dict], folder_name: str):
    """
    分割したドキュメントからFAISSインデックスを作成する
    """
    # HuggingFaceのGLuCoSE埋め込みモデルをロード
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # ドキュメント形式に変換
    texts = [doc["content"] for doc in split_documents]
    metadatas = [{"source": doc["file_path"], "chunk_id": doc["chunk_id"]} for doc in split_documents]
    
    # FAISSインデックスを作成
    db = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    
    # FAISSインデックスを保存
    index_path = f"index/faiss_index_{folder_name}"
    db.save_local(index_path)
    print(f"FAISSインデックスを保存しました: {index_path}")
    
    return index_path

def process_folder(folder_path: str):
    """
    指定されたフォルダ内のMarkdownファイルを処理する
    """
    folder_name = os.path.basename(folder_path)
    print(f"フォルダ処理開始: {folder_name}")
    
    # ファイル読み込み
    documents = load_markdown_files(folder_path)
    print(f"読み込んだファイル数: {len(documents)}")
    
    # 空のフォルダの場合はスキップ
    if not documents:
        print(f"フォルダ内にMarkdownファイルがありません: {folder_path}")
        return
    
    # ドキュメント分割
    split_documents = split_markdown_documents(documents)
    
    # FAISSインデックス作成
    index_path = create_faiss_index(split_documents, folder_name)
    
    print(f"フォルダ処理完了: {folder_name}, インデックス: {index_path}")

def main():
    # サンプルフォルダのパス
    base_folder = os.path.join(os.getcwd(), "sample")
    
    # サブフォルダを取得
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    
    if not subfolders:
        print("サブフォルダが見つかりません")
        # サンプルフォルダ全体を処理
        process_folder(base_folder)
    else:
        # 各サブフォルダを処理
        for subfolder in subfolders:
            process_folder(subfolder)

if __name__ == "__main__":
    main()