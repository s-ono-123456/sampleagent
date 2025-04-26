from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vector_stores(embedding_model_name="pkshatech/GLuCoSE-base-ja-v2"):
    """
    バッチ設計と画面設計のFAISSインデックスを読み込み、結合したベクトルストアを返す関数
    
    Args:
        embedding_model_name: 埋め込みモデルの名前
        
    Returns:
        db: 結合されたFAISSベクトルストア
    """
    # HuggingFaceのGLuCoSE埋め込みモデルをロード
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # バッチ設計のFAISSインデックスを読み込む
    batch_design_index_path = "faiss_index_batch_design"
    batch_design_db = FAISS.load_local(batch_design_index_path, embeddings, allow_dangerous_deserialization=True)

    # 画面設計のFAISSインデックスを読み込む
    screen_design_index_path = "faiss_index_screen_design"
    screen_design_db = FAISS.load_local(screen_design_index_path, embeddings, allow_dangerous_deserialization=True)

    # 両方のインデックスを結合する
    screen_design_db.merge_from(batch_design_db)
    return screen_design_db