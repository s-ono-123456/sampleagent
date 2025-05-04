from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_EMBEDDING_MODEL_NAME="pkshatech/GLuCoSE-base-ja-v2"

def load_vector_stores(embedding_model_name=BASE_EMBEDDING_MODEL_NAME):
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
    batch_design_index_path = "index/faiss_index_batch_design"
    batch_design_db = FAISS.load_local(batch_design_index_path, embeddings, allow_dangerous_deserialization=True)

    # 画面設計のFAISSインデックスを読み込む
    screen_design_index_path = "index/faiss_index_screen_design"
    screen_design_db = FAISS.load_local(screen_design_index_path, embeddings, allow_dangerous_deserialization=True)

    # 両方のインデックスを結合する
    screen_design_db.merge_from(batch_design_db)
    return screen_design_db

def load_vector_stores_category(category, embedding_model_name=BASE_EMBEDDING_MODEL_NAME):
    """
    バッチ設計と画面設計のFAISSインデックスを読み込み、結合したベクトルストアを返す関数
    
    Args:
        category: カテゴリ名
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
    index_path = f"index/faiss_index_{category}"
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    return db

def load_vector_stores_multiple_categories(categories, embedding_model_name=BASE_EMBEDDING_MODEL_NAME):
    """
    複数のカテゴリのFAISSインデックスを読み込み、結合したベクトルストアを返す関数
    
    Args:
        categories: カテゴリ名のリスト
        embedding_model_name: 埋め込みモデルの名前
        
    Returns:
        db: 結合されたFAISSベクトルストア
    """
    if not categories or len(categories) == 0:
        raise ValueError("少なくとも1つのカテゴリを指定してください")
    
    # HuggingFaceのGLuCoSE埋め込みモデルをロード
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 最初のカテゴリのインデックスをロード
    first_category = categories[0]
    index_path = f"index/faiss_index_{first_category}"
    merged_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 残りのカテゴリのインデックスをロードして結合
    for category in categories[1:]:
        index_path = f"index/faiss_index_{category}"
        try:
            db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            merged_db.merge_from(db)
        except Exception as e:
            print(f"カテゴリ {category} のロード中にエラーが発生しました: {e}")
    
    return merged_db