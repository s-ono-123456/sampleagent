import streamlit as st
import os
from services.pdf_to_markdown import convert_pdf_to_markdown

# wideレイアウトを有効化
st.set_page_config(layout="wide")

# 保存先ディレクトリ
UPLOAD_DIR = "uploads"
MARKDOWN_DIR = "markdown"

# ディレクトリがなければ作成
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

st.title("PDF→Markdown変換ツール（pymupdf4llm利用）")

# ファイルアップロードUI
uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type=["pdf"])

if uploaded_file is not None:
    # 一時的にアップロードファイルを保存
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"アップロード完了: {uploaded_file.name}")

    # PDF→Markdown変換
    md_path = convert_pdf_to_markdown(pdf_path, MARKDOWN_DIR)
    st.success(f"Markdown変換完了: {os.path.basename(md_path)}")

    # 変換後にPDFファイルを削除
    try:
        os.remove(pdf_path)
        st.info(f"変換後、PDFファイルを削除しました: {uploaded_file.name}")
    except Exception as e:
        st.warning(f"PDFファイルの削除に失敗しました: {e}")

    # Markdownファイルのダウンロードリンク（プレビューの上に移動）
    with open(md_path, "rb") as f:
        st.download_button(
            label="Markdownファイルをダウンロード",
            data=f,
            file_name=os.path.basename(md_path),
            mime="text/markdown"
        )

    # Markdown内容を表示
    with open(md_path, encoding="utf-8") as f:
        md_content = f.read()
    st.subheader("変換結果（Markdownプレビュー）")
    st.markdown(md_content)

# markdownディレクトリ内の既存Markdown一覧表示
st.sidebar.header("保存済みMarkdown一覧")
md_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith(".md")]
for md_file in md_files:
    if st.sidebar.button(md_file):
        with open(os.path.join(MARKDOWN_DIR, md_file), encoding="utf-8") as f:
            st.markdown(f.read())
