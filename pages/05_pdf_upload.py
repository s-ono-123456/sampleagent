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
    # アップロードされたPDFファイル名の()を_に置換
    safe_filename = uploaded_file.name.replace('(', '_').replace(')', '_')
    # 一時的にアップロードファイルを保存
    pdf_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"アップロード完了: {safe_filename}")

    # PDF→Markdown変換
    md_path = convert_pdf_to_markdown(pdf_path, MARKDOWN_DIR)
    st.success(f"Markdown変換完了: {os.path.basename(md_path)}")

    # 変換後にPDFファイルを削除
    try:
        os.remove(pdf_path)
        st.info(f"変換後、PDFファイルを削除しました: {safe_filename}")
    except Exception as e:
        st.warning(f"PDFファイルの削除に失敗しました: {e}")

    # Markdown内容を表示
    with open(md_path, encoding="utf-8") as f:
        md_content = f.read()
    st.subheader("変換結果（Markdownプレビュー）")

    # --- Markdownと画像をまとめてzipでダウンロードする処理（ボタンを上に移動） ---
    import io
    import zipfile
    import re  # 正規表現モジュール
    # 画像パスを抽出
    image_pattern = r'!\[[^\]]*\]\(([^\)]+)\)'
    image_paths = re.findall(image_pattern, md_content)
    # zipファイルをメモリ上に作成
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        # Markdownファイルを追加
        zipf.writestr(os.path.basename(md_path), md_content)
        # 画像ファイルを追加
        for img_path in image_paths:
            abs_img_path = img_path
            abs_img_path = os.path.normpath(abs_img_path)
            if os.path.exists(abs_img_path):
                # zip内のパスをmarkdown/images/xxx.png形式にする
                arcname = os.path.join(MARKDOWN_DIR, os.path.relpath(abs_img_path, MARKDOWN_DIR))
                zipf.write(abs_img_path, arcname)
    zip_buffer.seek(0)
    st.download_button(
        label="Markdown＋画像をまとめてダウンロード（zip）",
        data=zip_buffer,
        file_name=os.path.splitext(os.path.basename(md_path))[0] + ".zip",
        mime="application/zip"
    )
    # --- zipダウンロードここまで ---

    # Markdownファイルのダウンロードリンク（プレビューの上に移動）
    with open(md_path, "rb") as f:
        st.download_button(
            label="Markdownファイルをダウンロード",
            data=f,
            file_name=os.path.basename(md_path),
            mime="text/markdown"
        )

    # --- 画像とテキストを順番に表示する処理 ---
    import re  # 正規表現モジュール
    # 画像付きMarkdownのパース用正規表現
    pattern = r'(!\[[^\]]*\]\([^\)]+\))'
    parts = re.split(pattern, md_content)
    for part in parts:
        # 画像部分の場合
        img_match = re.match(r'!\[([^\]]*)\]\(([^\)]+)\)', part)
        if img_match:
            alt_text = img_match.group(1)
            img_path = img_match.group(2)
            abs_img_path = img_path
            abs_img_path = os.path.normpath(abs_img_path)
            if os.path.exists(abs_img_path):
                st.image(abs_img_path, caption=alt_text or os.path.basename(img_path))
            else:
                st.warning(f"画像が見つかりません: {abs_img_path}")
        else:
            # テキスト部分はMarkdownとして表示
            if part.strip():
                st.markdown(part)
    # --- 画像とテキスト順次表示ここまで ---

# markdownディレクトリ内の既存Markdown一覧表示
st.sidebar.header("保存済みMarkdown一覧")
md_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith(".md")]
for md_file in md_files:
    if st.sidebar.button(md_file):
        with open(os.path.join(MARKDOWN_DIR, md_file), encoding="utf-8") as f:
            st.markdown(f.read())
