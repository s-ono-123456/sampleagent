import streamlit as st
import os
import re
from services.pdf_to_markdown import convert_pdf_to_markdown
import PyPDF2  # PDF結合用ライブラリを追加

# win32com.clientのインポートをtry-exceptでラップ
try:
    import win32com.client  # win32comをインポート
except ImportError:
    win32com = None  # インポート失敗時はNoneにする

# wideレイアウトを有効化
st.set_page_config(layout="wide")

# 保存先ディレクトリ
UPLOAD_DIR = "uploads"
MARKDOWN_DIR = "markdown"

# ディレクトリがなければ作成
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

st.title("PDF→Markdown変換ツール")

# ファイルアップロードUI（PDFとExcel両方対応）
uploaded_file = st.file_uploader("PDFまたはExcelファイルをアップロードしてください", type=["pdf", "xlsx"])

if uploaded_file is not None:
    # アップロードされたファイル名の()を_に置換
    safe_filename = uploaded_file.name.replace('(', '_').replace(')', '_')
    # 拡張子を取得
    ext = os.path.splitext(safe_filename)[1].lower()
    # 一時的にアップロードファイルを保存
    input_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"アップロード完了: {safe_filename}")

    # PDFファイルパスを初期化
    pdf_path = None

    if ext == ".pdf":
        # PDFの場合はそのまま
        pdf_path = input_path
    elif ext == ".xlsx":
        # Excel→PDF変換の前にwin32comが使えるかチェック
        if win32com is None:
            st.error("win32com.clientがインポートできません。pywin32がインストールされているか確認してください。")
            pdf_path = None
        else:
            try:
                # Excelアプリケーションの起動を試みる
                excel = win32com.client.Dispatch("Excel.Application")
                excel.Visible = False  # Excelウィンドウを表示しない
                try:
                    wb = excel.Workbooks.Open(os.path.abspath(input_path))
                    # シートごとにPDFを一時保存し、後で結合する
                    sheet_pdfs = []  # 一時PDFファイルのパスリスト
                    for i, sheet in enumerate(wb.Sheets):
                        # 各シートをアクティブにしてPDF出力
                        sheet.Select()
                        sheet_pdf_path = os.path.splitext(input_path)[0] + f"_sheet{i+1}.pdf"
                        wb.ActiveSheet.ExportAsFixedFormat(0, sheet_pdf_path)
                        sheet_pdfs.append(sheet_pdf_path)
                    wb.Close(False)
                    # 複数PDFを結合
                    if len(sheet_pdfs) > 1:
                        # PyPDF2.PdfMergerをwith文で使用してリソースを自動解放
                        merged_pdf_path = os.path.splitext(input_path)[0] + "_merged.pdf"
                        with PyPDF2.PdfMerger() as merger:
                            for pdf in sheet_pdfs:
                                with open(pdf, "rb") as f:
                                    merger.append(f)
                            with open(merged_pdf_path, "wb") as fout:
                                merger.write(fout)
                        pdf_path = merged_pdf_path
                        # 一時PDF削除
                        for pdf in sheet_pdfs:
                            os.remove(pdf)
                    elif len(sheet_pdfs) == 1:
                        pdf_path = sheet_pdfs[0]
                    else:
                        pdf_path = None
                    if pdf_path:
                        st.info(f"Excelファイルの全シートをPDFに変換・結合しました: {os.path.basename(pdf_path)}")
                    else:
                        st.warning("Excelファイルにシートが見つかりませんでした。PDFの生成に失敗しました。")
                except Exception as e:
                    st.error(f"Excel→PDF変換に失敗しました: {e}")
                    pdf_path = None
                finally:
                    excel.Quit()
            except Exception as e:
                # ExcelのCOM起動自体が失敗した場合
                st.error(f"Excelアプリケーションの起動に失敗しました。Excelがインストールされているか、ビット数が一致しているか確認してください。詳細: {e}")
                pdf_path = None
    else:
        st.error("対応していないファイル形式です。PDFまたはExcelファイルをアップロードしてください。")

    # PDF変換に成功した場合のみMarkdown変換
    if pdf_path and os.path.exists(pdf_path):
        # PDF→Markdown変換
        md_path = convert_pdf_to_markdown(pdf_path, MARKDOWN_DIR)
        st.success(f"Markdown変換完了: {os.path.basename(md_path)}")

        # 変換後に一時PDFファイルを削除（Excelアップロード時もPDFを削除）
        try:
            os.remove(pdf_path)
            st.info(f"変換後、PDFファイルを削除しました: {os.path.basename(pdf_path)}")
        except Exception as e:
            st.warning(f"PDFファイルの削除に失敗しました: {e}")
        # Excelファイルも削除（必要なら）
        if ext == ".xlsx":
            try:
                os.remove(input_path)
                st.info(f"アップロードされたExcelファイルを削除しました: {safe_filename}")
            except Exception as e:
                st.warning(f"Excelファイルの削除に失敗しました: {e}")

        # Markdown内容を表示
        with open(md_path, encoding="utf-8") as f:
            md_content = f.read()
        st.subheader("変換結果（Markdownプレビュー）")

        # --- Markdownと画像をまとめてzipでダウンロードする処理 ---
        import io
        import zipfile
        # 正規表現モジュールは既にインポート済み
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

        # --- 画像とテキストを順番に表示する処理 ---
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

