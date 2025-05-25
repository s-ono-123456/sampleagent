# pymupdf4llmを利用してPDFファイルをMarkdownに変換する関数を定義
import os
from pymupdf4llm.helpers.pymupdf_rag import to_markdown  # 直接to_markdownをインポート

# PDFファイルをMarkdownに変換し、指定ディレクトリに保存する関数
def convert_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    """
    PDFファイルをMarkdownに変換し、output_dirに保存する。
    変換後のMarkdownファイルのパスを返す。
    """
    # ファイル名から拡張子を除去し、mdファイル名を生成
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(output_dir, f"{base_name}.md")
    image_dir = os.path.join(output_dir, "images")

    # to_markdown関数でPDF全体をMarkdownテキストに変換
    markdown_text = to_markdown(pdf_path,
                                filename=md_path,  # 出力するMarkdownファイルのパス
                                write_images=True,  # 画像を埋め込む
                                image_path=image_dir,  # 画像の保存先ディレクトリ
                                )
    
    # markdown_text内の画像パス「(markdown/images/」を「(images/」に置換
    markdown_text = markdown_text.replace('(markdown/images/', '(images/')
    
    # Markdown形式で保存
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return md_path
