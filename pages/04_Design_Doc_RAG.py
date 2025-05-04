import streamlit as st
import json
import os
from typing import List, Dict, Any, Optional

# 自作モジュールのインポート
from services.rag_service import RAGManager, DEFAULT_MODEL_NAME, DEFAULT_EMBEDDING_MODEL_NAME, MAX_RESULTS

class RAGApp:
    """アプリケーション全体を統括するクラス（UI部分）"""
    
    def __init__(self):
        """初期化"""
        # RAGマネージャーの初期化
        self.rag_manager = RAGManager()
        
        # セッション状態の初期化
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """セッション状態を初期化"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "search_results" not in st.session_state:
            st.session_state.search_results = []
            
        # 設定のデフォルト値
        if "categories" not in st.session_state:
            st.session_state.categories = []
            
        if "max_results" not in st.session_state:
            st.session_state.max_results = MAX_RESULTS
            
        if "model_name" not in st.session_state:
            st.session_state.model_name = DEFAULT_MODEL_NAME
            
        # カテゴリが選択されたかどうかのフラグ
        if "categories_selected" not in st.session_state:
            st.session_state.categories_selected = False
    
    def run(self):
        """アプリケーションを実行"""
        st.title("設計書検索アシスタント")
        
        # サイドバーの設定
        self._setup_sidebar()
        
        # タブの設定
        tab1, tab2 = st.tabs(["チャット", "検索結果"])
        
        # チャットタブの表示
        with tab1:
            self._display_chat()
            # 入力フィールドをチャットタブの中に移動
            self._setup_input_field()
        
        # 検索結果タブの表示
        with tab2:
            self._display_search_results()
    
    def _setup_sidebar(self):
        """サイドバーの設定"""
        st.sidebar.header("オプション設定")
        
        # カテゴリ選択をチェックボックスで実装（複数選択可能）
        st.sidebar.subheader("検索カテゴリ (必須)")
        
        # カテゴリを選択肢として表示
        batch_checked = st.sidebar.checkbox("バッチ設計", key="batch_design_checkbox")
        screen_checked = st.sidebar.checkbox("画面設計", key="screen_design_checkbox")
        
        # 選択されたカテゴリをリストに格納
        selected_categories = []
        if batch_checked:
            selected_categories.append("batch_design")
        if screen_checked:
            selected_categories.append("screen_design")
        
        # 選択状態をセッションに保存
        st.session_state.categories = selected_categories
        
        # カテゴリが選択されたかチェック
        if selected_categories:
            st.session_state.categories_selected = True
            st.sidebar.success(f"{len(selected_categories)}つのカテゴリが選択されました")
        else:
            st.session_state.categories_selected = False
            st.sidebar.warning("少なくとも1つのカテゴリを選択してください（必須）")
        
        # 検索結果数の調整
        st.session_state.max_results = st.sidebar.slider("表示する検索結果数", 1, 10, MAX_RESULTS)
        
        # モデル選択
        model_options = ["gpt-4.1-nano", "gpt-4o"]
        selected_model = st.sidebar.selectbox("回答生成モデル", model_options, index=0)
        
        # モデルが変更された場合、RAGマネージャーのモデルも更新
        if selected_model != st.session_state.model_name:
            st.session_state.model_name = selected_model
            self.rag_manager.update_model(selected_model)
    
    def _display_chat(self):
        """チャット履歴の表示"""
        # カテゴリが選択されていない場合の警告表示
        if not st.session_state.categories_selected:
            st.warning("⚠️ サイドバーからカテゴリを選択してください。カテゴリ選択は必須です。")
            
        # チャット履歴の表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def _display_search_results(self):
        """検索結果の表示"""
        if not st.session_state.search_results:
            if st.session_state.categories_selected:
                st.info("検索結果はありません。質問を入力すると、関連する設計書が表示されます。")
            else:
                st.warning("サイドバーからカテゴリを選択してから質問してください。")
            return
        
        st.subheader("最新の質問に関する検索結果")
        
        # カテゴリ情報の表示
        current_categories = ", ".join(
            ["バッチ設計" if category == "batch_design" else "画面設計" for category in st.session_state.categories]
        )
        st.markdown(f"**検索カテゴリ:** {current_categories}")
        
        for idx, result in enumerate(st.session_state.search_results, 1):
            with st.expander(f"結果 {idx}: {result['file_name']}"):
                st.markdown(f"**ファイル**: {result['file_name']}")
                
                if result["score"] is not None:
                    score_percentage = round(result["score"] * 100, 2)
                    st.markdown(f"**関連度**: {score_percentage}%")
                
                st.markdown("**内容**:")
                st.markdown(result["content"])
    
    def _setup_input_field(self):
        """入力フィールドの設定"""
        # プレースホルダーテキストを調整
        if not st.session_state.categories_selected:
            placeholder_text = "サイドバーからカテゴリを選択してから質問してください"
        else:
            category_names = ", ".join(
                ["バッチ設計" if category == "batch_design" else "画面設計" for category in st.session_state.categories]
            )
            placeholder_text = f"{category_names}について質問する..."
        
        # フォームではなく、チャット入力を使用
        prompt = st.chat_input(placeholder_text, disabled=not st.session_state.categories_selected)
        
        if prompt:
            # ユーザーの入力をセッションに追加
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 最新のメッセージを表示
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 処理中の表示
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("回答を生成中です...")
                
                # RAGプロセスの実行
                self._process_query(prompt, message_placeholder)
    
    def _process_query(self, question: str, message_placeholder):
        """クエリ処理の実行"""
        try:
            # カテゴリが選択されていない場合は処理しない（UI側で無効化しているので通常は実行されない）
            if not st.session_state.categories_selected:
                error_message = "サイドバーからカテゴリを選択してから質問してください。"
                message_placeholder.warning(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                return
            
            # RAGマネージャーを使用してクエリを処理
            answer, search_results = self.rag_manager.process_query(
                question, 
                st.session_state.categories, 
                st.session_state.max_results
            )
            
            # 検索結果をセッションに保存
            st.session_state.search_results = search_results
            
            # 回答をセッションに追加
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # 回答を表示
            message_placeholder.markdown(answer)
            
        except Exception as e:
            error_message = f"処理中にエラーが発生しました: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


# アプリケーションの実行
if __name__ == "__main__":
    app = RAGApp()
    app.run()