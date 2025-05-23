# 設計書調査・質問対応AIエージェント - Plan and Execute型（LangGraph）

## 1. 概要

このAIエージェントは、設計書に関する調査や質問に回答・対応するために設計されたPlan and Execute型のシステムです。設計書の内容を理解し、ユーザーからの複雑な質問に対して段階的な計画を立てながら回答を導き出します。LangGraphを利用することで、より柔軟で透明性の高いワークフローを実現しています。

## 2. アーキテクチャ

### 2.1 全体構成

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  ユーザー入力    │────▶│    プランナー    │────▶│  タスク実行者   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         ▲                      │                      │
         │                      │                      │
         │                      ▼                      ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    回答生成     │◀────│ 情報充足度評価   │◀────│  ベクトル検索   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │   計画評価ノード  │
                        └──────────────────┘
```

### 2.2 主要コンポーネント

1. **ユーザーインターフェース**
   - ユーザーからの質問を受け付け
   - 回答を表示

2. **グラフノード**
   - **プランナーノード**: ユーザーの質問から計画を生成
   - **タスク実行ノード**: 各ステップを実行
   - **ベクトル検索ノード**: 関連情報の取得
   - **計画評価ノード**: 結果に基づく計画修正の判断
   - **計画修正ノード**: 必要に応じた計画の修正
   - **回答生成ノード**: 最終回答の作成

3. **状態管理**
   - グラフの進行状況を追跡
   - ノード間のデータフローを管理
   - 中間結果の保存と活用

4. **ベクトルデータベース**
   - 設計書のベクトル化データを格納
   - 類似度検索による関連情報の抽出

5. **階層的なグラフ構造**
   - **メイングラフ**: 全体のワークフローを管理
   - **サブグラフ**: 特定のステップ（検索や分析）を詳細に実行

## 3. 実装詳細

### 3.1 状態定義

エージェントの状態は以下の要素を含むPydanticモデルとして定義されます：

- **query**: ユーザーからの質問文字列
- **plan**: 計画ステップのリスト（各ステップはステップ番号、説明、アクションタイプを含む）
- **current_step_index**: 現在実行中のステップインデックス
- **execution_results**: 実行結果のリスト
- **need_plan_revision**: 計画修正が必要かどうかのフラグ
- **final_answer**: 最終的な回答
- **next_step**: 次に実行すべきステップの識別子

### 3.2 LangGraph による実装

#### 3.2.1 計画作成ノード（create_plan）

**機能**:
- ユーザーの質問から解決のための計画ステップを生成
- JSON形式で構造化された計画を作成
- 計画は検索、分析、統合の基本ステップを含む

**処理フロー**:
1. ユーザーの質問を入力として受け取る
2. LLM（gpt-4.1-nano）を使用して計画を生成
3. 生成された計画をJSON形式で解析
4. エラー発生時にはフォールバック計画を提供
5. 状態の計画属性を更新

#### 3.2.2 タスク実行ノード（execute_step）

**機能**:
- 計画の各ステップを順番に実行
- アクションタイプ（search/analyze/synthesize）に基づいた処理
- 実行結果の記録と成功/失敗の判断

**アクションタイプ**:
- **search**: ベクトルデータベースで関連情報を検索
- **analyze**: 収集した情報を分析して洞察を提供
- **synthesize**: 最終回答の統合準備

**処理フロー**:
1. 現在のステップ情報を取得
2. アクションタイプに応じた処理を実行
3. 結果を記録し、成功/失敗を判断
4. 失敗時に計画修正フラグを設定
5. 状態を更新して次のステップへ進行

#### 3.2.3 ベクトル検索機能

**機能**:
- ユーザーの質問や計画ステップに基づいた複数検索クエリの生成
- ベクトルデータベース（FAISS）を使用した類似度検索
- 検索結果の重複除去と整形
- 検索結果のフォーマットと返却

**処理フロー**:
1. 複数の検索クエリの生成（LLMを活用）
2. ベクトルストアのロード
3. 各クエリでの類似度検索の実行
4. 重複ドキュメントの除外
5. 検索結果のフォーマット

**実装詳細**:

1. **検索クエリ生成**:
   - ステップ説明と元の質問から効果的な検索クエリを複数生成
   - ChatGPTを使用して自然言語から複数の異なる視点のキーワードベースのクエリへ変換
   - 構造化出力（MultiSearchQuery）による一貫性の確保
   - 通常3つの異なるクエリを生成し、検索の多様性を確保

2. **ベクトルストア管理**:
   - FAISSライブラリによる高速な類似度検索
   - 複数のインデックスの結合（バッチ設計と画面設計）
   - 埋め込みモデルはHuggingFace GLuCoSE日本語モデルを使用
   - 埋め込みの正規化によるコサイン類似度の最適化

3. **類似度検索実行**:
   - 各クエリごとに上位k件（デフォルトで3件）の関連ドキュメント取得
   - 重複ドキュメントの除外（ドキュメントのソースと内容の先頭部分で判定）
   - 収集したドキュメントの統合と整理
   - 検索失敗時の適切なフォールバック処理

4. **重複除去処理**:
   - ドキュメントのソースと内容の先頭100文字をキーとして使用
   - セット構造を利用した効率的な重複検出
   - 各クエリで得られた新しいドキュメントのみを追加
   - 重複のないユニークな情報セットの構築

5. **結果フォーマット**:
   - ドキュメントソースと内容の構造化
   - マルチドキュメントの結合と整形
   - 後続の分析ステップで利用しやすい形式での提供
   - 検索クエリ情報をUIでの表示用に返却

**エラー処理**:
- インデックスが見つからない場合のグレースフルエラー
- 空の検索結果に対する適切なフィードバック
- 埋め込みモデルのロード失敗時の代替戦略
- 個別クエリの検索エラー時に他のクエリ結果を使用する冗長性

#### 3.2.4 計画評価ノード（evaluate_plan）

**機能**:
- 現在の実行状態を評価し、次のアクションを決定
- 条件分岐の制御ポイントとして機能

**分岐条件**:
- 計画修正が必要な場合 → revise_planノードへ
- 次のステップがある場合 → execute_stepノードへ
- すべてのステップが完了した場合 → generate_answerノードへ

**処理フロー**:
1. 実行状態の検査
   - 計画修正フラグ（need_plan_revision）の確認
   - 現在のステップインデックスと計画の長さの比較
2. 条件に基づく分岐先の決定
   - 各条件に対応する次ノードの識別子を設定
3. 状態の更新
   - next_step属性を適切な値に設定
4. 分岐情報の返却

**実装詳細**:

1. **状態判定ロジック**:
   - `need_plan_revision`フラグを最優先で確認
   - ステップの進行状況を確認するための`current_step_index`とプラン配列長の比較
   - 条件を順次評価し最初に合致した分岐を選択

2. **分岐先の制御**:
   - 状態に応じた分岐先を`next_step`属性として返却
   - LangGraphの条件付きエッジと連携して適切なノードへフロー制御
   - シンプルな三項分岐による効率的な実装

3. **エッジケース処理**:
   - 計画が空の場合でも適切に動作するロバスト性
   - 予期しない状態の検出と適切な対応
   - 明示的な戻り値指定による動作保証

**状態更新**:
- `next_step`属性のみを更新（他の状態属性は変更しない）
- 更新された状態は条件付きエッジによって適切なノードへ渡される

**エラー処理**:
- 状態オブジェクトの整合性チェック
- 無効なステップ状態の検出
- 明示的なフロー制御による例外防止

**グラフ内の位置づけ**:
- エージェントのワークフローの中心的な制御ポイント
- 各実行ステップ後に必ず通過するハブノード
- グラフの実行フローを動的に制御する意思決定ポイント

#### 3.2.5 計画修正ノード（revise_plan）

**機能**:
- 実行結果に基づいて計画を修正
- 失敗したステップに対する代替アプローチの提案
- 新しい計画の生成とフォーマット

**処理フロー**:
1. 実行結果の集約
2. LLMを使用した修正計画の生成
3. 新しい計画のJSON解析
4. エラー時のフォールバック計画提供
5. 状態の更新と実行の再開

#### 3.2.6 回答生成ノード（generate_answer）

**機能**:
- すべての実行結果から最終回答を生成
- ユーザーの質問に対する明確で根拠のある回答の作成

**処理フロー**:
1. 全実行結果の集約
2. LLMを使用した最終回答の生成
3. 状態の更新（final_answer属性）

**実装詳細**:

1. **情報の統合**:
   - 全ステップの実行結果を集約し、構造化された情報セットを作成
   - 各ステップタイプ（検索、分析など）の結果を区別して処理
   - 重要な情報を優先的に含める仕組み

2. **回答生成**:
   - 統合された情報を元に一貫性のある回答を生成
   - オリジナルの質問に直接対応する回答を優先
   - 明確で簡潔、かつ情報に富んだ回答形式

**回答生成プロンプト**:
回答生成には以下のプロンプトテンプレートを使用しています：

```
収集したすべての情報に基づいて、ユーザーの質問に対する最終的な回答を作成してください。

質問: {query}

収集した情報:
{all_results}

回答は明確で簡潔、かつ情報に富んだものにしてください。
設計書の内容に基づいて事実を述べ、根拠となる情報を含めてください。

回答:
```

このプロンプトにより、収集した情報を元に簡潔かつ情報豊富な回答を生成します。特に設計書の内容に基づいた事実と根拠を含めることを重視しています。

#### 3.2.7 情報充足度評価ノード（assess_information_sufficiency）

**機能**:
- 最終回答を生成する前に、収集した情報の充足度を評価
- 再計画の必要性を判断し、必要に応じて追加調査を実施
- より良い回答のための情報品質確認
- 前の計画で収集した情報も累積的に保持・活用

**処理フロー**:
1. これまでのすべての計画で収集した実行結果を累積的に分析（情報は破棄せず保持）
2. 情報の量、質、関連性、一貫性の評価
3. 追加情報が必要かどうかの判断
4. 再計画が必要な場合は計画ノードへ、十分な場合は回答生成ノードへ

**実装詳細**:

1. **情報評価基準**:
   - **網羅性**: 全ての関連情報が収集されているか
   - **一貫性**: 矛盾する情報がないか
   - **適切性**: 質問に直接関連する情報か
   - **詳細度**: 十分な詳細情報が含まれているか
   - **最新性**: 情報が最新かどうか

2. **情報累積メカニズム**:
   - 計画修正が行われた場合でも、以前の計画で収集した有用な情報はすべて保持
   - `execution_results`配列に過去のすべての結果が累積的に格納される
   - 重複情報の検出と統合による効率的な情報管理
   - 過去の検索結果・分析結果を含めた総合的な評価

**評価プロンプト**:
情報充足度評価には以下のプロンプトテンプレートを使用しています：

```
あなたは質問に回答するために得られた洞察の充足度を評価するAIアシスタントです。
ユーザーの質問に対して、以下の洞察が十分であるかどうかを評価してください。

質問: {query}

得られた洞察:
{all_insights}

以下の評価基準に基づいて得られた洞察の充足度を評価してください:
1. **網羅性**: 質問に関連するすべての重要な側面がカバーされているか
2. **一貫性**: 矛盾する情報がないか
3. **適切性**: 洞察が質問に直接関連しているか
4. **詳細度**: 質問に答えるために十分な詳細情報が含まれているか
5. **最新性**: 情報が最新かどうか（もし判断できる場合）

充足度スコアは1〜5の整数で評価してください（1が最も不十分、5が最も十分）。
不足している情報がある場合は具体的に列挙してください。
再調査が必要かどうかを判断し（スコアが4未満の場合は必要）、その理由を説明してください。
```

このプロンプトにより、収集された情報の質と量を5段階で評価し、追加調査の必要性を判断します。スコアが4未満の場合には再調査が必要とみなされます。

## 3.3 ベクトルストアの実装概要

**機能**:
- 設計書文書のベクトル表現を格納
- 効率的な類似度検索の提供
- 複数のベクトルストアの管理

**設計**:
- OpenAI Embeddingsを使用した埋め込みモデル
- FAISSライブラリによるベクトルインデックス
- 複数のインデックスパスをサポート
- エラー処理とフォールバックメカニズム

## 4. データフロー

LangGraphを使用したエージェントのデータフローは以下の通りです：

1. ユーザーが設計書に関する質問を入力
2. PlanExecuteAgentクラスのrunメソッドが呼び出される
3. 初期状態（AgentState）が作成され、グラフの実行が開始
4. プランナーノード（create_plan）が計画を生成
5. タスク実行ノード（execute_step）が計画の各ステップを順番に実行
   - 検索アクション：ベクトル検索で関連情報を取得
   - 分析アクション：収集した情報を分析
   - 統合アクション：情報を統合して回答準備
6. 評価ノード（evaluate_plan）が各ステップの結果を評価し、次のアクションを決定
   - 計画修正が必要な場合（need_plan_revision = True）：修正ノードへ
   - 次のステップがある場合：実行ノードへ
   - すべてのステップが完了した場合：情報充足度評価ノードへ
7. 情報充足度評価ノード（assess_information_sufficiency）がこれまでの収集情報を評価
   - 情報が十分である場合：回答生成ノードへ
   - 追加情報が必要な場合：計画修正ノードへ
8. 計画修正が必要な場合、修正ノード（revise_plan）が新しい計画を作成
9. 情報が充分な場合、回答生成ノード（generate_answer）が最終的な回答を作成
10. 最終結果が辞書形式で返され、ユーザーに提示

## 5. エージェントの特徴

### 5.1 LangGraphの利点

- **明示的なワークフロー**: グラフ構造によってエージェントの挙動を視覚化可能
- **状態管理の簡素化**: Pydanticモデルによる型安全な状態管理
- **条件分岐の柔軟性**: 条件付きエッジによる動的なワークフロー制御
- **再利用性**: モジュール化されたノードによる高い再利用性
- **デバッグの容易さ**: 各ノードの実行と状態の変化を追跡可能
- **エラー処理**: 各ステップでの例外処理とフォールバック戦略の実装

### 5.2 エラー処理と堅牢性

- **JSONパース失敗時のフォールバック**: LLMの出力がJSONとして解析できない場合の代替計画
- **検索失敗時の処理**: 情報が見つからない場合の計画修正フロー
- **例外処理**: 各ステップでの例外キャッチと適切なエラーメッセージ
- **状態の一貫性保持**: エラー発生時も状態の整合性を維持

### 5.3 課題と対策

| 課題 | 対策 |
|------|------|
| グラフの複雑化 | モジュール化とサブグラフの利用 |
| 状態管理のオーバーヘッド | 効率的な状態更新とキャッシング戦略 |
| デバッグの複雑さ | グラフ実行のロギングとモニタリング |
| スケーラビリティ | 非同期処理とバッチ処理の導入 |
| LLM出力の不確実性 | 堅牢なエラー処理とフォールバック戦略 |

## 6. 使用技術

- **LLM**: OpenAI GPT-4.1-nano
- **フレームワーク**: LangChain + LangGraph
- **ベクトルDB**: FAISS via LangChain
- **埋め込み**: OpenAI Embeddings
- **開発言語**: Python
- **状態管理**: Pydantic BaseModel
- **環境変数管理**: `os.environ` と `os.getenv`

## 7. Streamlitによるユーザーインターフェース

### 7.1 UIの全体構成

```
┌───────────────────────┐     ┌───────────────────────┐
│  サイドバー           │     │  メインコンテンツ      │
│                       │     │                       │
│  - モデル選択         │     │  ┌───────────────┐   │
│  - パラメータ設定     │     │  │  タブ1:       │   │
│  - 質問例             │     │  │  チャット     │   │
│  - デバッグモード     │     │  └───────────────┘   │
│                       │     │                       │
│                       │     │  ┌───────────────┐   │
│                       │     │  │  タブ2:       │   │
│                       │     │  │  実行計画     │   │
│                       │     │  └───────────────┘   │
└───────────────────────┘     └───────────────────────┘
```

### 7.2 主要コンポーネント

1. **サイドバー**
   - モデル選択ドロップダウン
   - 温度パラメータスライダー
   - 質問例の表示
   - デバッグモード切替

2. **チャットタブ**
   - チャット履歴の表示
   - ユーザー入力フィールド
   - リアルタイム実行ステータス表示
   - 詳細なデバッグ情報（オプション）

3. **実行計画タブ**
   - 実行計画の詳細表示
   - 各ステップの詳細説明とアクションタイプ
   - 実行結果の詳細表示（成功/失敗状態の視覚的表現）
   - 実行計画が未実行の場合のガイドメッセージ

### 7.3 UIの実装詳細

#### 7.3.1 セッション状態管理

**状態変数**:
- `plan_execute_messages`: チャット履歴を格納
- `plan_execute_steps`: 最新の実行計画ステップを格納
- `plan_execute_results`: 実行結果を格納

**状態の初期化**:
アプリケーション起動時に、これらの状態変数が存在しない場合は空のリストとして初期化されます。これにより、アプリケーションの状態が正しく管理され、ユーザーセッション間でデータの一貫性が保たれます。

#### 7.3.2 エージェント実行フロー

1. ユーザーが質問を入力
2. 質問をチャット履歴に追加
3. エージェントの初期化（キャッシュされたリソースを活用）
4. 実行ステータスの表示とリアルタイム更新
5. エージェント実行結果の保存と表示
6. チャット履歴への回答追加

#### 7.3.3 視覚化コンポーネント

**実行計画の表示**:
- 計画ステップの詳細表示
- ステップごとの説明とアクションタイプの表示
- 実行結果の成功/失敗状態の視覚的表現（✅/❌）

**デバッグ情報の表示**:
- デバッグモードが有効な場合の詳細な実行情報表示
- 各ステップの処理内容と結果のカスタムUI表示
- 成功/失敗状態の視覚的フィードバック

#### 7.3.4 エラー処理と例外管理

- try-except構造によるエラーのグレースフルハンドリング
- エラーメッセージの視覚的表示
- デバッグモード時のスタックトレース表示

### 7.4 UIの利点と特徴

- **対話型インターフェース**: ユーザーフレンドリーなチャットベースUI
- **透明性の高い実行**: 各ステップの実行状況をリアルタイムで可視化
- **デバッグ機能**: 開発者向けの詳細なステップ実行情報を確認可能
- **直感的なレイアウト**: タブによる情報の論理的な分離
- **視覚的なフィードバック**: 成功/失敗状態の明示的な表示
- **柔軟なパラメータ調整**: サイドバーからのモデルパラメータ変更

### 7.5 Streamlit実装の利点

- **高速な開発**: 少ないコードで洗練されたUIを実現
- **インタラクティブ要素**: スライダー、ドロップダウン、エクスパンダーなどの豊富なUI要素
- **状態管理の簡素化**: session_stateによる簡潔な状態管理
- **動的コンテンツ更新**: 処理状況に応じたリアルタイムUI更新
- **マークダウンサポート**: 書式付きテキスト表示の容易さ
- **グラフ・図表の統合**: メルマイド図などの視覚化ツールとの連携

## 8. 将来の拡張可能性

1. **マルチモデル対応**: 複数のLLMモデルを切り替え可能にする
2. **ベクトルストア多様化**: 複数のベクトルストアをコンテキストに応じて使い分ける
3. **UIの機能拡張**: 設計書のアップロード機能や回答履歴のエクスポート機能
4. **パフォーマンス最適化**: キャッシングや並列処理の導入
5. **フィードバックループ**: ユーザーフィードバックに基づく計画の自己修正機能
6. **可視化オプションの拡充**: グラフやチャートによる分析結果の表示