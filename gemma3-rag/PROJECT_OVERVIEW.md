# プロジェクト概要 - Gemma3 RAG KasenSabo MVP

## 🎯 プロジェクト目的

河川・ダム・砂防の技術基準を知識ベースとして、Gemma 3モデル（INT4/INT8量子化）を使用したRAGシステムの性能をベンチマーク評価する。

## 📊 主要コンポーネント

```
┌─────────────────────────────────────────────────────────────┐
│                    Gemma3 RAG System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐    ┌─────────────┐ │
│  │  Knowledge   │      │   Vector     │    │   Ollama    │ │
│  │    Base      │─────▶│   Store      │◀───│  Gemma 3    │ │
│  │  (Markdown)  │      │  (ChromaDB)  │    │ INT4/INT8   │ │
│  └──────────────┘      └──────────────┘    └─────────────┘ │
│         │                     │                    │        │
│         │                     │                    │        │
│         ▼                     ▼                    ▼        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Query Engine (LlamaIndex)                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Evaluation (EM, F1, BLEU, ROUGE, etc.)        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 ファイル構成と役割

### 設定ファイル

| ファイル | 役割 |
|---------|------|
| `config.yaml` | システム全体の設定（モデル、パラメータ、パス） |
| `requirements.txt` | Python依存パッケージ |
| `.gitignore` | Git管理対象外ファイル |

### スクリプト

| ファイル | 役割 | 実行時間 |
|---------|------|----------|
| `scripts/build_index.py` | 知識ベースをChromaDBにインデックス化 | 5-10分 |
| `scripts/run_rag.py` | 単一クエリでRAGを実行（テスト用） | 数秒 |
| `scripts/evaluate.py` | 評価指標の計算（EM, F1, BLEU, ROUGE） | 即時 |
| `scripts/run_benchmark.py` | 200問のフルベンチマーク実行 | 30-60分 |

### データディレクトリ

| ディレクトリ | 内容 |
|-------------|------|
| `data/kasensabo_knowledge_base/` | 技術基準Markdown（8ファイル） |
| `questions/` | ベンチマーク質問JSON（200問） |
| `index/` | ChromaDBベクトルインデックス（生成） |
| `results/` | ベンチマーク結果JSON/CSV（生成） |

### ドキュメント

| ファイル | 内容 |
|---------|------|
| `README.md` | プロジェクト概要と使用方法 |
| `SETUP_GUIDE.md` | 詳細セットアップ手順 |
| `PROJECT_OVERVIEW.md` | このファイル |
| `quickstart.ps1` | 自動セットアップスクリプト |

## 🔄 実行フロー

### 1. セットアップフェーズ

```
Ollama準備 → Python環境構築 → インデックス構築
    ↓              ↓                 ↓
モデルDL      依存関係install    ChromaDB作成
(5分)          (3分)             (5-10分)
```

### 2. ベンチマークフェーズ

```
質問読込 → INT4実行 → INT8実行 → 評価計算 → 結果保存
   ↓         ↓          ↓          ↓         ↓
 200問    15-30分    15-30分    全指標    JSON/CSV
```

## 📈 評価指標の詳細

### テキスト品質指標

| 指標 | 説明 | 計算方法 | 範囲 |
|------|------|----------|------|
| **EM** | 完全一致 | 正規化後の文字列比較 | 0 or 1 |
| **F1** | トークンF1 | Precision × Recall の調和平均 | 0.0-1.0 |
| **BLEU-1/2/3/4** | n-gram一致率 | n-gramの重複度 | 0.0-1.0 |
| **ROUGE-1/2/L** | リコール重視 | 最長共通部分列 | 0.0-1.0 |

### パフォーマンス指標

| 指標 | 説明 | 単位 |
|------|------|------|
| **Response Time** | クエリ応答時間 | 秒 |
| **Memory Usage** | メモリ増分 | MB |

### 集計統計

各指標について以下を計算：
- **Mean**: 平均値
- **Std**: 標準偏差
- **Min/Max**: 最小値/最大値
- **Median**: 中央値

## 🔬 比較軸

### モデル比較（INT4 vs INT8）

| 比較項目 | 評価方法 |
|---------|---------|
| **精度** | F1, BLEU, ROUGEスコアの平均 |
| **速度** | Response Timeの平均・分散 |
| **メモリ** | Memory Usageの平均・最大値 |
| **安定性** | スコアの標準偏差 |

### カテゴリ別分析

技術基準の分類別に精度を分析：
- 河川計画
- ダム設計
- 砂防施設
- 維持管理

## 🎓 期待される知見

### 1. 量子化による影響

```
仮説: INT4は速度重視、INT8は精度重視

検証:
- Response Time: INT4 < INT8
- F1 Score: INT4 < INT8
- Memory Usage: INT4 < INT8
```

### 2. RAGパラメータの最適化

```
実験変数:
- chunk_size: 256, 512, 1024
- similarity_top_k: 2, 3, 5
- temperature: 0.0, 0.1, 0.3
```

### 3. 分野別の適性

```
分析観点:
- どのカテゴリで精度が高い/低いか
- エラーパターンの分析
- ハルシネーションの傾向
```

## 🚀 拡張可能性

### 短期的拡張

1. **他のモデルとの比較**
   - Llama 3
   - Mistral
   - Phi-3

2. **評価指標の追加**
   - BERT Score
   - 専門用語の正確性
   - ファクトチェック

3. **質問セットの拡充**
   - 難易度別分類
   - マルチホップ質問
   - ネガティブサンプル

### 中期的拡張

1. **API化**
   - FastAPI/Flask
   - REST/GraphQL
   - 認証・認可

2. **Webインターフェース**
   - Streamlit/Gradio
   - リアルタイムフィードバック
   - 可視化ダッシュボード

3. **継続学習**
   - ファインチューニング
   - RLHF
   - ドメイン適応

### 長期的展開

1. **本番運用**
   - スケーリング
   - モニタリング
   - A/Bテスト

2. **マルチモーダル対応**
   - 図表の認識
   - PDF直接処理
   - 動画解説

## 📊 結果の活用方法

### 1. レポート作成

```python
# results/からデータを読み込んで分析
import pandas as pd
import json

# 結果の読み込み
with open('results/model_comparison_*.csv') as f:
    df = pd.read_csv(f)

# 可視化
import matplotlib.pyplot as plt
df.plot(kind='bar', x='Model')
plt.savefig('comparison_chart.png')
```

### 2. モデル選定

```
用途別推奨:
- プロトタイプ/デモ: INT4（速度優先）
- 本番/精度重視: INT8（品質優先）
- 大規模展開: クラウドGPU + INT8
```

### 3. パラメータチューニング

```yaml
# 精度重視の設定例
index:
  chunk_size: 1024
  chunk_overlap: 100
rag:
  temperature: 0.0
  similarity_top_k: 5

# 速度重視の設定例
index:
  chunk_size: 256
  chunk_overlap: 25
rag:
  temperature: 0.1
  similarity_top_k: 2
```

## 🎯 成功基準

### 最低基準（MVP合格ライン）

- [ ] インデックス構築が完了
- [ ] 両モデルで200問実行完了
- [ ] 評価指標が全て算出
- [ ] 結果がJSON/CSVで保存

### 目標基準（実用レベル）

- [ ] F1スコア平均 > 0.70
- [ ] 応答時間平均 < 3秒
- [ ] BLEU-1スコア > 0.60
- [ ] エラー率 < 5%

### 優秀基準（本番投入可能）

- [ ] F1スコア平均 > 0.80
- [ ] 応答時間平均 < 2秒
- [ ] ROUGE-Lスコア > 0.75
- [ ] カテゴリ別精度の均等性

## 📚 参考情報

### 技術スタック

- **LLM**: Gemma 3 (2B)
- **RAGフレームワーク**: LlamaIndex 0.9.x
- **ベクトルDB**: ChromaDB 0.4.x
- **埋め込み**: multilingual-e5-large
- **実行環境**: Ollama

### 関連リンク

- [Gemma Model Card](https://ai.google.dev/gemma)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Ollama Docs](https://ollama.ai/docs)

### 論文・記事

- RAG Survey Papers
- Gemma Technical Report
- Quantization Methods (INT4/INT8)
- Evaluation Metrics for NLG

---

**作成日**: 2025年11月19日  
**バージョン**: 1.0.0  
**ライセンス**: MIT
