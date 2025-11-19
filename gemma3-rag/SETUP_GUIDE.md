# Gemma3 RAG KasenSabo MVP - セットアップガイド

## 🎯 クイックスタート（10分で始める）

### ステップ1: Ollama準備（5分）

```powershell
# 1. Ollamaがインストールされているか確認
ollama --version

# 2. Gemma 3モデルをダウンロード（各1-2分）
ollama pull gemma:2b-instruct-q4_K_M
ollama pull gemma:2b-instruct-q8_0

# 3. 動作確認
ollama run gemma:2b-instruct-q4_K_M "こんにちは"
# Ctrl+Dで終了
```

### ステップ2: Python環境（3分）

```powershell
# プロジェクトディレクトリに移動
cd C:\Users\yasun\LangChain\learning-langchain\gemma3-rag

# 仮想環境作成
python -m venv venv

# 仮想環境をアクティベート
.\venv\Scripts\Activate.ps1

# 依存パッケージをインストール
pip install -r requirements.txt

# NLTKデータのダウンロード
python -c "import nltk; nltk.download('punkt')"
```

### ステップ3: インデックス構築（5-10分）

```powershell
# インデックスを構築
python scripts/build_index.py

# ✓ Loaded XX documents と表示されればOK
```

### ステップ4: 動作確認（1分）

```powershell
# テスト実行
python scripts/run_rag.py

# モデルを選択（1: INT4, 2: INT8）
# 3つのテスト質問で動作確認
```

---

## 📋 詳細セットアップ手順

### A. システム要件

#### 必須要件
- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.9以上（推奨: 3.10 or 3.11）
- **RAM**: 最低8GB（推奨: 16GB以上）
- **ストレージ**: 10GB以上の空き容量

#### 推奨要件
- **GPU**: 必須ではないが、あると高速化
- **CPU**: 4コア以上

### B. Ollamaのインストール

#### Windows

```powershell
# 公式サイトからインストーラーをダウンロード
# https://ollama.ai/download

# または、wingetを使用
winget install Ollama.Ollama

# インストール後、PowerShellで確認
ollama --version
```

#### macOS

```bash
# Homebrewでインストール
brew install ollama

# または公式サイトからダウンロード
# https://ollama.ai/download
```

#### Linux

```bash
# インストールスクリプトを実行
curl -fsSL https://ollama.ai/install.sh | sh
```

### C. Gemma 3モデルのダウンロード

```powershell
# INT4モデル（約1.5GB）
ollama pull gemma:2b-instruct-q4_K_M

# INT8モデル（約2.5GB）
ollama pull gemma:2b-instruct-q8_0

# ダウンロード済みモデルの確認
ollama list

# 出力例:
# NAME                           ID              SIZE      MODIFIED
# gemma:2b-instruct-q4_K_M      abc123def...    1.5 GB    2 minutes ago
# gemma:2b-instruct-q8_0        def456ghi...    2.5 GB    1 minute ago
```

### D. Python環境のセットアップ

#### 1. 仮想環境の作成

```powershell
# プロジェクトディレクトリで実行
cd C:\Users\yasun\LangChain\learning-langchain\gemma3-rag

# 仮想環境を作成
python -m venv venv

# アクティベート（Windows PowerShell）
.\venv\Scripts\Activate.ps1

# アクティベート（Windows CMD）
.\venv\Scripts\activate.bat

# アクティベート（Linux/macOS）
source venv/bin/activate

# (venv) が表示されることを確認
```

#### 2. 依存パッケージのインストール

```powershell
# pip のアップグレード
python -m pip install --upgrade pip

# 依存パッケージをインストール
pip install -r requirements.txt

# インストール確認
pip list | Select-String "llama-index|chromadb|ollama"
```

#### 3. NLTKデータのダウンロード

```powershell
# 対話的にダウンロード
python -c "import nltk; nltk.download('punkt')"

# または、Pythonスクリプト内で
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
print('✓ NLTK data downloaded')
"
```

### E. データの準備確認

```powershell
# 知識ベースデータの確認
ls data/kasensabo_knowledge_base/

# 出力例:
# 00_training_overview_2025.md
# 01_training_chousa_2025.md
# ...

# ベンチマーク質問の確認
ls questions/

# 出力例:
# bench_questions_200.json
```

### F. インデックスの構築

```powershell
# インデックス構築スクリプトを実行
python scripts/build_index.py

# 実行中の表示例:
# ==================================================
# Gemma3 RAG - Index Building
# ==================================================
# 
# [1] Loading documents from: data/kasensabo_knowledge_base
# ✓ Loaded 8 documents
# 
# [2] Initializing embedding model: intfloat/multilingual-e5-large
# ✓ Embedding model loaded
# ✓ Chunk size: 512, Overlap: 50
# 
# [3] Initializing ChromaDB at: index/chroma_index
# ✓ ChromaDB initialized
# 
# [4] Building vector index...
# [進行状況バー]
# ✓ Index built successfully
# 
# ==================================================
# ✅ Index building completed successfully!
```

**所要時間**: 5〜10分（データ量とマシン性能による）

**トラブルシューティング**:
- メモリ不足エラー → `config.yaml`の`chunk_size`を256に減らす
- ChromaDBエラー → `index/`ディレクトリを削除して再実行

---

## 🧪 動作確認

### 1. 単一クエリテスト

```powershell
# RAG実行スクリプトを起動
python scripts/run_rag.py

# プロンプトに従って操作:
# Available models:
# 1. gemma:2b-instruct-q4_K_M (INT4)
# 2. gemma:2b-instruct-q8_0 (INT8)
# 
# Select model (1 or 2): 1
# 
# Initializing RAG system with gemma:2b-instruct-q4_K_M...
# ✓ Embedding model initialized: intfloat/multilingual-e5-large
# ✓ Index loaded from: index/chroma_index
# ✓ Query engine created with model: gemma:2b-instruct-q4_K_M
# 
# Running test queries...
# [Query 1] 河川の計画高水流量とは何ですか？
# Response time: 2.34s
# Response: 計画高水流量は...
```

### 2. 評価機能のテスト

```powershell
# 評価スクリプトのデモを実行
python scripts/evaluate.py

# 出力例:
# ==================================================
# RAG Evaluator Demo
# ==================================================
# 
# [Individual Evaluations]
# Case 1:
#   exact_match: 0
#   f1_score: 0.7500
#   bleu_1: 0.6234
#   rouge1_f: 0.7123
#   response_time: 2.5000
```

---

## 🚀 本格実行

### フルベンチマークの実行

```powershell
# ベンチマークスクリプトを実行
python scripts/run_benchmark.py

# 実行時間: 約30〜60分
```

**実行内容**:
1. 200問の質問を読み込み
2. INT4モデルで全質問を実行（約15-30分）
3. INT8モデルで全質問を実行（約15-30分）
4. 評価指標を計算
5. 結果をJSON/CSV形式で保存
6. モデル間の比較表を生成

**結果の保存先**:
- `results/gemma_2b-instruct-q4_K_M_benchmark_*.json`
- `results/gemma_2b-instruct-q8_0_benchmark_*.json`
- `results/model_comparison_*.csv`

---

## 🔧 トラブルシューティング

### エラー別対処法

#### 1. `ModuleNotFoundError: No module named 'XXX'`

```powershell
# 仮想環境がアクティブか確認
# プロンプトに (venv) が表示されているか？

# パッケージを再インストール
pip install -r requirements.txt --force-reinstall
```

#### 2. `ollama.ConnectionError: Could not connect to Ollama`

```powershell
# Ollamaサービスが起動しているか確認
ollama list

# エラーが出る場合、Ollamaを再起動
# Windows: タスクマネージャーでOllamaを終了→再起動
# macOS: Ollamaアプリを再起動
```

#### 3. `FileNotFoundError: Index not found`

```powershell
# インデックスを再構築
python scripts/build_index.py
```

#### 4. メモリ不足エラー

```yaml
# config.yaml を編集
index:
  chunk_size: 256  # 512 → 256に変更
  chunk_overlap: 25  # 50 → 25に変更

rag:
  similarity_top_k: 2  # 3 → 2に変更
```

#### 5. CUDA/GPU関連エラー

```yaml
# config.yaml を編集してCPUモードに
embedding:
  device: "cpu"  # "cuda" → "cpu"
```

---

## 📊 設定のカスタマイズ

### `config.yaml` の主要設定

```yaml
# チャンク分割の調整
index:
  chunk_size: 512        # 大きい→精度↑、処理↓
  chunk_overlap: 50      # 大きい→連続性↑

# RAGパラメータ
rag:
  temperature: 0.1       # 0に近い→決定的、1に近い→創造的
  similarity_top_k: 3    # 参照する文書数

# ベンチマーク設定
benchmark:
  batch_size: 10         # メモリ節約したい場合は小さく
  save_interval: 50      # 中間保存の頻度
```

---

## 🎓 次のステップ

1. ✅ **基本動作確認完了**
2. 🔄 **カスタム質問でテスト**: 自分の質問を追加
3. 📊 **フルベンチマーク実行**: 200問で評価
4. 🔬 **パラメータ最適化**: temperatureやtop_kを調整
5. 📈 **結果分析**: カテゴリ別の精度を確認

---

## 💡 ヒント

- **高速化**: GPU利用、モデルをINT4に統一
- **精度向上**: chunk_sizeを大きく、INT8モデルを使用
- **デバッグ**: 少数の質問でまずテスト
- **バックアップ**: indexディレクトリは構築に時間がかかるので保存推奨

---

**問題が解決しない場合は、エラーメッセージをコピーしてIssueを作成してください！**
