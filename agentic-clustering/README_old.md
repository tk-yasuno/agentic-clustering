# 橋梁維持管理クラスタリング MVP 🌉

**山口県を対象にした「橋梁維持管理クラスタリングMVP」**

山口県の公開データ（橋梁管理一覧表・財政状況資料・人口統計）を基盤に、橋梁の健全度・補修履歴・財政力指数・人口動態を組み合わせて「補修優先度の高い群」を抽出するシステムです。

---

## 🎯 プロジェクト概要

### 目的
- 山口県内の橋梁を維持管理困難度でクラスタリング
- 補修優先度の高い群を定量的に抽出
- 財政力指数や人口動態を組み合わせた合理的な補修計画の支援

### 主な機能
- **データ前処理**: 橋梁データ、財政データ、人口統計の統合
- **クラスタリング**: KMeansアルゴリズムによる群分け
- **最適化**: シルエットスコアによる最適クラスタ数の決定
- **可視化**: PCA散布図、ヒートマップ、レーダーチャート、箱ひげ図
- **レポート生成**: クラスタごとの特性分析レポート

---

## 📂 プロジェクト構造

```
agentic-clustering/
│
├── data/                                    # データフォルダ
│   ├── YamaguchiPrefBridgeListOpen251122_154891.xlsx   # 橋梁管理一覧表
│   ├── 全市町村の主要財政指標_000917808.xlsx          # 財政力指数
│   └── 市区町村別年齢階級別人口_2304ssnen.xlsx       # 人口統計
│
├── output/                                  # 出力フォルダ（自動生成）
│   ├── processed_bridge_data.csv            # 前処理済みデータ
│   ├── cluster_results.csv                  # クラスタリング結果
│   ├── cluster_summary.csv                  # クラスタサマリー
│   ├── cluster_pca_scatter.png              # PCA散布図
│   ├── cluster_heatmap.png                  # ヒートマップ
│   ├── cluster_radar.png                    # レーダーチャート
│   ├── cluster_distribution.png             # 分布図
│   ├── feature_boxplots.png                 # 箱ひげ図
│   └── cluster_report.txt                   # 分析レポート
│
├── config.py                                # 設定ファイル
├── data_preprocessing.py                    # データ前処理スクリプト
├── clustering.py                            # クラスタリングメインスクリプト
├── visualization.py                         # 可視化スクリプト
├── requirements.txt                         # 依存パッケージ
└── README.md                                # このファイル
```

---

## 🚀 セットアップ手順

### 1. 必要なパッケージのインストール

```powershell
pip install -r requirements.txt
```

**主な依存パッケージ:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- openpyxl >= 3.1.0
- japanize-matplotlib >= 1.1.3

### 2. データの配置

`data/` フォルダに以下のファイルを配置してください:
- 橋梁管理一覧表（Excel形式）
- 財政力指数データ（Excel形式）
- 人口統計データ（Excel形式）

---

## 💻 使い方

### ステップ1: データ前処理

```powershell
python data_preprocessing.py
```

**処理内容:**
- 橋梁データの読み込みと橋齢計算
- 健全度スコアの数値化
- 補修優先度の算出
- 財政力指数・人口統計との統合
- `output/processed_bridge_data.csv` に保存

### ステップ2: クラスタリング実行

```powershell
python clustering.py
```

**処理内容:**
- 特徴量の標準化
- PCAによる次元削減
- シルエットスコアで最適クラスタ数を探索
- KMeansクラスタリングの実行
- クラスタごとの特性分析
- `output/cluster_results.csv` と `output/cluster_summary.csv` に保存

### ステップ3: 結果の可視化

```powershell
python visualization.py
```

**処理内容:**
- PCA 2次元散布図の生成
- クラスタ特性ヒートマップ
- レーダーチャート
- クラスタ分布の棒グラフ
- 特徴量の箱ひげ図
- テキストレポートの生成

---

## 📊 出力ファイルの説明

### CSV形式
- **processed_bridge_data.csv**: 前処理済みの統合データ
- **cluster_results.csv**: 各橋梁のクラスタ割り当て結果
- **cluster_summary.csv**: クラスタごとの特徴量平均値

### 可視化画像
- **cluster_pca_scatter.png**: PCAで次元削減したクラスタの散布図
- **cluster_heatmap.png**: クラスタごとの特徴量比較ヒートマップ
- **cluster_radar.png**: クラスタ特性のレーダーチャート
- **cluster_distribution.png**: クラスタごとの橋梁数分布
- **feature_boxplots.png**: 特徴量のクラスタ別箱ひげ図

### レポート
- **cluster_report.txt**: クラスタ分析の詳細レポート（リスク評価含む）

---

## 🔑 特徴量の説明

| 特徴量 | 説明 | 備考 |
|--------|------|------|
| `bridge_age` | 橋齢（架設からの年数） | 現在年 - 架設年 |
| `condition_score` | 健全度スコア | Ⅰ=1, Ⅱ=2, Ⅲ=3, Ⅳ=4 |
| `maintenance_priority` | 補修優先度 | 橋齢 × 健全度スコア |
| `population_decline` | 人口減少率（%） | 市町村ごと |
| `aging_rate` | 高齢化率（%） | 65歳以上人口割合 |
| `fiscal_index` | 財政力指数 | 市町村の財政力 |

---

## 🎨 クラスタ解釈の目安

### 🔴 高リスク群
- 高橋齢 (50年以上)
- 健全度低下 (Ⅲ・Ⅳ)
- 高補修優先度 (100以上)
- 人口減少 (15%以上)
- 高齢化 (35%以上)
- 財政力弱 (0.5未満)

**3つ以上該当** → 高リスク

### 🟡 中リスク群
**2つ該当** → 中リスク

### 🟢 低リスク群
**1つ以下** → 低リスク

---

## 📈 期待される成果

1. **定量的な群分け**: 山口県内の橋梁を維持管理困難度でクラスタリング
2. **優先順位付け**: 補修優先度の高い群を明確化
3. **合理的な意思決定**: 財政力や人口動態を考慮した補修計画の策定
4. **可視化**: ダッシュボードで地図表示＋群別特性の直感的な提示

---

## ⚙️ カスタマイズ方法

### クラスタ数の調整
`config.py` で調整可能:
```python
MIN_CLUSTERS = 2  # 最小クラスタ数
MAX_CLUSTERS = 8  # 最大クラスタ数
```

### 特徴量の追加・変更
`config.py` の `FEATURE_COLUMNS` を編集:
```python
FEATURE_COLUMNS = [
    "bridge_age",
    "condition_score",
    "maintenance_priority",
    "population_decline",
    "aging_rate",
    "fiscal_index",
    # 新しい特徴量を追加可能
]
```

### データファイル名の変更
`config.py` で各データファイルのパスを指定:
```python
BRIDGE_DATA_FILE = os.path.join(DATA_DIR, "your_bridge_data.xlsx")
FISCAL_DATA_FILE = os.path.join(DATA_DIR, "your_fiscal_data.xlsx")
POPULATION_DATA_FILE = os.path.join(DATA_DIR, "your_population_data.xlsx")
```

---

## 🛠️ トラブルシューティング

### エラー: `ファイルが見つかりません`
- `data/` フォルダに必要なExcelファイルが配置されているか確認
- `config.py` のファイルパスが正しいか確認

### エラー: `列が見つかりません`
- Excelファイルの列名が想定と異なる可能性
- `data_preprocessing.py` の列名推定ロジックを調整

### 日本語表示が文字化けする
```powershell
pip install japanize-matplotlib
```

---

## 📝 今後の拡張案

1. **地図プロット**: 緯度経度情報を使った地図上の可視化
2. **時系列分析**: 過去データとの比較による劣化予測
3. **Tableauダッシュボード**: インタラクティブな可視化
4. **予測モデル**: 将来の補修コスト予測
5. **最適化アルゴリズム**: 予算制約下での補修計画最適化

---

## 📞 お問い合わせ

プロジェクトに関するご質問やフィードバックは、GitHubのIssueでお願いします。

---

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**Developed with ❤️ for Bridge Maintenance Optimization**
