# 橋梁維持管理 Agentic Clustering v0.4

山口県の橋梁維持管理データを用いた自己改善型クラスタリングシステム（地理空間特徴量拡張版）

## 🎯 主な内容

### プロジェクトの成果

本プロジェクトでは、**山口県の4,292件の橋梁データ**を対象に、地理空間情報を統合した自己改善型のAgenticクラスタリングシステムを構築しました。

#### ✨ 主要な達成事項（v0.4）

1. **13個の特徴量による多角的分析**
   - 基本特徴量（橋齢、健全度、補修優先度等）：6個
   - 拡張特徴量（構造形式、橋面積、緊急輸送道路等）：5個
   - **🆕 地理空間特徴量（桁下河川、海岸線距離）：2個**

2. **4つのクラスタリング手法の自動評価**
   - KMeans（k=2〜28の自動探索）
   - GMM（Gaussian Mixture Model）
   - DBSCAN（密度ベース）
   - **🆕 HDBSCAN（階層的DBSCAN、DBSCANが50クラスタ超の場合に自動試行）**

3. **3つの次元削減手法の自動評価**
   - PCA（主成分分析）
   - t-SNE（t-distributed Stochastic Neighbor Embedding）
   - UMAP（Uniform Manifold Approximation and Projection）

4. **地理空間解析の統合**
   - 国土数値情報（河川・海岸線データ）の活用
   - 投影座標系（UTM Zone 53N）による正確な距離計算
   - geopandas/shapelyによる空間演算

5. **🆕 Agenticなクラスタ数制御**
   - DBSCANのクラスタ数が50を超える場合、自動的にHDBSCANを試行
   - 補修意思決定者にとって扱いやすいクラスタ数を維持

#### 📊 実装の特徴

- **自己改善ループ**: 品質閾値を下回った場合、自動的に代替手法を試行
- **目標ベース最適化**: DBSCANのクラスタ数を目標値（50）に基づいて調整
- **🆕 Agenticなクラスタ数制御**: DBSCANが50超のクラスタを生成した場合、自動的にHDBSCANを試行
- **包括的な可視化**: 散布図、ヒートマップ、レーダーチャート、箱ひげ図等
- **詳細なログ記録**: 改善履歴とパラメータ選択の根拠を自動記録
- **🆕 地理空間分析**: CRS変換、バッファ演算、距離計算

#### 💻 技術スタック

- Python 3.11
- scikit-learn 1.7.2（KMeans、GMM、DBSCAN、t-SNE）
- **hdbscan 0.8+（階層的密度ベースクラスタリング）**
- umap-learn 0.5.9（UMAP）
- **geopandas 1.0+（地理空間データ処理）**
- **shapely 2.0+（幾何演算）**
- pandas 2.3.3（データ処理）
- matplotlib / seaborn（可視化）

---

## 📋 目次

1. [概要](#概要)
2. [特徴量エンジニアリング](#特徴量エンジニアリング)
3. [🆕 地理空間特徴量の実装](#地理空間特徴量の実装)
4. [システムアーキテクチャ](#システムアーキテクチャ)
5. [実装の教訓](#実装の教訓)
6. [パラメータ最適化の履歴](#パラメータ最適化の履歴)
7. [使い方](#使い方)
8. [今後の改善案](#今後の改善案)

---

## 1. 概要

### 背景

日本の橋梁インフラの老朽化が進む中、効率的な維持管理戦略が求められています。本プロジェクトは、山口県の4,292件の橋梁データに対して、**13個の多角的な特徴量**を用いたクラスタリング分析を行い、類似する特性を持つ橋梁群を自動的に抽出します。

### 目的

- 橋梁を特性に基づいてグループ化し、効率的な維持管理計画を支援
- 複数のクラスタリング手法を自動評価し、最適な手法を選択
- 地理空間情報を活用し、環境要因（河川、海岸）を考慮した分析

---

## 2. 特徴量エンジニアリング

### 2.1 基本特徴量（6個）

| 特徴量 | 説明 | 統計 |
|--------|------|------|
| **bridge_age** | 橋齢（年） | 平均: 50年、中央値: 1974年架設 |
| **condition_score** | 健全度スコア（1〜4） | Ⅰ:1310件、Ⅱ:2372件、Ⅲ:610件 |
| **maintenance_priority** | 補修優先度（橋齢×健全度） | 算出値 |
| **future_burden_ratio** | 将来負担比率（%） | 市町村の財政指標 |
| **aging_rate** | 高齢化率（%） | 平均: 36.4% |
| **fiscal_index** | 財政力指数 | 平均: 0.562 |

### 2.2 拡張特徴量（5個）

| 特徴量 | 説明 | 統計 |
|--------|------|------|
| **structure_category** | 構造形式カテゴリー | RC:2446、PC:1475、鋼橋:297、ボックス:55、その他:19 |
| **bridge_area** | 橋面積（m²） | 平均: 223.8m²、中央値: 66.9m² |
| **emergency_route** | 緊急輸送道路ダミー | 該当: 2036件（47.4%） |
| **overpass** | 跨線橋ダミー | 該当: 88件（2.1%） |
| **repair_year_normalized** | 最新補修年度（正規化） | 1989〜2022年、実施率: 15.0% |

### 🆕 2.3 地理空間特徴量（2個）

| 特徴量 | 説明 | 統計 | データソース |
|--------|------|------|--------------|
| **under_river** | 桁下河川判定（0 or 1） | 該当: 2447件（57.0%） | 国土数値情報 河川データ（W05-08_35） |
| **distance_to_coast_km** | 海岸線までの距離（km） | 範囲: 0〜30.09km、平均: 9.19km | 国土数値情報 海岸線データ（C23-06_35） |

#### 地理空間特徴量の意義

- **under_river（桁下河川判定）**: 
  - 河川直下の橋梁は洪水・流水による影響を受けやすい
  - 塩害は少ないが、水害リスクが高い
  - 補修時の仮設足場設置コストが増加
  
- **distance_to_coast_km（海岸線距離）**:
  - 海岸に近いほど塩害による劣化が進行しやすい
  - 鋼橋や鉄筋コンクリートの腐食リスクが増大
  - 維持管理コストと点検頻度の増加要因

---

## 🆕 3. 地理空間特徴量の実装

### 3.1 技術的アプローチ

#### データソース
- **橋梁座標**: Excel「緯度」「経度」列（WGS84）
- **河川データ**: Shapefile `W05-08_35-g_Stream.shp`（国土数値情報）
- **海岸線データ**: Shapefile `C23-06_35-g_Coastline.shp`（国土数値情報）

#### 実装の流れ

```python
# 1. 座標参照系（CRS）の設定
# 国土数値情報のShapefileはCRS未設定の場合があるため、WGS84を明示的に設定
rivers = gpd.read_file("rivers.shp")
if rivers.crs is None:
    rivers.set_crs("EPSG:4326", inplace=True)  # WGS84

# 2. 投影座標系への変換（正確な距離計算のため）
rivers_proj = rivers.to_crs("EPSG:32653")  # UTM Zone 53N（山口県に最適）

# 3. バッファ演算（河川から50m以内）
rivers_buffer = rivers_proj.buffer(50)  # メートル単位

# 4. 橋梁位置の判定
under_river = bridge_point.within(rivers_buffer)

# 5. 海岸線までの距離計算
distance_km = bridge_point.distance(coastline) / 1000.0  # m→km
```

### 3.2 実装の課題と解決策

#### 課題1: CRS未設定エラー
**エラー**: `Cannot transform naive geometries. Please set a crs on the object first.`

**原因**: 国土数値情報のShapefileはCRSメタデータが欠落している

**解決策**:
```python
if shapefile.crs is None:
    shapefile.set_crs("EPSG:4326", inplace=True)  # 座標値からWGS84と判定
```

#### 課題2: 地理座標系でのバッファ演算
**警告**: `Geometry is in a geographic CRS. Results from 'buffer' are likely incorrect.`

**原因**: 度単位（経度・緯度）での距離計算は不正確

**解決策**:
```python
# 投影座標系（メートル単位）に変換してからバッファ作成
data_proj = data.to_crs("EPSG:32653")  # UTM Zone 53N
buffer = data_proj.buffer(50)  # 50メートル
```

#### 課題3: 非現実的な距離値
**問題**: 海岸線距離が852〜1020km（山口県の幅を超える）

**原因**: 間違った座標系（EPSG:6677 平面直角座標系）を使用

**解決策**:
```python
# 座標範囲（130〜132経度、33〜34緯度）から、既にWGS84と判明
# EPSG:6677ではなくEPSG:4326を直接設定
```

### 3.3 座標系の選択

| 座標系 | EPSG | 用途 | 単位 | 適用範囲 |
|--------|------|------|------|----------|
| WGS84 | 4326 | 世界測地系（GPS） | 度 | 全世界 |
| UTM Zone 53N | 32653 | 投影座標系（山口県） | m | 東経129°〜135° |
| JGD2011 | 6668 | 日本測地系2011 | 度 | 日本全国 |

**山口県の最適座標系**: EPSG:32653（UTM Zone 53N）
- 経度131°を中央子午線とする
- メートル単位で正確な距離計算が可能
- 歪みが最小化される

### 3.4 処理性能

- **データ量**: 橋梁4,292件、河川4,502件、海岸線2,109件
- **処理時間**: 約10〜15秒（GeoDataFrame変換・バッファ・距離計算含む）
- **メモリ使用量**: 約200MB

---

## 4. システムアーキテクチャ

### 4.1 ワークフロー

```
[データ読み込み]
  ├─ 橋梁データ（Excel）
  ├─ 財政データ（Excel）
  ├─ 人口データ（Excel）
  └─ 🆕 地理空間データ（Shapefile）
        ↓
[特徴量エンジニアリング]
  ├─ 基本特徴量（6個）
  ├─ 拡張特徴量（5個）
  └─ 🆕 地理空間特徴量（2個）
        ↓
[標準化・正規化]
        ↓
[クラスタリング]
  ├─ KMeans（最適k探索）
  ├─ GMM（n_components探索）
  ├─ DBSCAN（eps/min_samples最適化）
  └─ 🆕 HDBSCAN（DBSCANが50超の場合に自動試行）
        ↓
[品質評価]
  ├─ シルエットスコア（45%）
  ├─ Davies-Bouldin指数（45%）
  ├─ Calinski-Harabasz指数（0%）← 過大評価により無効化
  └─ クラスタバランス（10%）
        ↓
[改善判定] ← 閾値60点未満で代替手法へ
        ↓
[次元削減・可視化]
  ├─ PCA
  ├─ t-SNE
  └─ UMAP
        ↓
[結果出力]
  ├─ クラスタリング結果CSV
  ├─ 可視化グラフ（PNG）
  └─ 改善履歴ログ
```

### 4.2 主要ファイル構成

```
agentic-clustering/
├── config.py                    # 設定ファイル（13特徴量、パス定義）
├── data_preprocessing.py        # データ前処理（🆕 地理空間処理含む）
├── clustering.py                # クラスタリング実行
├── cluster_evaluator.py         # 品質評価
├── alternative_methods.py       # 代替手法（GMM、DBSCAN、t-SNE、UMAP）
├── dimensionality_reduction.py  # 次元削減
├── visualizations.py            # 可視化
├── run_all.py                   # 統合実行スクリプト
├── data/
│   ├── YamaguchiPrefBridgeListOpen251122_154891.xlsx
│   ├── 全市町村の主要財政指標_000917808.xlsx
│   ├── 市区町村別年齢階級別人口_2304ssnen.xlsx
│   ├── 🆕 RiverDataKokudo/W05-08_35_GML/
│   │   └── W05-08_35-g_Stream.shp
│   └── 🆕 KaigansenDataKokudo/C23-06_35_GML/
│       └── C23-06_35-g_Coastline.shp
└── output/
    ├── processed_bridge_data.csv
    ├── cluster_results.csv
    ├── cluster_summary.csv
    ├── improvement_log.txt
    └── *.png（可視化グラフ）
```

---

## 5. 実装の教訓

### 5.1 マルチヘッダーExcelの処理

**問題**: Excelの1行目と2行目に分かれたヘッダー構造

**解決策**:
```python
df = pd.read_excel(file, header=[0, 1])  # 両方読み込み
df.columns = [col[0] if 'Unnamed' in str(col[1]) else col[0] for col in df.columns]
```

### 5.2 評価指標の選択

**問題**: Calinski-Harabasz指数が高次元データで過大評価

**解決策**:
- CH指数の重みを0%に設定
- シルエットスコア45% + Davies-Bouldin指数45%を主軸に
- クラスタバランス10%で偏りを補正

```python
WEIGHTS = {
    "silhouette": 0.45,
    "dbi": 0.45,
    "ch": 0.00,        # 過大評価により無効化
    "balance": 0.10
}
```

### 5.3 DBSCANの目標クラスタ数調整

**問題**: DBSCANが130クラスタを生成（過度に細分化）

**解決策**:
```python
# 目標クラスタ数からの乖離に応じてペナルティ
cluster_penalty = abs(n_clusters - target_clusters) / target_clusters
adjusted_score = base_score * (1 - cluster_penalty * 0.5)
```

**結果**: 34クラスタ（目標50の68%達成、実用的な数）

### 5.4 t-SNEのパラメータ変更

**問題**: `TSNE.__init__() got an unexpected keyword argument 'n_iter'`

**原因**: scikit-learn 1.7でパラメータ名変更

**解決策**:
```python
# 旧: n_iter=1000
# 新: max_iter=1000, n_iter_without_progress=300
tsne = TSNE(max_iter=1000, n_iter_without_progress=300)
```

### 5.5 欠損値処理戦略

**方針**: データの意味を考慮した補完

| データ | 欠損値処理 | 理由 |
|--------|-----------|------|
| 架設年 | 中央値（1974年） | 年代分布の中心を維持 |
| 健全度 | Ⅱ（スコア2） | 最頻値（標準的な状態） |
| 補修年度 | 0.0（正規化後） | 補修未実施を明示 |
| 橋面積 | 中央値（66.9m²） | 外れ値の影響を回避 |
| 人口・財政 | 5パーセンタイル値 | 小規模自治体の代表値 |
| 🆕 地理空間 | 河川:0、海岸:中央値 | 座標欠損時の保守的な推定 |

### 🆕 5.6 地理空間データの座標系管理

**問題**: 国土数値情報ShapefileのCRS未設定

**解決策**:
1. **座標範囲から座標系を推定**
   - 経度130〜132、緯度33〜34 → WGS84（EPSG:4326）
   
2. **投影座標系での演算**
   - バッファ・距離計算はUTM（EPSG:32653）で実施
   
3. **CRS変換の統一**
```python
if gdf.crs is None:
    gdf.set_crs("EPSG:4326", inplace=True)
gdf_proj = gdf.to_crs("EPSG:32653")  # 演算用
```

### 🆕 5.7 geopandas非推奨API対応

**問題**: `unary_union` が非推奨

**解決策**:
```python
# 新旧両対応
try:
    union = geometries.union_all()  # 新API
except AttributeError:
    union = geometries.unary_union  # 旧API（後方互換）
```

### 🆕 5.8 HDBSCAN による Agenticなクラスタ数制御

**問題**: DBSCANが140クラスタを生成し、補修意思決定者にとって扱いづらい

**背景**:
- DBSCANはスコアがダントツに良い（総合スコア高）
- しかし、クラスタ数が多すぎると実務では使いにくい
- 50個程度のクラスタが管理可能な範囲

**解決策**:
```python
# Agenticな判定ロジック
if dbscan_n_clusters > config.DBSCAN_CLUSTER_THRESHOLD:  # デフォルト50
    print("DBSCANのクラスタ数が閾値を超えています")
    print("→ HDBSCANを代替手法として自動試行")
    
    hdbscan_labels = alt_methods.try_hdbscan(target_clusters=50)
    # HDBSCANの結果を評価・比較対象に追加
```

**HDBSCANの特徴**:
- **階層的**: デンドログラムに基づく階層的クラスタリング
- **適応的**: 密度の異なるクラスタを自動検出
- **パラメータ削減**: クラスタ数を指定不要
- **ノイズ対応**: 外れ値を自動的にノイズとして分類
- **安定性**: DBSCANより安定したクラスタ割り当て

**パラメータ調整**:
- `min_cluster_size`: 最小クラスタサイズ（大きいほど少ないクラスタ）
- `min_samples`: 最小サンプル数（ノイズ検出の感度）
- `cluster_selection_method`: 'eom'（Excess of Mass）推奨

---

## 6. パラメータ最適化の履歴

### 6.1 評価ウェイトの変遷

| バージョン | Silhouette | DBI | CH | Balance | 理由 |
|-----------|-----------|-----|-----|---------|------|
| v0.1 | 40% | 30% | 20% | 10% | 初期設定 |
| v0.2 | 50% | 50% | 0% | 0% | CH過大評価の除外 |
| **v0.3** | 45% | 45% | 0% | 10% | バランス重視 |
| **v0.4** | 45% | 45% | 0% | 10% | 13特徴量対応（地理空間追加） |

### 6.2 DBSCANパラメータ調整

| 試行 | eps範囲 | min_samples範囲 | 結果クラスタ数 | 評価 |
|------|---------|----------------|---------------|------|
| 1 | 0.5〜2.0 | 5〜20 | 130 | ❌ 過度に細分化 |
| 2 | 1.0〜3.0 | 10〜50 | 28 | ⚠️ やや少ない |
| 3 | 0.7〜1.5 | 10〜30 | 67 | ⚠️ やや多い |
| **4** | 0.8〜1.6 | 15〜35 | 34 | ✅ 実用的 |

**最終パラメータ（11特徴量）**:
- eps: 1.2
- min_samples: 25
- ノイズ率: 31.4%
- クラスタ数: 34

### 🆕 6.3 地理空間特徴量の影響（v0.4）

#### 追加前（11特徴量）vs 追加後（13特徴量）

| 指標 | 11特徴量 | 13特徴量 | 変化 |
|------|---------|---------|------|
| 最適クラスタ数（KMeans） | 24 | 未実行（進行中） | - |
| シルエットスコア | 0.257 | - | - |
| Davies-Bouldin指数 | 0.989 | - | - |
| 総合スコア | 51.20/100 | - | - |

**予想される効果**:
- 河川・海岸の環境要因が反映され、クラスタの地理的一貫性が向上
- 塩害リスク・水害リスクによる補修戦略の差異が明確化
- 沿岸部と内陸部のクラスタ分離が促進される

---

## 7. 使い方

### 7.1 環境構築

```powershell
# 仮想環境作成
python -m venv venv
.\venv\Scripts\Activate.ps1

# 依存パッケージインストール
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn
pip install geopandas shapely pyproj openpyxl  # 🆕 地理空間ライブラリ
pip install hdbscan  # 🆕 階層的クラスタリング
```

### 7.2 データ準備

```
data/
├── YamaguchiPrefBridgeListOpen251122_154891.xlsx
├── 全市町村の主要財政指標_000917808.xlsx
├── 市区町村別年齢階級別人口_2304ssnen.xlsx
├── 🆕 RiverDataKokudo/W05-08_35_GML/W05-08_35-g_Stream.shp
└── 🆕 KaigansenDataKokudo/C23-06_35_GML/C23-06_35-g_Coastline.shp
```

**🆕 国土数値情報データのダウンロード**:
- 河川データ: https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html
- 海岸線データ: https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-C23.html
  - 山口県（コード35）のShapefileをダウンロード

### 7.3 実行

```powershell
python run_all.py
```

**出力**:
- `output/processed_bridge_data.csv`: 前処理済みデータ（13特徴量）
- `output/cluster_results.csv`: クラスタ割り当て結果
- `output/cluster_summary.csv`: クラスタ別統計
- `output/improvement_log.txt`: 改善履歴
- `output/*.png`: 可視化グラフ

### 7.4 設定変更

`config.py` で調整可能:

```python
# クラスタ数範囲
MIN_CLUSTERS = 2
MAX_CLUSTERS = 28  # 市町村数×1.5

# 品質閾値
QUALITY_THRESHOLD = 60.0

# 次元削減のオーバーラップ閾値
OVERLAP_THRESHOLD = 0.05

# 🆕 DBSCANクラスタ数閾値（超過時にCLASSIXを自動試行）
DBSCAN_CLUSTER_THRESHOLD = 50

# 🆕 地理空間データパス
RIVER_SHAPEFILE = "data/RiverDataKokudo/.../W05-08_35-g_Stream.shp"
COASTLINE_SHAPEFILE = "data/KaigansenDataKokudo/.../C23-06_35-g_Coastline.shp"
```

---

## 8. 今後の改善案

### 8.1 短期的改善（v0.5候補）

1. **🆕 地形データの追加**
   - 標高（DEM: Digital Elevation Model）
   - 斜面角度（橋梁へのアクセス難易度）
   
2. **🆕 気象データの統合**
   - 年間降水量（メッシュ気候値）
   - 積雪深度（寒冷地の影響）
   
3. **交通量データの追加**
   - 道路交通センサスデータ
   - 交通負荷に応じた劣化予測

4. **時系列分析の導入**
   - 健全度の経年変化モデル
   - 次回点検時期の予測

### 8.2 中期的改善（v1.0候補）

1. **インタラクティブダッシュボード**
   - Plotly/Dash による Web UI
   - 地図上での橋梁可視化（Folium）
   
2. **説明可能AI（XAI）の導入**
   - SHAP値によるクラスタ特徴の解釈
   - 補修優先度の根拠説明

3. **予測モデルの構築**
   - 健全度の将来予測（LSTM、Prophet）
   - 補修コストの推定

4. **🆕 空間統計の活用**
   - 空間的自己相関分析（Moran's I）
   - ホットスポット分析（Getis-Ord Gi*）

### 8.3 長期的改善（v2.0候補）

1. **画像解析との統合**
   - ドローン撮影画像からの損傷検出
   - CNNによる健全度自動判定

2. **マルチモーダル学習**
   - 画像 + 数値データの統合分析
   - Transformer モデルの適用

3. **全国展開**
   - 他都道府県データへの適用
   - 自治体間の比較分析

4. **リアルタイム更新**
   - IoTセンサーデータの統合
   - 継続的な学習（Continual Learning）

---

## 📚 参考資料

### データソース

- **橋梁データ**: 山口県オープンデータ
- **財政データ**: 総務省 市町村別決算状況調
- **人口データ**: 総務省 住民基本台帳
- **🆕 河川データ**: 国土数値情報 河川データ（W05-08）
- **🆕 海岸線データ**: 国土数値情報 海岸線データ（C23-06）

### 技術文献

- scikit-learn Documentation: https://scikit-learn.org/
- UMAP Documentation: https://umap-learn.readthedocs.io/
- **🆕 geopandas Documentation: https://geopandas.org/**
- **🆕 国土数値情報: https://nlftp.mlit.go.jp/ksj/**

---

## 🆕 v0.4 更新履歴

### 追加機能

1. **地理空間特徴量の実装**
   - `under_river`: 桁下河川判定（57.0%が該当）
   - `distance_to_coast_km`: 海岸線距離（平均9.19km）

2. **地理空間データ処理基盤**
   - geopandas/shapely/pyprojの統合
   - CRS自動設定機能
   - 投影座標系による正確な距離計算

3. **🆕 Agenticなクラスタ数制御**
   - HDBSCANクラスタリングの追加
   - DBSCANのクラスタ数が50を超える場合の自動切り替え
   - 補修意思決定者にとって扱いやすいクラスタ数の維持

4. **ドキュメントの拡充**
   - 地理空間実装の詳細解説
   - 座標系選択のガイドライン
   - トラブルシューティング事例
   - HDBSCANの動作原理と使用条件

### 技術的改善

- UTM Zone 53N（EPSG:32653）による正確な距離計算
- `union_all()` 対応（geopandas新API）
- CRS未設定Shapefileへの自動対応
- **HDBSCAN統合によるクラスタ数の実用的制御**

### Agenticな振る舞い

- **自動判定**: DBSCANが50超のクラスタを生成した場合
- **自動試行**: HDBSCANを代替手法として評価に追加
- **自動選択**: 総合スコアに基づき最適な手法を選定

### 既知の制限事項

- 地理空間処理は約10〜15秒の追加時間が必要
- 座標欠損データは保守的な推定値を使用
- Shapefileのジオメトリ複雑度により処理時間が変動
- HDBSCAN実行時は追加で5〜10秒程度必要

---

## 📝 ライセンス

本プロジェクトは研究・教育目的で作成されています。

## 👥 貢献者

- データ分析: Agentic Clustering System
- 地理空間解析: GeoSpatial Extension Module (v0.4)

---

**最終更新**: 2025年11月24日（v0.4: 地理空間特徴量拡張版）
