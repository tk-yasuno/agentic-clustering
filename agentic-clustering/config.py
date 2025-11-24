# -*- coding: utf-8 -*-
"""
設定ファイル: 山口県橋梁維持管理クラスタリングMVP
"""

import os

# プロジェクトルートディレクトリ
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# データファイルパス
BRIDGE_DATA_FILE = os.path.join(DATA_DIR, "YamaguchiPrefBridgeListOpen251122_154891.xlsx")
FISCAL_DATA_FILE = os.path.join(DATA_DIR, "全市町村の主要財政指標_000917808.xlsx")
POPULATION_DATA_FILE = os.path.join(DATA_DIR, "市区町村別年齢階級別人口_2304ssnen.xlsx")

# 出力ファイルパス
PROCESSED_DATA_FILE = os.path.join(OUTPUT_DIR, "processed_bridge_data.csv")
CLUSTER_RESULT_FILE = os.path.join(OUTPUT_DIR, "cluster_results.csv")
CLUSTER_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "cluster_summary.csv")

# クラスタリングパラメータ
MIN_CLUSTERS = 2
MAX_CLUSTERS = 28  # 山口県の市町村数(19) × 1.5
RANDOM_STATE = 42

# Agenticワークフローパラメータ
QUALITY_THRESHOLD = 60.0  # クラスタリング品質閾値
OVERLAP_THRESHOLD = 0.10  # オーバーラップ閾値
DBSCAN_CLUSTER_THRESHOLD = 50  # DBSCANのクラスタ数がこの値を超える場合はCLASSIXを試行

# PCAコンポーネント数
PCA_COMPONENTS = 2

# 地理空間データファイルパス
RIVER_SHAPEFILE = os.path.join(DATA_DIR, "RiverDataKokudo", "W05-08_35_GML", "W05-08_35-g_Stream.shp")
COASTLINE_SHAPEFILE = os.path.join(DATA_DIR, "KaigansenDataKokudo", "C23-06_35_GML", "C23-06_35-g_Coastline.shp")

# 特徴量カラム名（データ前処理後の標準化された名前）
FEATURE_COLUMNS = [
    "bridge_age",           # 橋齢
    "condition_score",      # 健全度スコア
    "maintenance_priority", # 補修優先度
    "future_burden_ratio",  # 将来負担比率（財政指標）
    "aging_rate",           # 高齢化率
    "fiscal_index",         # 財政力指数
    "structure_category",   # 構造形式カテゴリー（橋梁の種類から）
    "bridge_area",          # 橋面積（橋長×幅員）
    "emergency_route",      # 緊急輸送道路ダミー変数
    "overpass",             # 跨線橋ダミー変数
    "repair_year_normalized", # 最新補修年度の正規化値（min-max）
    "under_river",          # 桁下河川判定（0 or 1）
    "distance_to_coast_km"  # 海岸線までの距離（km）
]

# 可視化設定
FIGURE_SIZE = (10, 8)
FIGURE_DPI = 100
COLOR_PALETTE = "Set2"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
