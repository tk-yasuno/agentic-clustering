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
MAX_CLUSTERS = 8
RANDOM_STATE = 42

# PCAコンポーネント数
PCA_COMPONENTS = 2

# 特徴量カラム名（データ前処理後の標準化された名前）
FEATURE_COLUMNS = [
    "bridge_age",           # 橋齢
    "condition_score",      # 健全度スコア
    "maintenance_priority", # 補修優先度
    "population_decline",   # 人口減少率
    "aging_rate",           # 高齢化率
    "fiscal_index"          # 財政力指数
]

# 可視化設定
FIGURE_SIZE = (10, 8)
FIGURE_DPI = 100
COLOR_PALETTE = "Set2"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
