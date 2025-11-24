# -*- coding: utf-8 -*-
"""
データ前処理スクリプト: 山口県橋梁維持管理クラスタリングMVP
- 橋梁データ、人口統計、財政力指数を統合
"""

import pandas as pd
import numpy as np
from datetime import datetime
import config

def load_bridge_data():
    """山口県橋梁データを読み込む"""
    print("🌉 橋梁データを読み込み中...")
    try:
        df = pd.read_excel(config.BRIDGE_DATA_FILE)
        print(f"  ✓ 橋梁データ: {len(df)}件")
        return df
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return None

def load_fiscal_data():
    """財政力指数データを読み込む"""
    print("💰 財政力指数データを読み込み中...")
    try:
        df = pd.read_excel(config.FISCAL_DATA_FILE)
        print(f"  ✓ 財政データ: {len(df)}件")
        return df
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return None

def load_population_data():
    """人口統計データを読み込む"""
    print("👥 人口統計データを読み込み中...")
    try:
        df = pd.read_excel(config.POPULATION_DATA_FILE)
        print(f"  ✓ 人口データ: {len(df)}件")
        return df
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return None

def calculate_bridge_age(df, construction_year_col=None):
    """橋齢を計算する"""
    current_year = datetime.now().year
    
    # 架設年の列名を推定（実際のデータに応じて調整）
    possible_cols = ['架設年次', '建設年', '竣工年', '年次', '架設年度']
    year_col = None
    
    if construction_year_col and construction_year_col in df.columns:
        year_col = construction_year_col
    else:
        for col in possible_cols:
            if col in df.columns:
                year_col = col
                break
    
    if year_col:
        df['bridge_age'] = current_year - pd.to_numeric(df[year_col], errors='coerce')
        df['bridge_age'] = df['bridge_age'].clip(lower=0, upper=150)  # 異常値を除外
    else:
        print("  ⚠ 架設年の列が見つかりません。橋齢をデフォルト値50で設定します。")
        df['bridge_age'] = 50
    
    return df

def extract_condition_score(df):
    """健全度スコアを抽出・数値化する"""
    # 健全度の列名を推定
    possible_cols = ['健全度', '健全性', '判定区分', '診断結果', '評価']
    condition_col = None
    
    for col in possible_cols:
        if col in df.columns:
            condition_col = col
            break
    
    if condition_col:
        # 健全度を数値化（Ⅰ=1, Ⅱ=2, Ⅲ=3, Ⅳ=4 など）
        condition_map = {'Ⅰ': 1, 'I': 1, '1': 1,
                        'Ⅱ': 2, 'II': 2, '2': 2,
                        'Ⅲ': 3, 'III': 3, '3': 3,
                        'Ⅳ': 4, 'IV': 4, '4': 4}
        
        df['condition_score'] = df[condition_col].astype(str).map(condition_map)
        df['condition_score'] = df['condition_score'].fillna(2)  # 欠損値は平均的な値2
    else:
        print("  ⚠ 健全度の列が見つかりません。デフォルト値2で設定します。")
        df['condition_score'] = 2
    
    return df

def calculate_maintenance_priority(df):
    """補修優先度を計算する（橋齢と健全度から）"""
    df['maintenance_priority'] = df['bridge_age'] * df['condition_score']
    return df

def merge_municipal_data(bridge_df, fiscal_df, population_df):
    """市町村データを橋梁データに結合する"""
    print("🔗 市町村データを統合中...")
    
    # 市町村名の列を推定
    municipal_cols = ['市町村', '自治体', '市区町村', '管理者']
    bridge_municipal_col = None
    
    for col in municipal_cols:
        if col in bridge_df.columns:
            bridge_municipal_col = col
            break
    
    if not bridge_municipal_col:
        print("  ⚠ 橋梁データに市町村名列が見つかりません。")
        # ダミーデータで代用
        bridge_df['population_decline'] = 10.0
        bridge_df['aging_rate'] = 30.0
        bridge_df['fiscal_index'] = 0.5
        return bridge_df
    
    # 財政力指数の処理
    fiscal_processed = process_fiscal_data(fiscal_df)
    
    # 人口統計の処理
    population_processed = process_population_data(population_df)
    
    # 市町村名で結合
    if fiscal_processed is not None:
        bridge_df = bridge_df.merge(fiscal_processed, 
                                     left_on=bridge_municipal_col, 
                                     right_on='municipality',
                                     how='left')
    else:
        bridge_df['fiscal_index'] = 0.5
    
    if population_processed is not None:
        bridge_df = bridge_df.merge(population_processed,
                                     left_on=bridge_municipal_col,
                                     right_on='municipality',
                                     how='left')
    else:
        bridge_df['population_decline'] = 10.0
        bridge_df['aging_rate'] = 30.0
    
    # 欠損値を平均値で埋める
    bridge_df['fiscal_index'] = bridge_df['fiscal_index'].fillna(bridge_df['fiscal_index'].mean())
    bridge_df['population_decline'] = bridge_df['population_decline'].fillna(10.0)
    bridge_df['aging_rate'] = bridge_df['aging_rate'].fillna(30.0)
    
    return bridge_df

def process_fiscal_data(fiscal_df):
    """財政力指数データを処理する"""
    try:
        # 財政力指数の列を探す
        fiscal_cols = ['財政力指数', '財政指数']
        municipal_cols = ['市町村', '団体名', '自治体名']
        
        fiscal_col = None
        municipal_col = None
        
        for col in fiscal_cols:
            if col in fiscal_df.columns:
                fiscal_col = col
                break
        
        for col in municipal_cols:
            if col in fiscal_df.columns:
                municipal_col = col
                break
        
        if fiscal_col and municipal_col:
            result = fiscal_df[[municipal_col, fiscal_col]].copy()
            result.columns = ['municipality', 'fiscal_index']
            result['fiscal_index'] = pd.to_numeric(result['fiscal_index'], errors='coerce')
            return result
        else:
            print("  ⚠ 財政力指数の適切な列が見つかりません。")
            return None
    except Exception as e:
        print(f"  ⚠ 財政データ処理エラー: {e}")
        return None

def process_population_data(population_df):
    """人口統計データを処理する"""
    try:
        # 市町村名、総人口、高齢者人口などの列を探す
        municipal_cols = ['市区町村', '市町村', '自治体名']
        total_pop_cols = ['総人口', '人口総数', '総数']
        elderly_cols = ['65歳以上', '高齢者', '65歳以上人口']
        
        municipal_col = None
        total_pop_col = None
        elderly_col = None
        
        for col in municipal_cols:
            if col in population_df.columns:
                municipal_col = col
                break
        
        for col in total_pop_cols:
            if col in population_df.columns:
                total_pop_col = col
                break
        
        for col in elderly_cols:
            if col in population_df.columns:
                elderly_col = col
                break
        
        if municipal_col and total_pop_col:
            result = population_df[[municipal_col]].copy()
            result.columns = ['municipality']
            
            # 人口減少率（仮）- 実際には過去データとの比較が必要
            result['population_decline'] = 10.0  # 仮の値（%）
            
            # 高齢化率
            if elderly_col:
                total_pop = pd.to_numeric(population_df[total_pop_col], errors='coerce')
                elderly_pop = pd.to_numeric(population_df[elderly_col], errors='coerce')
                result['aging_rate'] = (elderly_pop / total_pop * 100).fillna(30.0)
            else:
                result['aging_rate'] = 30.0  # デフォルト値
            
            return result
        else:
            print("  ⚠ 人口統計の適切な列が見つかりません。")
            return None
    except Exception as e:
        print(f"  ⚠ 人口データ処理エラー: {e}")
        return None

def preprocess_all_data():
    """すべてのデータを前処理して統合する"""
    print("\n" + "="*60)
    print("📊 データ前処理を開始します")
    print("="*60 + "\n")
    
    # データ読み込み
    bridge_df = load_bridge_data()
    fiscal_df = load_fiscal_data()
    population_df = load_population_data()
    
    if bridge_df is None:
        print("\n❌ 橋梁データが読み込めませんでした。処理を中断します。")
        return None
    
    # 橋梁データの処理
    print("\n🔧 橋梁データを処理中...")
    bridge_df = calculate_bridge_age(bridge_df)
    bridge_df = extract_condition_score(bridge_df)
    bridge_df = calculate_maintenance_priority(bridge_df)
    
    # 市町村データと結合
    if fiscal_df is not None or population_df is not None:
        bridge_df = merge_municipal_data(bridge_df, fiscal_df, population_df)
    else:
        print("  ⚠ 市町村データがないため、デフォルト値を使用します。")
        bridge_df['population_decline'] = 10.0
        bridge_df['aging_rate'] = 30.0
        bridge_df['fiscal_index'] = 0.5
    
    # 特徴量カラムのみを抽出
    feature_cols = config.FEATURE_COLUMNS
    available_cols = [col for col in feature_cols if col in bridge_df.columns]
    
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"  ⚠ 以下の特徴量が不足しています: {missing}")
    
    # 結果を保存
    bridge_df.to_csv(config.PROCESSED_DATA_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ 前処理完了: {len(bridge_df)}件の橋梁データ")
    print(f"📁 保存先: {config.PROCESSED_DATA_FILE}")
    print(f"📋 特徴量: {', '.join(available_cols)}")
    
    return bridge_df

if __name__ == "__main__":
    preprocess_all_data()
