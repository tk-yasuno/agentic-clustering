# -*- coding: utf-8 -*-
"""
可視化スクリプト: 山口県橋梁維持管理クラスタリングMVP
- クラスタ散布図
- 特徴量ヒートマップ
- クラスタ特性レーダーチャート
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config
import os

# 日本語フォント設定
try:
    import japanize_matplotlib
    japanize_matplotlib.japanize()
except ImportError:
    print("⚠ japanize_matplotlib がインストールされていません。")
    print("  日本語表示には 'pip install japanize-matplotlib' を実行してください。")

def load_cluster_results():
    """クラスタリング結果を読み込む"""
    print("\n" + "="*60)
    print("📂 クラスタリング結果を読み込み中...")
    print("="*60)
    
    try:
        df = pd.read_csv(config.CLUSTER_RESULT_FILE)
        cluster_summary = pd.read_csv(config.CLUSTER_SUMMARY_FILE, index_col=0)
        print(f"✓ データ読み込み完了: {len(df)}件")
        return df, cluster_summary
    except FileNotFoundError:
        print("\n❌ クラスタリング結果が見つかりません。")
        print("先に clustering.py を実行してください。")
        return None, None
    except Exception as e:
        print(f"\n❌ データ読み込みエラー: {e}")
        return None, None

def plot_pca_clusters(df):
    """PCAによる2次元クラスタ散布図"""
    print("\n📊 PCA散布図を作成中...")
    
    # 特徴量を取得
    feature_cols = [col for col in config.FEATURE_COLUMNS if col in df.columns]
    X = df[feature_cols].fillna(0)
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # プロット
    plt.figure(figsize=config.FIGURE_SIZE)
    
    n_clusters = df['cluster'].nunique()
    colors = plt.cm.get_cmap(config.COLOR_PALETTE, n_clusters)
    
    for cluster_id in sorted(df['cluster'].unique()):
        mask = df['cluster'] == cluster_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors(cluster_id)],
                   label=f'クラスタ {cluster_id}',
                   alpha=0.6,
                   edgecolors='black',
                   linewidth=0.5,
                   s=100)
    
    plt.xlabel(f'第1主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    plt.ylabel(f'第2主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    plt.title('橋梁維持管理クラスタリング結果（PCA 2次元可視化）', fontsize=14, fontweight='bold')
    plt.legend(title='クラスタ', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(config.OUTPUT_DIR, 'cluster_pca_scatter.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 保存完了: {output_path}")
    plt.show()

def plot_cluster_heatmap(cluster_summary):
    """クラスタ特性ヒートマップ"""
    print("\n🔥 クラスタ特性ヒートマップを作成中...")
    
    # 特徴量を標準化（相対比較用）
    scaler = StandardScaler()
    cluster_summary_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_summary),
        columns=cluster_summary.columns,
        index=cluster_summary.index
    )
    
    plt.figure(figsize=(10, 6))
    
    sns.heatmap(cluster_summary_scaled.T,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn_r',
                center=0,
                cbar_kws={'label': '標準化スコア'},
                linewidths=0.5,
                linecolor='white')
    
    plt.xlabel('クラスタ', fontsize=12)
    plt.ylabel('特徴量', fontsize=12)
    plt.title('クラスタごとの特徴量比較（標準化）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(config.OUTPUT_DIR, 'cluster_heatmap.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 保存完了: {output_path}")
    plt.show()

def plot_radar_chart(cluster_summary):
    """クラスタ特性レーダーチャート"""
    print("\n📡 レーダーチャートを作成中...")
    
    # 特徴量を0-1スケールに正規化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_summary_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_summary),
        columns=cluster_summary.columns,
        index=cluster_summary.index
    )
    
    n_clusters = len(cluster_summary_scaled)
    n_features = len(cluster_summary_scaled.columns)
    
    # 角度を計算
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # 円を閉じる
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.get_cmap(config.COLOR_PALETTE, n_clusters)
    
    for i, (cluster_id, row) in enumerate(cluster_summary_scaled.iterrows()):
        values = row.tolist()
        values += values[:1]  # 円を閉じる
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'クラスタ {cluster_id}',
                color=colors(i))
        ax.fill(angles, values, alpha=0.25, color=colors(i))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cluster_summary_scaled.columns, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    plt.title('クラスタ特性レーダーチャート', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    
    output_path = os.path.join(config.OUTPUT_DIR, 'cluster_radar.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 保存完了: {output_path}")
    plt.show()

def plot_cluster_distribution(df):
    """クラスタ分布の棒グラフ"""
    print("\n📊 クラスタ分布を作成中...")
    
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    plt.figure(figsize=(8, 6))
    
    colors = plt.cm.get_cmap(config.COLOR_PALETTE, len(cluster_counts))
    bars = plt.bar(cluster_counts.index, cluster_counts.values,
                   color=[colors(i) for i in range(len(cluster_counts))],
                   edgecolor='black',
                   linewidth=1.5)
    
    # 棒の上に値を表示
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('クラスタ', fontsize=12)
    plt.ylabel('橋梁数', fontsize=12)
    plt.title('クラスタごとの橋梁分布', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(config.OUTPUT_DIR, 'cluster_distribution.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 保存完了: {output_path}")
    plt.show()

def plot_feature_boxplots(df):
    """特徴量のクラスタ別箱ひげ図"""
    print("\n📦 特徴量の箱ひげ図を作成中...")
    
    feature_cols = [col for col in config.FEATURE_COLUMNS if col in df.columns]
    n_features = len(feature_cols)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols):
        if i < len(axes):
            df.boxplot(column=feature, by='cluster', ax=axes[i])
            axes[i].set_title(feature, fontsize=12, fontweight='bold')
            axes[i].set_xlabel('クラスタ', fontsize=10)
            axes[i].set_ylabel('値', fontsize=10)
            axes[i].get_figure().suptitle('')  # デフォルトタイトルを削除
    
    # 使わない軸を非表示
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('特徴量のクラスタ別分布', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(config.OUTPUT_DIR, 'feature_boxplots.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 保存完了: {output_path}")
    plt.show()

def create_cluster_report(df, cluster_summary):
    """クラスタレポートをテキストファイルで出力"""
    print("\n📝 クラスタレポートを作成中...")
    
    output_path = os.path.join(config.OUTPUT_DIR, 'cluster_report.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("橋梁維持管理クラスタリング分析レポート\n")
        f.write("="*70 + "\n\n")
        
        # クラスタ分布
        f.write("【クラスタ分布】\n")
        cluster_counts = df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            f.write(f"  クラスタ {cluster_id}: {count}件 ({count/len(df)*100:.1f}%)\n")
        f.write(f"\n  合計: {len(df)}件\n\n")
        
        # クラスタ特性
        f.write("【クラスタごとの特徴量平均】\n")
        f.write(cluster_summary.to_string())
        f.write("\n\n")
        
        # クラスタ解釈
        f.write("【クラスタ解釈】\n")
        for cluster_id in cluster_summary.index:
            row = cluster_summary.loc[cluster_id]
            f.write(f"\n■ クラスタ {cluster_id}\n")
            
            # 各特徴量の値を記述
            for col in cluster_summary.columns:
                f.write(f"  - {col}: {row[col]:.2f}\n")
            
            # リスク評価
            high_risk_factors = []
            if 'bridge_age' in cluster_summary.columns and row['bridge_age'] > 50:
                high_risk_factors.append("高橋齢")
            if 'condition_score' in cluster_summary.columns and row['condition_score'] >= 3:
                high_risk_factors.append("健全度低下")
            if 'maintenance_priority' in cluster_summary.columns and row['maintenance_priority'] > 100:
                high_risk_factors.append("高補修優先度")
            if 'population_decline' in cluster_summary.columns and row['population_decline'] > 15:
                high_risk_factors.append("人口減少")
            if 'aging_rate' in cluster_summary.columns and row['aging_rate'] > 35:
                high_risk_factors.append("高齢化")
            if 'fiscal_index' in cluster_summary.columns and row['fiscal_index'] < 0.5:
                high_risk_factors.append("財政力弱")
            
            if len(high_risk_factors) >= 3:
                risk_level = "🔴 高リスク"
            elif len(high_risk_factors) >= 2:
                risk_level = "🟡 中リスク"
            else:
                risk_level = "🟢 低リスク"
            
            f.write(f"\n  【評価】{risk_level}\n")
            if high_risk_factors:
                f.write(f"  【要因】{', '.join(high_risk_factors)}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ 保存完了: {output_path}")

def main():
    """メイン処理"""
    print("\n" + "="*60)
    print("📊 クラスタリング結果の可視化")
    print("="*60)
    
    # データ読み込み
    df, cluster_summary = load_cluster_results()
    if df is None or cluster_summary is None:
        return
    
    # 各種可視化
    plot_pca_clusters(df)
    plot_cluster_heatmap(cluster_summary)
    plot_radar_chart(cluster_summary)
    plot_cluster_distribution(df)
    plot_feature_boxplots(df)
    
    # レポート作成
    create_cluster_report(df, cluster_summary)
    
    print("\n" + "="*60)
    print("✅ 可視化完了！")
    print("="*60)
    print(f"\n📁 出力ファイル:")
    print(f"   {config.OUTPUT_DIR}/")
    print(f"     - cluster_pca_scatter.png")
    print(f"     - cluster_heatmap.png")
    print(f"     - cluster_radar.png")
    print(f"     - cluster_distribution.png")
    print(f"     - feature_boxplots.png")
    print(f"     - cluster_report.txt")

if __name__ == "__main__":
    main()
