# -*- coding: utf-8 -*-
"""
統合実行スクリプト: 山口県橋梁維持管理クラスタリングMVP
全処理を一括で実行します
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_preprocessing
import clustering
import visualization

def main():
    """全処理を順番に実行"""
    print("\n" + "="*70)
    print("🚀 橋梁維持管理クラスタリング MVP - 統合実行")
    print("="*70 + "\n")
    
    try:
        # ステップ1: データ前処理
        print("\n" + "─"*70)
        print("【ステップ 1/3】データ前処理")
        print("─"*70)
        df_processed = data_preprocessing.preprocess_all_data()
        
        if df_processed is None:
            print("\n❌ データ前処理に失敗しました。処理を中断します。")
            return False
        
        input("\n⏸️  続行するにはEnterキーを押してください...")
        
        # ステップ2: クラスタリング
        print("\n" + "─"*70)
        print("【ステップ 2/3】クラスタリング実行")
        print("─"*70)
        result = clustering.main()
        
        if result is None:
            print("\n❌ クラスタリングに失敗しました。処理を中断します。")
            return False
        
        input("\n⏸️  続行するにはEnterキーを押してください...")
        
        # ステップ3: 可視化
        print("\n" + "─"*70)
        print("【ステップ 3/3】結果の可視化")
        print("─"*70)
        visualization.main()
        
        # 完了メッセージ
        print("\n" + "="*70)
        print("✅ すべての処理が完了しました！")
        print("="*70)
        print("\n📁 結果は output/ フォルダに保存されています。")
        print("\n次のファイルを確認してください:")
        print("  📊 cluster_pca_scatter.png - PCA散布図")
        print("  🔥 cluster_heatmap.png - 特徴量ヒートマップ")
        print("  📡 cluster_radar.png - レーダーチャート")
        print("  📊 cluster_distribution.png - クラスタ分布")
        print("  📦 feature_boxplots.png - 箱ひげ図")
        print("  📝 cluster_report.txt - 分析レポート")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  ユーザーにより処理が中断されました。")
        return False
    
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 分析が正常に完了しました！")
    else:
        print("\n💔 分析を完了できませんでした。")
    
    input("\n終了するにはEnterキーを押してください...")
