# -*- coding: utf-8 -*-
# GMM無効化の簡易確認テスト
import sys
import os

print("GMM無効化確認テスト")
print("=" * 60)

# agentic_workflow.pyを読み込んで確認
with open('agentic_workflow.py', 'r', encoding='utf-8') as f:
    content = f.read()

# GMMの実行がコメントアウトされているか確認
if '# GMM (オフ: K-Meansと同様のスコアのため実行スキップ)' in content:
    print("✅ GMM実行コードがコメントアウトされています")
    
    # try_gmm()がコメントアウトされているか確認
    if '# gmm_labels = alt_methods.try_gmm()' in content or '#     gmm_labels = alt_methods.try_gmm()' in content:
        print("✅ try_gmm()呼び出しがコメントアウトされています")
    else:
        print("❌ try_gmm()がまだアクティブです")
        
    # GMMの評価がコメントアウトされているか確認
    if "#     self.clustering_results['GMM']" in content or "# self.clustering_results['GMM']" in content:
        print("✅ GMM結果の保存コードがコメントアウトされています")
    else:
        print("❌ GMM結果の保存コードがまだアクティブです")
else:
    print("❌ GMMコメントアウトが見つかりません")

print("=" * 60)
print("\nコメントアウトされたGMMセクションの内容:")
print("=" * 60)

# コメントアウトされた部分を表示
lines = content.split('\n')
gmm_section_found = False
for i, line in enumerate(lines):
    if 'GMM (オフ:' in line:
        gmm_section_found = True
        start_idx = i
    if gmm_section_found and 'DBSCAN' in line and 'GMM' not in line:
        # GMMセクション終了
        for j in range(start_idx, i):
            print(lines[j])
        break

print("\n✅ GMM実行がオフになりました。")
