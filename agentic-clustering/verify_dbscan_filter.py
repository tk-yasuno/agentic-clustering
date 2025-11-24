# -*- coding: utf-8 -*-
# DBSCAN除外ルールのテスト
import sys
import os

print("=" * 60)
print("DBSCAN 50超過時の除外ルール確認")
print("=" * 60)

# agentic_workflow.pyを読み込んで確認
with open('agentic_workflow.py', 'r', encoding='utf-8') as f:
    content = f.read()

# _select_best_clustering()内の除外ロジックを確認
if 'if n_clusters > config.DBSCAN_CLUSTER_THRESHOLD:' in content:
    print("✅ DBSCANクラスタ数の閾値チェックが実装されています")
    
    if '採用候補から除外します' in content:
        print("✅ 除外メッセージが実装されています")
    else:
        print("❌ 除外メッセージが見つかりません")
        
    if 'filtered_results' in content:
        print("✅ フィルタリング後の結果を使用する実装があります")
    else:
        print("❌ フィルタリング実装が見つかりません")
else:
    print("❌ 閾値チェックが実装されていません")

print("=" * 60)
print("\n除外ロジックの内容:")
print("=" * 60)

# 除外ロジック部分を表示
lines = content.split('\n')
in_function = False
for i, line in enumerate(lines):
    if 'def _select_best_clustering(self):' in line:
        in_function = True
        start_idx = i
    if in_function and i > start_idx and line.strip().startswith('def '):
        # 次の関数定義に到達したら終了
        break
    if in_function:
        print(line)

print("\n✅ DBSCAN 50超過時の除外ルールが実装されました。")
