"""
Post Evaluation Script - 既存のベンチマーク結果に評価指標を追加
参照回答を手動で入力して、EM/F1/BLEU/ROUGEスコアを計算
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from evaluate import RAGEvaluator

def load_benchmark_results(filepath: str) -> Dict:
    """ベンチマーク結果の読み込み"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_evaluation_results(filepath: str, results: Dict):
    """評価結果の保存"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def interactive_evaluation(results: Dict, sample_size: int = 20):
    """
    インタラクティブな評価
    サンプルの質問に対して参照回答を入力してもらい、評価指標を計算
    
    Args:
        results: ベンチマーク結果
        sample_size: 評価するサンプル数
    """
    evaluator = RAGEvaluator()
    detailed_results = results.get('detailed_results', [])
    
    print("\n" + "="*60)
    print("Post Evaluation - 回答品質の評価")
    print("="*60)
    print(f"\n総質問数: {len(detailed_results)}")
    print(f"評価サンプル数: {sample_size}問")
    print("\n各質問に対して、正しい参照回答を入力してください。")
    print("（スキップする場合は Enter キーを押してください）\n")
    
    evaluated_count = 0
    evaluation_scores = {
        'exact_match': [],
        'f1_score': [],
        'bleu_1': [],
        'bleu_2': [],
        'rouge1_f': [],
        'rougeL_f': []
    }
    
    # カテゴリごとに均等にサンプリング
    categories = {}
    for item in detailed_results:
        cat = item.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
    
    # 各カテゴリから均等にサンプリング
    samples = []
    per_category = max(1, sample_size // len(categories))
    for cat, items in categories.items():
        samples.extend(items[:per_category])
    samples = samples[:sample_size]
    
    for i, item in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"質問 {i}/{len(samples)}")
        print(f"カテゴリ: {item.get('category', 'unknown')}")
        print(f"{'='*60}")
        print(f"\n質問: {item['question']}")
        print(f"\nRAG回答:\n{item['response'][:300]}...")
        
        reference = input("\n✏️  正しい参照回答を入力してください（Enter でスキップ）: ").strip()
        
        if not reference:
            print("⏭️  スキップしました")
            continue
        
        # 評価指標の計算
        eval_result = evaluator.evaluate(
            item['response'],
            reference,
            item.get('response_time'),
            item.get('memory_usage_mb')
        )
        
        # 結果を保存
        item['reference'] = reference
        item['evaluation'] = eval_result
        
        # スコアを集計
        for key in evaluation_scores:
            if key in eval_result:
                evaluation_scores[key].append(eval_result[key])
        
        evaluated_count += 1
        
        # 評価結果の表示
        print(f"\n📊 評価結果:")
        print(f"  EM: {eval_result.get('exact_match', 0):.2f}")
        print(f"  F1: {eval_result.get('f1_score', 0):.2f}")
        print(f"  BLEU-1: {eval_result.get('bleu_1', 0):.2f}")
        print(f"  ROUGE-1: {eval_result.get('rouge1_f', 0):.2f}")
    
    # 集計結果の計算
    print(f"\n\n{'='*60}")
    print("評価結果サマリー")
    print(f"{'='*60}")
    print(f"評価完了数: {evaluated_count}/{len(samples)}")
    
    if evaluated_count > 0:
        print(f"\n平均スコア:")
        for metric, values in evaluation_scores.items():
            if values:
                avg = sum(values) / len(values)
                print(f"  {metric}: {avg:.3f}")
        
        # 結果に統計情報を追加
        results['evaluation_statistics'] = {
            'evaluated_count': evaluated_count,
            'total_samples': len(samples),
            'metrics': {
                metric: {
                    'mean': sum(values) / len(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0
                }
                for metric, values in evaluation_scores.items() if values
            }
        }
    
    return results

def llm_judge_evaluation(results: Dict, sample_size: int = 20):
    """
    LLM-as-a-Judge評価
    別のLLMを使って回答の品質を評価
    
    Args:
        results: ベンチマーク結果
        sample_size: 評価するサンプル数
    """
    print("\n⚠️  LLM-as-a-Judge評価は未実装です。")
    print("将来的には、別のLLMを使って自動評価を行う機能を追加予定です。")
    return results

def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ベンチマーク結果の事後評価')
    parser.add_argument('--input', '-i', required=True, help='ベンチマーク結果JSONファイル')
    parser.add_argument('--output', '-o', help='評価結果の出力先（省略時は元ファイルを上書き）')
    parser.add_argument('--sample-size', '-n', type=int, default=20, help='評価サンプル数（デフォルト: 20）')
    parser.add_argument('--mode', '-m', choices=['interactive', 'llm-judge'], default='interactive',
                        help='評価モード（interactive: 手動入力、llm-judge: LLM自動評価）')
    
    args = parser.parse_args()
    
    # 結果の読み込み
    print(f"\n📂 読み込み中: {args.input}")
    results = load_benchmark_results(args.input)
    
    # 評価の実行
    if args.mode == 'interactive':
        results = interactive_evaluation(results, args.sample_size)
    elif args.mode == 'llm-judge':
        results = llm_judge_evaluation(results, args.sample_size)
    
    # 結果の保存
    output_path = args.output or args.input
    print(f"\n💾 保存中: {output_path}")
    save_evaluation_results(output_path, results)
    
    print("\n✅ 評価完了！")
    
    # 統計情報の表示
    if 'evaluation_statistics' in results:
        stats = results['evaluation_statistics']
        print(f"\n最終統計:")
        print(f"  評価済み: {stats['evaluated_count']}/{stats['total_samples']}問")
        if stats['metrics']:
            print(f"\n  平均スコア:")
            for metric, values in stats['metrics'].items():
                print(f"    {metric}: {values['mean']:.3f}")

if __name__ == "__main__":
    main()
