"""
Benchmark Script - Gemma3 RAG KasenSabo MVP
200問のベンチマーク質問に対してRAGを実行し、評価
"""

import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm

# スクリプトのパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_rag import GemmaRAG
from evaluate import RAGEvaluator


class BenchmarkRunner:
    """ベンチマーク実行クラス"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.evaluator = RAGEvaluator()
        self.results = {
            "metadata": {},
            "models": {}
        }
    
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_questions(self) -> List[Dict[str, Any]]:
        """
        ベンチマーク質問の読み込み
        
        Returns:
            質問データのリスト
        """
        questions_path = self.config['data']['questions']
        
        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
        
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # データ構造の確認と正規化
        if isinstance(questions_data, list):
            questions = questions_data
        elif isinstance(questions_data, dict) and 'questions' in questions_data:
            questions = questions_data['questions']
        else:
            raise ValueError("Invalid questions file format")
        
        print(f"✓ Loaded {len(questions)} questions")
        
        return questions
    
    def run_benchmark(self, model_name: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        特定のモデルでベンチマークを実行
        
        Args:
            model_name: モデル名
            questions: 質問データのリスト
        
        Returns:
            ベンチマーク結果
        """
        print(f"\n{'='*60}")
        print(f"Running benchmark for: {model_name}")
        print(f"{'='*60}")
        
        # RAGシステムの初期化
        print(f"\n[1] Initializing RAG system...")
        rag = GemmaRAG(model_name=model_name)
        
        # ベンチマーク実行
        print(f"\n[2] Processing {len(questions)} questions...")
        
        results = []
        batch_size = self.config['benchmark']['batch_size']
        save_interval = self.config['benchmark']['save_interval']
        
        # tqdmの進捗バー設定（表示をクリーンに）
        for i, question_data in enumerate(tqdm(questions, desc="Progress", ncols=100, leave=True, position=0), 1):
            # 質問の取得
            if isinstance(question_data, dict):
                question = question_data.get('question', question_data.get('text', ''))
                reference = question_data.get('answer', question_data.get('expected_answer', ''))
                category = question_data.get('category', 'unknown')
                question_id = question_data.get('id', i)
            else:
                question = str(question_data)
                reference = ""
                category = "unknown"
                question_id = i
            
            # RAGクエリの実行
            rag_result = rag.query(question, measure_performance=True)
            
            # 評価（参照回答がある場合）
            eval_result = {}
            if reference:
                eval_result = self.evaluator.evaluate(
                    rag_result['response'],
                    reference,
                    rag_result.get('response_time'),
                    rag_result.get('memory_usage_mb')
                )
            
            # 結果の保存
            result = {
                "question_id": question_id,
                "question": question,
                "reference": reference,
                "category": category,
                "response": rag_result['response'],
                "response_time": rag_result.get('response_time', 0),
                "memory_usage_mb": rag_result.get('memory_usage_mb', 0),
                "source_nodes_count": len(rag_result.get('source_nodes', [])),
                "evaluation": eval_result
            }
            
            results.append(result)
            
            # 中間保存
            if i % save_interval == 0:
                self._save_intermediate_results(model_name, results, i)
        
        print(f"\n✓ Completed {len(results)} queries")
        
        # 統計情報の計算
        print(f"\n[3] Calculating statistics...")
        statistics = self._calculate_statistics(results)
        
        benchmark_result = {
            "model_name": model_name,
            "total_questions": len(questions),
            "timestamp": datetime.now().isoformat(),
            "statistics": statistics,
            "detailed_results": results
        }
        
        return benchmark_result
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        統計情報の計算
        
        Args:
            results: 結果のリスト
        
        Returns:
            統計情報の辞書
        """
        # DataFrameに変換
        df = pd.DataFrame(results)
        
        stats = {
            "response_time": {
                "mean": df['response_time'].mean(),
                "std": df['response_time'].std(),
                "min": df['response_time'].min(),
                "max": df['response_time'].max(),
                "median": df['response_time'].median()
            },
            "memory_usage_mb": {
                "mean": df['memory_usage_mb'].mean(),
                "std": df['memory_usage_mb'].std(),
                "min": df['memory_usage_mb'].min(),
                "max": df['memory_usage_mb'].max(),
                "median": df['memory_usage_mb'].median()
            }
        }
        
        # 評価指標の統計（参照回答がある場合）
        eval_metrics = ['exact_match', 'f1_score', 'bleu_1', 'bleu_2', 'rouge1_f', 'rougeL_f']
        
        for metric in eval_metrics:
            values = [r['evaluation'].get(metric, 0) for r in results if r.get('evaluation')]
            if values:
                stats[metric] = {
                    "mean": pd.Series(values).mean(),
                    "std": pd.Series(values).std(),
                    "min": pd.Series(values).min(),
                    "max": pd.Series(values).max()
                }
        
        # カテゴリ別統計
        if 'category' in df.columns:
            category_stats = {}
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                category_stats[category] = {
                    "count": len(cat_df),
                    "avg_response_time": cat_df['response_time'].mean()
                }
            stats["by_category"] = category_stats
        
        return stats
    
    def _save_intermediate_results(self, model_name: str, results: List[Dict[str, Any]], count: int):
        """中間結果の保存"""
        output_dir = self.config['benchmark']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace(':', '_')}_{count}_intermediate_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def save_results(self, model_name: str, benchmark_result: Dict[str, Any]):
        """
        結果の保存
        
        Args:
            model_name: モデル名
            benchmark_result: ベンチマーク結果
        """
        output_dir = self.config['benchmark']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で保存
        json_filename = f"{model_name.replace(':', '_')}_benchmark_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(benchmark_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Results saved to: {json_filepath}")
        
        # CSV形式で統計情報を保存
        csv_filename = f"{model_name.replace(':', '_')}_statistics_{timestamp}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        
        stats_df = pd.DataFrame([benchmark_result['statistics']])
        stats_df.to_csv(csv_filepath, index=False)
        
        print(f"✓ Statistics saved to: {csv_filepath}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]):
        """
        モデル間の比較
        
        Args:
            model_results: モデル別の結果辞書
        """
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}")
        
        comparison_data = []
        
        for model_name, result in model_results.items():
            stats = result['statistics']
            
            row = {
                "Model": model_name,
                "Avg Response Time (s)": f"{stats['response_time']['mean']:.2f}",
                "Avg Memory (MB)": f"{stats['memory_usage_mb']['mean']:.2f}",
            }
            
            # 評価指標の追加
            if 'f1_score' in stats:
                row["Avg F1 Score"] = f"{stats['f1_score']['mean']:.4f}"
            if 'bleu_1' in stats:
                row["Avg BLEU-1"] = f"{stats['bleu_1']['mean']:.4f}"
            if 'rouge1_f' in stats:
                row["Avg ROUGE-1"] = f"{stats['rouge1_f']['mean']:.4f}"
            
            comparison_data.append(row)
        
        # 比較表の表示
        df = pd.DataFrame(comparison_data)
        print("\n", df.to_string(index=False))
        
        # 比較結果の保存
        output_dir = self.config['benchmark']['output_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
        
        df.to_csv(comparison_file, index=False)
        print(f"\n✓ Comparison saved to: {comparison_file}")


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("Gemma3 RAG KasenSabo - Benchmark Execution")
    print("=" * 60)
    
    # ベンチマークランナーの初期化
    runner = BenchmarkRunner()
    
    # 質問の読み込み
    print("\n[Step 1] Loading benchmark questions...")
    questions = runner.load_questions()
    
    # 設定からモデルリストを取得
    models = runner.config['models']
    
    print(f"\n[Step 2] Models to benchmark:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']} ({model['quantization']})")
    
    # 各モデルでベンチマークを実行
    print(f"\n[Step 3] Running benchmarks...")
    
    model_results = {}
    
    for model in models:
        model_name = model['name']
        
        try:
            result = runner.run_benchmark(model_name, questions)
            model_results[model_name] = result
            
            # 結果の保存
            runner.save_results(model_name, result)
            
            # 統計情報の表示
            print(f"\n--- Statistics for {model_name} ---")
            stats = result['statistics']
            print(f"Avg Response Time: {stats['response_time']['mean']:.2f}s")
            print(f"Avg Memory Usage: {stats['memory_usage_mb']['mean']:.2f}MB")
            
        except Exception as e:
            print(f"\n❌ Error with model {model_name}: {str(e)}")
            continue
    
    # モデル間の比較
    if len(model_results) > 1:
        print(f"\n[Step 4] Comparing models...")
        runner.compare_models(model_results)
    
    print("\n" + "=" * 60)
    print("✅ Benchmark completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
