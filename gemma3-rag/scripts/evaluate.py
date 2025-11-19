"""
Evaluation Script - Gemma3 RAG KasenSabo MVP
RAG応答の評価指標を計算
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import f1_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


class RAGEvaluator:
    """RAG評価クラス"""
    
    def __init__(self):
        """初期化"""
        # NLTKデータのダウンロード（初回のみ）
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # ROUGEスコアラーの初期化
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def normalize_text(self, text: str) -> str:
        """
        テキストの正規化
        
        Args:
            text: 入力テキスト
        
        Returns:
            正規化されたテキスト
        """
        if text is None:
            return ""
        
        # 小文字化
        text = text.lower()
        
        # 余分な空白を削除
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def exact_match(self, prediction: str, reference: str) -> int:
        """
        完全一致スコア（EM）
        
        Args:
            prediction: 予測テキスト
            reference: 参照テキスト
        
        Returns:
            1（一致）または0（不一致）
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        return 1 if pred_norm == ref_norm else 0
    
    def token_f1_score(self, prediction: str, reference: str) -> float:
        """
        トークンレベルのF1スコア
        
        Args:
            prediction: 予測テキスト
            reference: 参照テキスト
        
        Returns:
            F1スコア（0.0〜1.0）
        """
        pred_tokens = self.normalize_text(prediction).split()
        ref_tokens = self.normalize_text(reference).split()
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # 共通トークンの数
        common = sum((min(pred_tokens.count(token), ref_tokens.count(token))
                     for token in set(pred_tokens)))
        
        if common == 0:
            return 0.0
        
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def bleu_score(self, prediction: str, reference: str, max_n: int = 4) -> Dict[str, float]:
        """
        BLEUスコア
        
        Args:
            prediction: 予測テキスト
            reference: 参照テキスト
            max_n: 最大n-gram
        
        Returns:
            BLEUスコアの辞書
        """
        pred_tokens = self.normalize_text(prediction).split()
        ref_tokens = self.normalize_text(reference).split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}
        
        smoothing = SmoothingFunction().method1
        
        scores = {}
        for n in range(1, max_n + 1):
            weights = [1.0 / n] * n + [0.0] * (max_n - n)
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=weights,
                smoothing_function=smoothing
            )
            scores[f"bleu_{n}"] = score
        
        return scores
    
    def rouge_score(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        ROUGEスコア
        
        Args:
            prediction: 予測テキスト
            reference: 参照テキスト
        
        Returns:
            ROUGEスコアの辞書
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        if not pred_norm or not ref_norm:
            return {
                "rouge1_f": 0.0,
                "rouge2_f": 0.0,
                "rougeL_f": 0.0
            }
        
        scores = self.rouge_scorer.score(ref_norm, pred_norm)
        
        return {
            "rouge1_f": scores['rouge1'].fmeasure,
            "rouge2_f": scores['rouge2'].fmeasure,
            "rougeL_f": scores['rougeL'].fmeasure
        }
    
    def evaluate(self, prediction: str, reference: str, 
                response_time: Optional[float] = None,
                memory_usage: Optional[float] = None) -> Dict[str, Any]:
        """
        総合評価
        
        Args:
            prediction: 予測テキスト
            reference: 参照テキスト（正解）
            response_time: 応答時間（秒）
            memory_usage: メモリ使用量（MB）
        
        Returns:
            評価指標の辞書
        """
        result = {
            # テキスト評価
            "exact_match": self.exact_match(prediction, reference),
            "f1_score": self.token_f1_score(prediction, reference),
        }
        
        # BLEUスコア
        bleu_scores = self.bleu_score(prediction, reference)
        result.update(bleu_scores)
        
        # ROUGEスコア
        rouge_scores = self.rouge_score(prediction, reference)
        result.update(rouge_scores)
        
        # パフォーマンス指標
        if response_time is not None:
            result["response_time"] = response_time
        
        if memory_usage is not None:
            result["memory_usage_mb"] = memory_usage
        
        return result
    
    def evaluate_batch(self, predictions: List[str], references: List[str],
                      response_times: Optional[List[float]] = None,
                      memory_usages: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        バッチ評価
        
        Args:
            predictions: 予測テキストのリスト
            references: 参照テキストのリスト
            response_times: 応答時間のリスト
            memory_usages: メモリ使用量のリスト
        
        Returns:
            集計された評価指標
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")
        
        all_results = []
        
        for i in range(len(predictions)):
            rt = response_times[i] if response_times else None
            mu = memory_usages[i] if memory_usages else None
            
            result = self.evaluate(predictions[i], references[i], rt, mu)
            all_results.append(result)
        
        # 集計統計
        aggregated = {
            "count": len(all_results),
            "metrics": {}
        }
        
        # 各指標の平均値を計算
        metric_keys = all_results[0].keys()
        
        for key in metric_keys:
            values = [r[key] for r in all_results if key in r]
            if values:
                aggregated["metrics"][f"{key}_mean"] = np.mean(values)
                aggregated["metrics"][f"{key}_std"] = np.std(values)
                aggregated["metrics"][f"{key}_min"] = np.min(values)
                aggregated["metrics"][f"{key}_max"] = np.max(values)
        
        # 個別結果も保持
        aggregated["individual_results"] = all_results
        
        return aggregated


def main():
    """メイン実行関数（デモ用）"""
    print("=" * 50)
    print("RAG Evaluator Demo")
    print("=" * 50)
    
    evaluator = RAGEvaluator()
    
    # テストケース
    test_cases = [
        {
            "prediction": "河川の計画高水流量は、基本高水のピーク流量から洪水調節施設による調節流量を差し引いたものです。",
            "reference": "計画高水流量は、基本高水のピーク流量から洪水調節により軽減された流量です。",
            "response_time": 2.5,
            "memory_usage": 150.0
        },
        {
            "prediction": "ダムの洪水調節方式には、自然調節方式とゲート調節方式があります。",
            "reference": "洪水調節方式には自然調節方式とゲート調節方式が存在します。",
            "response_time": 2.1,
            "memory_usage": 145.0
        }
    ]
    
    print("\n[Individual Evaluations]")
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}:")
        result = evaluator.evaluate(
            case["prediction"],
            case["reference"],
            case["response_time"],
            case["memory_usage"]
        )
        
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # バッチ評価
    print("\n" + "=" * 50)
    print("[Batch Evaluation]")
    print("=" * 50)
    
    predictions = [c["prediction"] for c in test_cases]
    references = [c["reference"] for c in test_cases]
    response_times = [c["response_time"] for c in test_cases]
    memory_usages = [c["memory_usage"] for c in test_cases]
    
    batch_result = evaluator.evaluate_batch(
        predictions, references, response_times, memory_usages
    )
    
    print(f"\nTotal evaluations: {batch_result['count']}")
    print("\nAggregated Metrics:")
    for key, value in batch_result["metrics"].items():
        if "mean" in key or "std" in key:
            print(f"  {key}: {value:.4f}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
