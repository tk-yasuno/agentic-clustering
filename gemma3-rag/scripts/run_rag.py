"""
Run RAG Script - Gemma3 RAG KasenSabo MVP
Ollama + Gemma3でRAGクエリを実行
"""

import os
import yaml
import time
import psutil
from typing import Dict, Any, Optional
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore


class GemmaRAG:
    """Gemma3 RAG実行クラス"""
    
    def __init__(self, config_path: str = "config.yaml", model_name: str = "gemma:2b-instruct-q4_K_M"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
            model_name: 使用するOllamaモデル名
        """
        self.config = self._load_config(config_path)
        self.model_name = model_name
        self.index = None
        self.query_engine = None
        
        # 設定の初期化
        self._initialize_settings()
        
        # インデックスの読み込み
        self._load_index()
        
        # クエリエンジンの作成
        self._create_query_engine()
    
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _initialize_settings(self):
        """グローバル設定の初期化"""
        embed_model_name = self.config['embedding']['model_name']
        device = self.config['embedding']['device']
        
        # Embeddingモデルの設定
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            device=device
        )
        Settings.embed_model = embed_model
        
        print(f"✓ Embedding model initialized: {embed_model_name}")
    
    def _load_index(self):
        """インデックスの読み込み"""
        index_path = self.config['index']['path']
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        # FAISSインデックスの読み込み
        faiss_index = faiss.read_index(os.path.join(index_path, "index.faiss"))
        
        # FaissVectorStoreの作成
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # StorageContextの作成
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=index_path
        )
        
        # インデックスの読み込み
        self.index = load_index_from_storage(storage_context)
        
        print(f"✓ Index loaded from: {index_path}")
    
    def _create_query_engine(self):
        """クエリエンジンの作成"""
        # Ollamaモデルの設定
        llm = Ollama(
            model=self.model_name,
            temperature=self.config['rag']['temperature'],
            request_timeout=120.0,
            system_prompt="あなたは日本の河川・ダム・砂防技術の専門家です。質問に対して、提供されたコンテキスト情報に基づいて、日本語で正確かつ簡潔に回答してください。"
        )
        
        Settings.llm = llm
        
        # クエリエンジンの作成
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config['rag']['similarity_top_k']
        )
        
        print(f"✓ Query engine created with model: {self.model_name}")
    
    def query(self, question: str, measure_performance: bool = True) -> Dict[str, Any]:
        """
        RAGクエリを実行
        
        Args:
            question: 質問文
            measure_performance: パフォーマンス測定を行うか
        
        Returns:
            結果の辞書（response, response_time, memory_usageなど）
        """
        result = {
            "question": question,
            "model": self.model_name,
            "response": None,
            "response_time": None,
            "memory_usage_mb": None,
            "source_nodes": []
        }
        
        # メモリ使用量の測定開始
        if measure_performance:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # クエリの実行
        start_time = time.time()
        
        try:
            response = self.query_engine.query(question)
            result["response"] = str(response)
            
            # ソースノード情報の取得
            if hasattr(response, 'source_nodes'):
                result["source_nodes"] = [
                    {
                        "score": node.score,
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text
                    }
                    for node in response.source_nodes
                ]
        
        except Exception as e:
            result["response"] = f"Error: {str(e)}"
            result["error"] = str(e)
        
        # 実行時間の計測
        end_time = time.time()
        result["response_time"] = end_time - start_time
        
        # メモリ使用量の測定終了
        if measure_performance:
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            result["memory_usage_mb"] = mem_after - mem_before
        
        return result
    
    def batch_query(self, questions: list) -> list:
        """
        複数のクエリをバッチ実行
        
        Args:
            questions: 質問文のリスト
        
        Returns:
            結果のリスト
        """
        results = []
        
        print(f"\nProcessing {len(questions)} queries...")
        
        for i, question in enumerate(questions, 1):
            print(f"\r[{i}/{len(questions)}] Processing...", end="")
            result = self.query(question)
            results.append(result)
        
        print("\n✓ Batch processing completed")
        
        return results


def main():
    """メイン実行関数（デモ用）"""
    import json
    
    print("=" * 50)
    print("Gemma3 RAG - Query Execution Demo")
    print("=" * 50)
    
    # モデルの選択
    print("\nAvailable models:")
    print("1. gemma:2b-instruct-q4_K_M (INT4)")
    print("2. gemma:2b-instruct-q8_0 (INT8)")
    
    model_choice = input("\nSelect model (1 or 2): ").strip()
    
    model_map = {
        "1": "gemma:2b-instruct-q4_K_M",
        "2": "gemma:2b-instruct-q8_0"
    }
    
    model_name = model_map.get(model_choice, "gemma:2b-instruct-q4_K_M")
    
    # RAGシステムの初期化
    print(f"\nInitializing RAG system with {model_name}...")
    rag = GemmaRAG(model_name=model_name)
    
    # テストクエリ
    test_questions = [
        "河川の計画高水流量とは何ですか？",
        "ダムの洪水調節方式について教えてください。",
        "砂防堰堤の設計で重要な点は何ですか？"
    ]
    
    print("\n" + "=" * 50)
    print("Running test queries...")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Query {i}] {question}")
        result = rag.query(question)
        
        print(f"Response time: {result['response_time']:.2f}s")
        print(f"Response: {result['response'][:200]}...")
        
        if result.get('source_nodes'):
            print(f"Source nodes: {len(result['source_nodes'])}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
