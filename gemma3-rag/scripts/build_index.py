"""
Build Index Script - Gemma3 RAG KasenSabo MVP
技術基準データをFAISSにインデックス化
"""

import os
import yaml
from pathlib import Path

# hf_transferを有効化（高速ダウンロード）
# 必ずインポート前に設定
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from tqdm import tqdm


def load_config(config_path: str = "config.yaml"):
    """設定ファイルの読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_index(config: dict):
    """知識ベースからインデックスを構築"""
    
    print("=" * 50)
    print("Gemma3 RAG - Index Building")
    print("=" * 50)
    
    # 設定の取得
    knowledge_base_path = config['data']['knowledge_base']
    index_path = config['index']['path']
    chunk_size = config['index']['chunk_size']
    chunk_overlap = config['index']['chunk_overlap']
    embed_model_name = config['embedding']['model_name']
    
    print(f"\n[1] Loading documents from: {knowledge_base_path}")
    
    # ドキュメントの読み込み
    if not os.path.exists(knowledge_base_path):
        raise FileNotFoundError(f"Knowledge base not found: {knowledge_base_path}")
    
    documents = SimpleDirectoryReader(
        knowledge_base_path,
        recursive=True,
        required_exts=[".md", ".txt", ".pdf"]
    ).load_data()
    
    print(f"✓ Loaded {len(documents)} documents")
    
    # Embeddingモデルの設定
    print(f"\n[2] Initializing embedding model: {embed_model_name}")
    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device=config['embedding']['device']
    )
    
    # グローバル設定
    Settings.embed_model = embed_model
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    
    print(f"✓ Embedding model loaded")
    print(f"✓ Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    # 埋め込み次元数を取得
    test_embedding = embed_model.get_text_embedding("test")
    dimension = len(test_embedding)
    print(f"✓ Embedding dimension: {dimension}")
    
    # FAISSの初期化
    print(f"\n[3] Initializing FAISS at: {index_path}")
    
    # インデックスディレクトリの作成
    os.makedirs(index_path, exist_ok=True)
    
    # FAISS インデックスの作成
    faiss_index = faiss.IndexFlatL2(dimension)
    
    # ベクトルストアの設定
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("✓ FAISS initialized")
    
    # インデックスの構築
    print(f"\n[4] Building vector index...")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    # インデックスの永続化
    index.storage_context.persist(persist_dir=index_path)
    
    # FAISSインデックスのバイナリ保存
    faiss.write_index(vector_store.client, os.path.join(index_path, "index.faiss"))
    
    print("✓ Index built successfully")
    
    # 統計情報の表示
    print("\n" + "=" * 50)
    print("Index Statistics")
    print("=" * 50)
    print(f"Total documents: {len(documents)}")
    print(f"Index path: {index_path}")
    print(f"Embedding model: {embed_model_name}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print("=" * 50)
    
    return index


def main():
    """メイン実行関数"""
    try:
        # 設定の読み込み
        config = load_config()
        
        # インデックスの構築
        index = build_index(config)
        
        print("\n✅ Index building completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
