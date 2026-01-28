"""Quick script to view Qdrant chunks - stop the app first, or use the API endpoint."""
import json
import os

# Option 1: View from exported JSON (if exists)
export_path = "data/chunks_export.json"

# Option 2: Direct Qdrant access (requires app to be stopped)
def view_from_qdrant():
    from qdrant_client import QdrantClient
    
    client = QdrantClient(path="data/qdrant_db")
    collection = client.get_collection("fra_documents")
    
    print(f"Total documents: {collection.points_count}")
    print(f"Vector dimension: {collection.config.params.vectors.size}")
    print("-" * 60)
    
    # Get sample chunks
    points, _ = client.scroll("fra_documents", limit=5, with_payload=True, with_vectors=False)
    
    for i, point in enumerate(points):
        print(f"\n{'='*60}")
        print(f"Chunk {i+1} (ID: {point.id})")
        print(f"{'='*60}")
        
        payload = point.payload
        print(f"Source: {payload.get('source', 'N/A')}")
        print(f"Entity Type: {payload.get('entity_type', 'N/A')}")
        print(f"Doc Type: {payload.get('doc_type', 'N/A')}")
        print(f"Topic: {payload.get('topic', 'N/A')}")
        print(f"\nContent Preview:")
        content = payload.get('content', payload.get('text', ''))[:500]
        print(content)
        print("...")

if __name__ == "__main__":
    print("To view chunks, first stop the running app (Ctrl+C in terminal)")
    print("Then run: python view_chunks.py")
    print()
    
    try:
        view_from_qdrant()
    except RuntimeError as e:
        if "already accessed" in str(e):
            print("❌ Database is locked by running app.")
            print("   Stop the app first, or use the web UI to query documents.")
        else:
            raise
