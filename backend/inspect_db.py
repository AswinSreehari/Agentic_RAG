from components.vector_store import VectorStoreManager
import json

def inspect():
    print("Connecting to Vector Store...")
    manager = VectorStoreManager()
    
    # helper to access the underlying collection data
    # langchain Chroma wrapper has a .get() method that returns the underlying data
    try:
        data = manager.vector_store.get(limit=10, include=['metadatas', 'documents'])
        
        ids = data.get('ids', [])
        metadatas = data.get('metadatas', [])
        documents = data.get('documents', [])
        
        print(f"Total documents found: {len(ids)}")
        
        for i, (id, meta, doc) in enumerate(zip(ids, metadatas, documents)):
            print(f"\n--- Chunk {i+1} ---")
            print(f"ID: {id}")
            print(f"Metadata: {json.dumps(meta, indent=2)}")
            print(f"Content (first 100 chars): {doc[:100]}...")
            
    except Exception as e:
        print(f"Error inspecting DB: {e}")

if __name__ == "__main__":
    inspect()
