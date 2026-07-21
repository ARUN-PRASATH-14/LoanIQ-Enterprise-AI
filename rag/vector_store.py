import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index.bin')
TEXTS_PATH = os.path.join(os.path.dirname(__file__), 'texts.npy')

class RAGSystem:
    def __init__(self):
        self.model = model
        self.index = None
        self.texts = []
        
        if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
            self.load_index()
        else:
            self.build_index()

    def load_index(self):
        self.index = faiss.read_index(INDEX_PATH)
        self.texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()

    def build_index(self):
        self.texts = []
        for filename in os.listdir(KNOWLEDGE_BASE_DIR):
            if filename.endswith(".txt"):
                path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple splitting by paragraph
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    self.texts.extend(paragraphs)

        if not self.texts:
            print("No texts found in knowledge base.")
            return

        embeddings = self.model.encode(self.texts)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        faiss.write_index(self.index, INDEX_PATH)
        np.save(TEXTS_PATH, np.array(self.texts))
        print(f"Index built with {len(self.texts)} chunks.")

    def query(self, text, top_k=2):
        if not self.index:
            return []
            
        vector = self.model.encode([text]).astype('float32')
        distances, indices = self.index.search(vector, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.texts):
                results.append(self.texts[idx])
        return results

# Singleton instance
rag_system = RAGSystem()
