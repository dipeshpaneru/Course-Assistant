import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from tqdm import tqdm
import pypdf


CHROMA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
COLLECTION    = "Course_Assistant"
EMBED_MODEL   = "all-MiniLM-L6-v2" 
BATCH_SIZE    = 500


class RAG:

    def __init__(self):
        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedder   = SentenceTransformer(EMBED_MODEL)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ RAG pipeline ready — {self.collection.count()} docs in store")


    def add_dataset(self, split_path, source_tag="train"):
        
        dataset = load_from_disk(split_path)

        for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
            batch = dataset[i:i + BATCH_SIZE]
            texts = [
                f"Question: {q}\nAnswer: {a}"
                for q, a in zip(batch["question"], batch["answer"])
            ]
            ids = [f"{source_tag}_{i + j}" for j in range(len(texts))]
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

            self.collection.add(
                ids = ids,
                documents = texts,
                embeddings = embeddings,
                metadatas  = [{"source": source_tag} for _ in texts]
            )



    def add_pdf(self, pdf_path):
        #This funciton is specifically for adding PDFs to the RAG store.  
        filename = os.path.basename(pdf_path)
       
        reader = pypdf.PdfReader(pdf_path)
        chunks = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                words = text.split()
                chunk_size = 100  # words per chunk
                for k in range(0, len(words), chunk_size):
                    chunk = " ".join(words[k:k + chunk_size])
                    if chunk.strip():
                        chunks.append({
                            "text":     chunk,
                            "page":     page_num + 1,
                            "filename": filename
                        })

        if not chunks:
            print(f"ERROR : These was an issue with extraction of {filename}")
            return

        # Embed and store
        texts = [c["text"] for c in chunks]
        ids = [f"pdf_{filename}_p{c['page']}_{idx}" for idx, c in enumerate(chunks)]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()
        metadatas  = [{"source": "pdf", "filename": c["filename"], "page": c["page"]} for c in chunks]

        self.collection.add(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas
        )
        print(f" Added {len(chunks)} chunks from {filename} — total docs: {self.collection.count()}")


    def retrieve(self, query, top_k=3):
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings = query_embedding,
            n_results        = top_k
        )
        return results["documents"][0]  # list of top-k strings


    def get_context(self, query, top_k=3):
        docs = self.retrieve(query, top_k=top_k)
        return "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])