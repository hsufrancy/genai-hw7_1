import os
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"

def load_pdf(path):
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts)

def load_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_txt_like(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_texts(texts):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vectors)

def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def build_index(folder: str) -> List[Dict[str, Any]]:
    """
    Scan a folder, load all supported files, split into chunks, embed them.
    Returns a list of dicts:
    {
        "source": filename,
        "chunk_id": int,
        "text": chunk text,
        "embedding": np.ndarray
    }
    """
    docs: List[Dict[str, Any]] = []

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(fname)[1].lower()

        if ext == ".pdf":
            raw = load_pdf(path)
        elif ext in [".txt", ".md"]:
            raw = load_txt_like(path)
        elif ext == ".docx":
            raw = load_docx(path)
        else:
            # Unsupported type -> skip
            continue

        chunks = split_text(raw)
        if not chunks:
            continue

        embeddings = embed_texts(chunks)

        for i, (ch, emb) in enumerate(zip(chunks, embeddings)):
            docs.append(
                {
                    "source": fname,
                    "chunk_id": i,
                    "text": ch,
                    "embedding": emb,
                }
            )

    return docs

def retrieve(docs: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed the query, compute cosine similarity with all chunks, return top-k.
    """
    if not docs:
        return []

    q_vec = embed_texts([query])[0]
    sims = [cosine_sim(q_vec, d["embedding"]) for d in docs]
    idxs = np.argsort(sims)[::-1][:k]

    results = []
    for idx in idxs:
        d = docs[idx]
        results.append(
            {
                "source": d["source"],
                "chunk_id": d["chunk_id"],
                "text": d["text"],
                "score": float(sims[idx]),
            }
        )
    return results

def generate_answer(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Combine retrieved contexts into a prompt and ask the LLM to answer.
    """
    context_text = "\n\n".join(c["text"] for c in contexts)

    system_msg = (
        "You are an expert data science tutor. "
        "First, try to answer using the provided context, which may contain class notes, "
        "homework descriptions, or project reports. "
        "If the context is not sufficient, you may use your general knowledge of data "
        "science, but clearly separate what comes from the context vs. your own knowledge. "
        "Always be technically accurate, use standard terminology, and aim explanations "
        "at a graduate student in data science."
    )

    user_msg = (
        f"Context:\n{context_text}\n\n"
        f"Student question: {query}\n\n"
        f"Explain your reasoning in a way a data science graduate student would understand."
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()