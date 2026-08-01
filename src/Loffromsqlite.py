import sqlite3
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch
import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer, models
import numpy as np
from typing import Optional
from datasets import Dataset, Features, Value, Sequence
    
# Database connection
conn = sqlite3.connect("../DocumentAnalisys/data/contracts.db")
conn.row_factory = sqlite3.Row

def dividir_texto(texto: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap."""
    words = texto.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def fetch_contexts(output_dir: str = "contracts_contexts") -> Optional[Dataset]:
    """Fetch text contexts from SQLite and create a dataset"""
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM Contratos_2025_2025")
        # Convert rows to dictionaries and ensure string values
        rows = [{k: str(v) if v is not None else "" for k, v in dict(row).items()} 
               for row in cursor.fetchall()]
       
        if not rows:
            print("No data found in the database.")
            return None
            
        print(f"Fetched {len(rows)} rows from the database.")
        
        # Create dataset with just contexts
        contexts = []
        metadata = []
        column_names=[]
        for row in rows:
            rowdisct=dict(row)  # Convert Row object to dict
            row = {key.strip(): value for key, value in rowdisct.items()}  # Ensure keys are stripped of whitespace
            text_chunks = dividir_texto(row["Texto"])
            
            for chunk in text_chunks:
                contexts.append(chunk)
                # Store all original row data as metadata (already stringified)
                metadata.append(row.copy())
        column_names = [key.strip() for key in rows[0].keys()] if rows else []
        # Define features
        features = Features({
            "context": Value("string"),
            "metadata": Features({
                **{col: Value("string") for col in column_names}
            })
        })
        
        # Create Hugging Face dataset with explicit types
        hf_dataset = Dataset.from_dict({
            "context": contexts,
            "metadata": metadata
        }, features=features)
        
        # Save dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        hf_dataset.save_to_disk(output_dir)
        print(f"Contexts dataset saved to {output_dir}")
        
        return hf_dataset
        
    except Exception as e:
        print(f"Dataset creation error: {e}")
        return None
    finally:
        cursor.close()

def train_retriever(dataset_path: str = "contracts_contexts"):
    """Train a simple retriever model (using sentence embeddings)"""
  
    try:
        # Load contexts dataset
        dataset = Dataset.load_from_disk(dataset_path)
        contexts = dataset["context"]
        
        # Initialize sentence transformer model
        word_embedding_model = models.Transformer('bert-base-uncased')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        # Generate embeddings
        print("Generating embeddings for contexts...")
        context_embeddings = model.encode(contexts, show_progress_bar=True)
        
        # Save model and embeddings
        output_dir = "retriever_model"
        model.save(output_dir)
        np.save(os.path.join(output_dir, "context_embeddings.npy"), context_embeddings)
        
        # Save context references
        with open(os.path.join(output_dir, "contexts.txt"), "w") as f:
            for context in contexts:
                f.write(context[:1000] + "\n---\n")
        
        print(f"Retriever model saved to {output_dir}")
        return model
        
    except Exception as e:
        print(f"Retriever training error: {e}")
        return None

def retrieve_contexts(question: str, top_k: int = 3):
    """Retrieve relevant contexts for a question"""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    try:
        # Load model and embeddings
        model = SentenceTransformer("retriever_model")
        context_embeddings = np.load("retriever_model/context_embeddings.npy")
        
        # Load contexts
        with open("retriever_model/contexts.txt", "r") as f:
            contexts = f.read().split("---\n")
        
        # Embed question
        question_embedding = model.encode(question)
        
        # Compute similarities
        scores = np.dot(context_embeddings, question_embedding.T).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return top contexts
        return [(contexts[i].strip(), scores[i]) for i in top_indices]
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return None

def answer_question(question: str, context: str):
    """Answer question using a pretrained QA model"""
    from transformers import pipeline
    
    try:
        # Load QA pipeline (using a small pretrained model)
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad",
            tokenizer="distilbert-base-uncased-distilled-squad"
        )
        
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
        
    except Exception as e:
        print(f"QA error: {e}")
        return None

def main():
    # Step 1: Create contexts dataset
    print("Creating contexts dataset...")
    dataset = fetch_contexts()
    if dataset is None:
        return
    
    # Step 2: Train retriever
    print("\nTraining retriever model...")
    retriever = train_retriever()
    if retriever is None:
        return
    
    # Step 3: Interactive QA
    print("\nYou can now ask questions. Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
            
        # Retrieve relevant contexts
        contexts = retrieve_contexts(question)
        if not contexts:
            print("No contexts found.")
            continue
            
        print("\nMost relevant contexts:")
        for i, (context, score) in enumerate(contexts, 1):
            print(f"\nContext {i} (score: {score:.2f}):")
            print(context[:500] + "...")
            
            # Get answer for this context
            answer = answer_question(question, context)
            print(f"\nAnswer: {answer}")
            
            if input("\nContinue with next context? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    main()
    conn.close()