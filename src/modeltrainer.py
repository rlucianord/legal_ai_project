import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments, pipeline
from datasets import Dataset

CHECKPOINT_DIR = "models/checkpoints"

def prepare_dataset(data_path="data/constituciones.jsonl"):
    """Prepara dataset para fine-tuning desde archivo JSONL."""
    from pathlib import Path
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")
    
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            articulo_texto = " ".join(obj["articulo"]["partes"])
            numero = obj["articulo"]["numero"]
            constitucion = obj["constitucion"]

            answer_text = articulo_texto
            answer_start = articulo_texto.find(answer_text)

            data.append({
                "context": articulo_texto,
                "question": f"¿Qué establece el artículo {numero} de la constitución dominicana de {constitucion}?",
                "answers": {"text": [answer_text], "answer_start": [answer_start]}
            })
    return Dataset.from_list(data)

def train_model(dataset, checkpoint_dir=CHECKPOINT_DIR):
    """Entrena modelo BERT para QA y guarda checkpoints."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

    def preprocess(batch):
        return tokenizer(
            batch["question"],
            batch["context"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized = dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=50,
        save_total_limit=2,
        logging_dir="logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

def build_index(data_path="data/constituciones.jsonl"):
    """Construye índice FAISS con embeddings multilingües."""
    from pathlib import Path
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")
    
    embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    index = faiss.IndexFlatL2(384)
    texts = []

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texto = " ".join(obj["articulo"].get("partes", []))
            texts.append(texto)
            emb = embedder.encode([texto])
            index.add(emb)

    return embedder, index, texts

def load_qa_pipeline(checkpoint_dir=CHECKPOINT_DIR):
    """Carga pipeline de QA desde checkpoints."""
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    model = BertForQuestionAnswering.from_pretrained(checkpoint_dir)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

class LegalChatBot:
    """Sistema de chat para consultas legales sobre constituciones."""
    
    def __init__(self, data_path="data/constituciones.jsonl", checkpoint_dir=CHECKPOINT_DIR):
        self.embedder, self.index, self.texts = build_index(data_path)
        self.qa_pipeline = load_qa_pipeline(checkpoint_dir)
    
    def chat(self, query):
        """Procesa consulta y retorna respuesta con contexto."""
        q_emb = self.embedder.encode([query])
        D, I = self.index.search(q_emb, k=1)
        contexto = self.texts[I[0][0]]
        result = self.qa_pipeline({"question": query, "context": contexto})
        return f"Pregunta: {query}\nRespuesta: {result['answer']}\n\nArtículo usado:\n{contexto}"

def main():
    """Función principal de entrenamiento y prueba."""
    dataset = prepare_dataset()
    train_model(dataset)
    
    chatbot = LegalChatBot()
    print(chatbot.chat("¿Quién ejerce la soberanía nacional?"))

if __name__ == "__main__":
    main()
