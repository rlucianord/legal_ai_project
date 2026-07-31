import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path
from preprocessing import fix_common_errors

CHECKPOINT_DIR = "models/checkpoints"
INDEX_FILE = "data/faiss_constitution.index"
METADATA_FILE = "data/faiss_metadata.json"

def prepare_dataset(data_path="data/constituciones.jsonl"):
    """Prepara dataset para fine-tuning desde archivo JSONL adaptado a la jerarquía completa."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")
    
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            
            articulo_obj = obj.get("articulo", {})
            partes = articulo_obj.get("parte", articulo_obj.get("partes", []))
            
            piezas = []
            if isinstance(partes, list):
                for p in partes:
                    if isinstance(p, str) and p.strip():
                        piezas.append(p.strip())
                    elif isinstance(p, dict):
                        txt_parte = p.get('texto', '').strip()
                        if txt_parte:
                            piezas.append(txt_parte)
            elif isinstance(partes, str) and partes.strip():
                piezas.append(partes.strip())
                
            articulo_texto = " ".join(piezas).strip()
            if not articulo_texto:
                articulo_texto = articulo_obj.get("texto_completo", "").strip()
                
            if not articulo_texto:
                continue
                
            articulo_texto = fix_common_errors(articulo_texto)
                
            numero = articulo_obj.get("numero", "S/N")
            constitucion = obj.get("constitucion", "Desconocida")

            answer_text = articulo_texto
            answer_start = articulo_texto.find(answer_text) if answer_text else 0
            if answer_start == -1:
                answer_start = 0

            data.append({
                "context": articulo_texto,
                "question": f"¿Qué establece el artículo {numero} de la constitución dominicana de {constitucion}?",
                "answers": {"text": [answer_text], "answer_start": [answer_start]}
            })
            
    if not data:
        raise ValueError("El dataset está vacío o no se pudieron extraer artículos válidos.")
        
    return Dataset.from_list(data)

def train_model(dataset, checkpoint_dir=CHECKPOINT_DIR):
    """Entrena modelo BERT para QA y guarda checkpoints."""
    tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    model = BertForQuestionAnswering.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            try:
                idx = 0
                while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                if context_start >= len(sequence_ids) or offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    token_start_index = context_start
                    while token_start_index <= context_end and offset[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    token_end_index = context_end
                    while token_end_index >= context_start and offset[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)
            except (TypeError, IndexError):
                start_positions.append(0)
                end_positions.append(0)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

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
        train_dataset=tokenized_dataset
    )

    trainer.train()
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

def build_index(data_path="data/constituciones.jsonl"):
    """Carga o construye el índice FAISS integrando la jerarquía completa y limpiando errores."""
    embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        print("Cargando índice FAISS y metadatos desde el disco...")
        index = faiss.read_index(INDEX_FILE)
        
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            texts = data["texts"]
            metadata = data["metadata"]
            
        return embedder, index, texts, metadata

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")
    
    print("Construyendo índice FAISS con estructura jerárquica y texto corregido...")
    index = faiss.IndexFlatL2(768)
    texts = []
    metadata = []
    
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            
            anio_constitucion = obj.get("constitucion", "Desconocido")
            titulo_art = obj.get("titulo", "")
            capitulo_art = obj.get("capitulo", "")
            seccion_art = obj.get("seccion", "")
            
            articulo_data = obj.get("articulo", {})
            num_articulo = articulo_data.get("numero", "Desconocido")
            
            piezas_texto = []
            partes = articulo_data.get("partes", articulo_data.get("parte", []))
            
            if isinstance(partes, list):
                for parte in partes:
                    if isinstance(parte, str) and parte.strip():
                        piezas_texto.append(parte.strip())
                    elif isinstance(parte, dict):
                        txt_val = parte.get('texto', '').strip()
                        if txt_val:
                            piezas_texto.append(txt_val)
            elif isinstance(partes, str) and partes.strip():
                piezas_texto.append(partes.strip())
            
            texto_completo = " ".join(piezas_texto).strip()
            if not texto_completo:
                texto_completo = articulo_data.get("texto_completo", "").strip()
            
            if not texto_completo:
                continue
                
            texto_completo = fix_common_errors(texto_completo)
                
            texts.append(texto_completo)
            metadata.append({
                "constitucion": anio_constitucion,
                "titulo": titulo_art,
                "capitulo": capitulo_art,
                "seccion": seccion_art,
                "articulo": {"numero": num_articulo},
                "texto": texto_completo
            })
            
            emb = embedder.encode([texto_completo])
            index.add(emb)

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metadata": metadata}, f, ensure_ascii=False, indent=4)
        
    print("¡Índice jerárquico y metadatos guardados en disco exitosamente!")
    
    return embedder, index, texts, metadata

def main():
    """Función principal para ejecutar el entrenamiento e indexación."""
    print("Iniciando proceso de preparación y entrenamiento...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dataset = prepare_dataset()
    train_model(dataset)
    print("¡Entrenamiento y construcción de índices finalizados con éxito!")

if __name__ == "__main__":
    main()