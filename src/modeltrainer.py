import os
import chromadb
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
from pathlib import Path
from processfiles import fix_common_errors

CHECKPOINT_DIR = "models/checkpoints"
RUTA_PERSISTENCIA = os.path.join("data", "chroma_db")
NOMBRE_COLECCION = "constituciones_dominicanas"

def cargar_datos_desde_chroma():
    """
    Extrae documentos y metadatos desde ChromaDB.
    
    Returns:
        Tupla con (lista documentos, lista metadatos)
    
    Raises:
        FileNotFoundError: Si la base de datos ChromaDB no existe
        ValueError: Si la colección no existe o está vacía
    """
    if not Path(RUTA_PERSISTENCIA).exists():
        raise FileNotFoundError(f"La base de datos ChromaDB no existe en '{RUTA_PERSISTENCIA}'. Ejecuta primero 'processfiles.py'.")
    
    cliente = chromadb.PersistentClient(path=RUTA_PERSISTENCIA)
    
    try:
        coleccion = cliente.get_collection(name=NOMBRE_COLECCION)
    except Exception:
        raise ValueError(f"No se encontró la colección '{NOMBRE_COLECCION}' en ChromaDB.")

    resultados = coleccion.get(include=["documents", "metadatas"])
    
    documents = resultados.get("documents", [])
    metadatas = resultados.get("metadatas", [])

    if not documents:
        raise ValueError("La colección en ChromaDB está vacía.")

    return documents, metadatas

def prepare_dataset():
    """
    Prepara el dataset para fine-tuning desde ChromaDB.
    
    Returns:
        Dataset de Hugging Face con datos preparados para entrenamiento
    
    Raises:
        ValueError: Si no se pueden generar muestras válidas
    """
    print("Cargando datos desde ChromaDB para entrenamiento...")
    documents, metadatas = cargar_datos_desde_chroma()
    
    data = []
    for texto_completo, meta in zip(documents, metadatas):
        if not texto_completo or not texto_completo.strip():
            continue
            
        texto_corregido = fix_common_errors(texto_completo.strip())
        
        numero = meta.get("articulo", "S/N")
        constitucion = meta.get("constitucion", "Desconocida")

        answer_text = texto_corregido
        answer_start = 0

        data.append({
            "context": texto_corregido,
            "question": f"¿Qué establece el artículo {numero} de la constitución dominicana de {constitucion}?",
            "answers": {"text": [answer_text], "answer_start": [answer_start]}
        })
        
    if not data:
        raise ValueError("No se pudieron generar muestras válidas para el dataset.")
        
    return Dataset.from_list(data)

def train_model(dataset, checkpoint_dir=CHECKPOINT_DIR):
    """
    Entrena modelo BERT español para Question Answering.
    
    Args:
        dataset: Dataset de Hugging Face preparado para entrenamiento
        checkpoint_dir: Directorio para guardar checkpoints del modelo
    """
    print("Cargando modelo y tokenizer de BERT...")
    tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    model = BertForQuestionAnswering.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    
    def preprocess_function(examples):
        """
        Preprocesa ejemplos para entrenamiento de QA.
        
        Args:
            examples: Batch de ejemplos del dataset
        
        Returns:
            Diccionario con inputs tokenizados y posiciones de respuesta
        """
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

    print("Tokenizando el dataset para entrenamiento...")
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

    print("Iniciando entrenamiento del modelo...")
    trainer.train()
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print("¡Entrenamiento finalizado y checkpoints guardados!")

def main():
    """
    Función principal para ejecutar el entrenamiento del modelo.
    
    Orquesta la preparación del dataset desde ChromaDB y el entrenamiento
    del modelo BERT español para Question Answering.
    """
    print("Iniciando proceso de preparación y entrenamiento con ChromaDB...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dataset = prepare_dataset()
    train_model(dataset)
    print("¡Proceso completado con éxito!")

if __name__ == "__main__":
    main()