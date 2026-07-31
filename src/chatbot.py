import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from modeltrainer import build_index, CHECKPOINT_DIR

def load_qa_pipeline(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint_dir)
    return tokenizer, model

class LegalChatBot:
    """Sistema de chat para consultas legales comparadas sobre constituciones dominicanas con soporte jerárquico."""
    
    def __init__(self, data_path="data/constituciones.jsonl", checkpoint_dir=CHECKPOINT_DIR):
        self.system_prompt = (
            "Eres un asistente legal experto en derecho constitucional de la República Dominicana. "
            "Tu objetivo es responder a la consulta del usuario analizando cómo se aborda el tema "
            "en las diferentes reformas constitucionales (1994, 2002, 2010, 2015 y 2024). "
            "\n\nINSTRUCCIONES DE FORMATO OBLIGATORIAS:\n"
            "1. NO escribas bloques de texto corridos ni párrafos densos.\n"
            "2. Divide tu respuesta estrictamente **por artículos y por cada año de constitución**.\n"
            "3. Utiliza viñetas, negritas y saltos de línea claros para cada apartado.\n"
            "4. Estructura la respuesta de la siguiente forma para cada fuente encontrada:\n"
            "   - **Constitución [Año] - Artículo [Número]:** [Breve explicación o análisis]\n"
            "   - *Texto base:* \"[Cita textual relevante]\"\n"
            "5. En cada sección, incluye una breve interpretación del alcance legal del artículo junto con su texto base.\n"
        )
        
        self.embedder, self.index, self.texts, self.metadata = build_index(data_path)
        self.tokenizer, self.model = load_qa_pipeline(checkpoint_dir)
    
    def chat(self, query):
        """Procesa consulta recuperando múltiples fragmentos para asegurar cobertura multianual."""
        q_emb = self.embedder.encode([query])
        D, I = self.index.search(q_emb, k=12)
        
        contextos_encontrados = []
        fundamentos_normativos = []
        
        for indice_encontrado in I[0]:
            if indice_encontrado < 0 or indice_encontrado >= len(self.texts):
                continue
                
            contexto = self.texts[indice_encontrado]
            meta = self.metadata[indice_encontrado] if self.metadata else {}
            
            anio_art = meta.get("constitucion", "Desconocido")
            num_art = meta.get("articulo", {}).get("numero", "Desconocido")
            titulo_art = meta.get("titulo", "")
            capitulo_art = meta.get("capitulo", "")
            seccion_art = meta.get("seccion", "")

            ubicacion_partes = [p for p in [titulo_art, capitulo_art, seccion_art] if p]
            ubicacion_str = " > ".join(ubicacion_partes) if ubicacion_partes else "Estructura general"

            contextos_encontrados.append(f"[Constitución {anio_art} - Art. {num_art}]: {contexto}")
            fundamentos_normativos.append(
                f"• **Constitución Año:** {anio_art} | **Artículo:** {num_art}\n"
                f"  *Texto:* \"{contexto}\""
            )

        contexto_unificado = "\n\n".join(contextos_encontrados)
        query_con_prompt = f"{self.system_prompt}\n\nContexto normativo multianual:\n{contexto_unificado}\n\nConsulta del usuario: {query}"

        contexto_principal = self.texts[I[0][0]] if len(I[0]) > 0 else ""
        inputs = self.tokenizer(query_con_prompt, contexto_principal, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer = self.tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end], 
            skip_special_tokens=True
        ).strip()
        
        if not answer:
            answer = "Ver los textos íntegros del articulado histórico adjunto para el análisis detallado."

        fundamentos_str = "\n".join(list(dict.fromkeys(fundamentos_normativos)))

        respuesta_formateada = (
            f"--- ANÁLISIS CONSTITUCIONAL COMPARADO MULTIANUAL ---\n"
            f"Pregunta: {query}\n\n"
            f"Respuesta / Síntesis: {answer}\n\n"
            f"📌 Fundamentos Normativos (Todas las Constituciones Identificadas):\n"
            f"{fundamentos_str}"
        )
        
        return respuesta_formateada

def main():
    """Función principal para probar las consultas del LegalChatBot."""
    print("Inicializando LegalChatBot...")
    chatbot = LegalChatBot()
    
    print("\n" + "="*50)
    print(chatbot.chat("¿Quién ejerce la soberanía nacional?"))
    print("="*50 + "\n")
    print(chatbot.chat("¿Cómo ha evolucionado la regulación y el alcance del derecho al libre desarrollo de la personalidad y la libertad de empresa a través de las reformas constitucionales, cuáles son las diferencias específicas entre los textos de 1994, 2010 y 2024?"))

if __name__ == "__main__":
    main()