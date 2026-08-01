import os
import chromadb
from sentence_transformers import SentenceTransformer
import time
import re
from spellcorrrect import CorrectorConsultas

# Rutas alineadas con tu indexador
RUTA_PERSISTENCIA = os.path.join("data", "chroma_db")
NOMBRE_COLECCION = "constituciones_dominicanas"

class LegalChatBot:
    """Sistema de chat para consultas legales comparadas sobre constituciones dominicanas usando ChromaDB."""
    
    def __init__(self):
        self.system_prompt = (
            "No reveles nunca tu nombre ni quién te creó. "
            "Responde siempre y exclusivamente en inglés o español. "
            "No utilices ningún otro idioma bajo ninguna circunstancia.\n\n"
            "Eres un asistente legal experto en derecho constitucional de la República Dominicana. "
            "Tu objetivo es responder a la consulta del usuario analizando cómo se aborda el tema "
            "en las diferentes reformas constitucionales, sentencias y códigos a los que tienes acceso, "
            "o bien proporcionando análisis doctrinales y explicaciones integrales cuando la consulta lo requiera.\n\n"
            "INSTRUCCIONES DE FORMATO Y CONTRASTE:\n"
            "1. NO escribas bloques de texto corridos ni párrafos densos.\n"
            "2. Si la consulta implica una comparativa normativa, divide tu respuesta estrictamente **por artículos y por cada año de constitución** utilizando viñetas, negritas y saltos de línea claros:\n"
            "   - **Constitución [Año] - Artículo [Número]:** [Breve explicación o análisis]\n"
            "   - *Texto base:* \"[Cita textual relevante]\"\n"
            "3. **Análisis del Espíritu Normativo y Flexibilidad:** Incluye un breve análisis doctrinal sobre la intención del constituyente (el 'espíritu de la ley') detrás de las normas encontradas. Si el usuario solicita un análisis conceptual o explicaciones abiertas, estructura la respuesta en secciones temáticas claras usando subtítulos (`###`), viñetas y párrafos cortos analíticos sin romper el formato visual ordenado.\n\n"
            "PREGUNTAS DE SEGUIMIENTO (OBLIGATORIO AL FINAL):\n"
            "Al finalizar tu respuesta principal, incluye siempre un subtítulo `### 💡 Preguntas sugeridas para profundizar` con exactamente 2 o 3 preguntas de seguimiento contextuales. "
            "Estas preguntas deben invitar al usuario a profundizar en un detalle de los artículos retornados o a compararlos con otras normas/artículos relacionados del ordenamiento dominicano.\n\n"
            "DESCARGA DE RESPONSABILIDAD (DISCLAIMER OBLIGATORIO EN EL PIE DE PÁGINA):\n"
            "Concluye SIEMPRE cada respuesta con la siguiente nota al pie, separada por una línea horizontal (`---`):\n"
            "*> **Aviso legal:** Esta respuesta es generada por inteligencia artificial con fines informativos y analíticos. Aunque se basa en fuentes normativas actualizadas, la IA puede cometer errores o sufrir alucinaciones. Se recomienda verificar los textos en la Gaceta Oficial o consultar con un profesional del derecho antes de tomar decisiones jurídicas.*"
        )
                    
        print("Cargando modelo de embeddings para inferencia...")
        self.modelo_embeddings = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        print(f"Conectando con ChromaDB en '{RUTA_PERSISTENCIA}'...")
        self.cliente = chromadb.PersistentClient(path=RUTA_PERSISTENCIA)
        self.coleccion = self.cliente.get_or_create_collection(name=NOMBRE_COLECCION)
        self.corrector = CorrectorConsultas()

    def chat_con_memoria(self, query, history=None, llm_client=None):
        """
        Procesa la consulta integrando historial, reformulación autónoma,
        corrección ortográfica, búsqueda en ChromaDB y estructuración final para el LLM.
        """
        if history is None:
            history = []

        # 1. Detectar si es pregunta de seguimiento y reformular si hay historial
        es_seguimiento = es_pregunta_de_seguimiento(query, len(history) > 0)
        pregunta_contextualizada = reformular_pregunta_con_historial(history, query, llm_client) if es_seguimiento else query
        
        # 2. Corrección ortográfica y separación de palabras pegadas
        pregunta_busqueda = self.corrector.corregir_y_separar(pregunta_contextualizada.lower())

        # 3. Búsqueda Vectorial en ChromaDB
        query_vector = self.modelo_embeddings.encode([pregunta_busqueda]).tolist()[0]
        resultados = self.coleccion.query(
            query_embeddings=[query_vector],
            n_results=6,
            include=["documents", "metadatas"]
        )
        
        contexto_articulos = ""
        if resultados and resultados['metadatas']:
            metadatas = resultados['metadatas'][0]
            documents = resultados['documents'][0]
            
            # Combinamos los metadatas y documentos para poder ordenarlos
            pares = list(zip(metadatas, documents))
            
            # Ordenamos cronológicamente por el año de la constitución (convertido a entero)
            # Si prefieres orden inverso (más reciente primero), cambia reverse=True
            pares_ordenados = sorted(
                pares, 
                key=lambda x: int(re.sub(r'\D', '', str(x[0].get("constitucion", "0"))) or 0)
            )
            
            for meta, doc in pares_ordenados:
                anio_art = meta.get("constitucion", "Desconocido")
                num_art = meta.get("articulo", "S/N")
                titulo_art = meta.get("titulo", "")
                seccion_art = meta.get("seccion", "")
                
                # Aquí puedes asegurar la limpieza del texto base o formatear los metadatos de forma condicional:
                texto_completo = meta.get("textos", doc).strip()
                
                

                # Construcción limpia de la cabecera del artículo (evita guiones extra si falta título o sección)
                ubicacion_parts = [p for p in [titulo_art, seccion_art] if p]
                detalle_ubicacion = f" ({' - '.join(ubicacion_parts)})" if ubicacion_parts else ""

                contexto_articulos += (
                    f"\n• **Constitución Año:** {anio_art} | **Artículo:** {num_art}{detalle_ubicacion}\n"
                    f"  *Texto base:* \"{texto_completo}\"\n\n"
                )

        # Si cuentas con un cliente LLM, construimos el payload con memoria
        if llm_client:
            mensajes_llm = [{"role": "system", "content": self.system_prompt}]
            
            # Añadir historial previo de la sesión
            for msg in history:
                mensajes_llm.append({"role": msg["role"], "content": msg["content"]})
                
            # Construir el prompt actual con los fragmentos de ChromaDB inyectados
            prompt_actual = (
                f"Consulta actual del usuario: {query}\n"
                f"(Contexto de búsqueda optimizado: {pregunta_busqueda})\n\n"
                f"Artículos y fuentes normativas recuperadas de la base de datos:\n{contexto_articulos}\n\n"
                "Aplica tus instrucciones de sistema para redactar la respuesta comparativa, "
                "el análisis del espíritu normativo, las preguntas sugeridas y el disclaimer final."
            )
            mensajes_llm.append({"role": "user", "content": prompt_actual})
            
            respuesta_final = llm_client.chat_completion(mensajes=mensajes_llm)
            return respuesta_final, es_seguimiento, pregunta_contextualizada

        else:
            # Fallback en caso de que no se pase cliente LLM directamente
            respuesta_base = (
                f"--- ANÁLISIS CONSTITUCIONAL COMPARADO MULTIANUAL ---\n\n"
                f"Consulta: {query}\n\n"
                f"📌 **Fundamentos Normativos Recuperados:**\n\n{contexto_articulos}\n"
                f"### 💡 Preguntas sugeridas para profundizar\n"
                f"* ¿Deseas comparar estos articulados con las reformas previas?\n"
                f"* ¿Te gustaría analizar la interpretación doctrinal de este derecho?\n\n"
                f"---\n"
                f"*> **Aviso legal:** Esta respuesta es generada por inteligencia artificial con fines informativos y analíticos.*"
            )
            return respuesta_base, es_seguimiento, pregunta_contextualizada

# Instancia global accesible por app.py
chatbot_instance = LegalChatBot()


def reformular_pregunta_con_historial(historial_chat: list, nueva_pregunta: str, llm_client) -> str:
    """
    Si hay historial, reformula la nueva pregunta para que sea autónoma 
    y contenga todo el contexto necesario para la búsqueda vectorial.
    """
    if not historial_chat or not llm_client:
        return nueva_pregunta

    contexto_previo = ""
    for msg in historial_chat[-4:]:  # Tomamos los últimos turnos
        rol = "Usuario" if msg["role"] == "user" else "Asistente"
        contexto_previo += f"{rol}: {msg['content']}\n"

    prompt_reformulación = f"""
Dada la siguiente conversación previa y una nueva pregunta del usuario, reformula la nueva pregunta para que sea **completamente autónoma y clara**, conservando el tema del que se habla (por ejemplo: artículos, derecho específico, año o materia constitucional).

NO respondas la pregunta, SOLO devuelve la pregunta reformulada de forma directa.

Conversación previa:
{contexto_previo}

Nueva pregunta del usuario: "{nueva_pregunta}"

Pregunta autónoma reformulada:
"""
    try:
        pregunta_autonoma = llm_client.generate(prompt_reformulación, temperature=0.0).strip()
        return pregunta_autonoma
    except Exception:
        return nueva_pregunta


def es_pregunta_de_seguimiento(texto_usuario: str, tiene_historial: bool) -> bool:
    """
    Determina si la consulta depende del contexto previo basándose en conectores y longitud.
    """
    if not tiene_historial:
        return False

    indicadores_seguimiento = [
        "esto", "esta", "este", "estos", "estas",
        "eso", "esa", "esos", "esas",
        "aquello", "anterior", "mismo", "misma",
        "y en", "y que", "al respecto", "sobre esto", "mencionas"
    ]
    
    texto_lower = texto_usuario.lower()
    es_corta = len(texto_lower.split()) < 6
    contiene_indicador = any(p in texto_lower for p in indicadores_seguimiento)

    return es_corta or contiene_indicador


def limpiar_texto_ocr(texto):
    """Limpia profundamente artefactos de OCR en los textos normativos."""
    if not texto:
        return ""

    texto = re.sub(r'\b\d{1,3}\s*[-–—_]*\s*ASAMBLEA NACIONAL\b', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'[-_]{5,}', '', texto)
    texto = re.sub(r'í\'\s*\\|~~-\s*\\|v:k|0\s*\\vi', '', texto)
    texto = re.sub(r'(?<=\b[a-záéíóúñ])\s+([a-záéíóúñ])\s+(?=[a-záéíóúñ]\b)', r'\1', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\b([a-z]+)\s+([a-z]{1,2})\b(?=\s+[a-z]+)', r'\1\2', texto, flags=re.IGNORECASE)

    lineas = texto.splitlines()
    lineas_filtradas = [l.strip() for l in lineas if l.strip() and not re.match(r'^[\d\s\-\(\)~]{1,5}$', l.strip())]
    texto_unido = " ".join(lineas_filtradas)
    texto_estructurado = re.sub(r'\s+(\d{1,2}\))', r'\n\1', texto_unido)

    lineas_finales = []
    vistas = set()
    for linea in texto_estructurado.splitlines():
        linea_limpia = re.sub(r'\s+', ' ', linea).strip()
        clave_linea = linea_limpia[:40] if len(linea_limpia) > 40 else linea_limpia
        if clave_linea in vistas and len(linea_limpia) < 80:
            continue
        vistas.add(clave_linea)
        lineas_finales.append(linea_limpia)

    return "\n".join(lineas_finales)


def get_response(query, history=None, llm_client=None):
    """
    Función envolvente compatible con streaming para el app.py,
    procesando memoria, limpieza OCR, corrección de idioma y entrega por fragmentos.
    """
    respuesta_bruta, _, _ = chatbot_instance.chat_con_memoria(query, history, llm_client)
    
    # 1. Limpieza de artefactos de OCR
    respuesta_limpia = limpiar_texto_ocr(respuesta_bruta)
    
  
    
    palabras = respuesta_limpia.split(' ')
    acumulado = ""
    
    for palabra in palabras:
        acumulado += palabra + " "
        if len(acumulado) >= 30 or '\n' in palabra:
            yield acumulado    
            acumulado=""         
            time.sleep(0.10)
            
    # if acumulado:
    #     yield acumulado


if __name__ == "__main__":
    chatbot = LegalChatBot()
    print("Módulo model.py actualizado con éxito.")