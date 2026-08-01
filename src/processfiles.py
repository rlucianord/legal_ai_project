import os
import re
from pathlib import Path
import unicodedata
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import chromadb
from sentence_transformers import SentenceTransformer
from spellcorrrect import CorrectorConsultas
# ==========================================
# 1. CONFIGURACIÓN LOCAL DE POPPLER Y TESSERACT
# ==========================================
POPPLER_PATH = os.path.join(os.getcwd(), "poppler-24.08.0", "Library", "bin")
if not os.path.exists(POPPLER_PATH):
    POPPLER_PATH = os.path.join(os.getcwd(), "poppler-24.08.0", "bin")

TESSERACT_PATH = os.path.join(os.getcwd(), "Tesseract-OCR", "tesseract.exe")
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"✅ Tesseract localizado en: {TESSERACT_PATH}")
else:
    print(f"⚠️ Advertencia: No se encontró Tesseract en '{TESSERACT_PATH}'.")

if os.path.exists(POPPLER_PATH):
    print(f"✅ Poppler localizado en: {POPPLER_PATH}")
else:
    print(f"⚠️ Advertencia: No se encontró Poppler en '{POPPLER_PATH}'.")


# ==========================================
# 2. EXTRACCIÓN Y LIMPIEZA
# ==========================================
def extract_text_with_ocr(file_path: str) -> str:
    """Extrae texto de PDFs escaneados usando Poppler y Tesseract-OCR."""
    try:
        if not os.path.exists(POPPLER_PATH) or not os.path.exists(TESSERACT_PATH):
            return ""
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        texto_completo = []
        for image in images:
            texto_pagina = pytesseract.image_to_string(image, lang='spa')
            texto_completo.append(texto_pagina)
        return "\n".join(texto_completo)
    except Exception as e:
        print(f"Error en extracción OCR para {file_path}: {e}")
        return ""

def extract_text(file_path: str) -> str:
    """Extrae texto priorizando pdfminer, respaldando con OCR y PyPDF2."""
    try:
        texto = pdfminer_extract_text(file_path)
        if texto and len(texto.strip()) > 50:
            return texto
    except Exception:
        pass

    try:
        texto_ocr = extract_text_with_ocr(file_path)
        if texto_ocr and len(texto_ocr.strip()) > 50:
            return texto_ocr
    except Exception:
        pass
    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

def normalize_spanish(text: str) -> str:
    """Normaliza caracteres Unicode a forma NFC."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)

def fix_common_errors(text: str) -> str:
    """Corrige errores frecuentes de OCR en textos jurídicos en español."""
    replacements = {
        r"\benseniianza\b": "enseñanza",
        r"\bensenianza\b": "enseñanza",
        r"\bniio\b": "niño",
        r"\bniia\b": "niña",
        r"\banios\b": "años",
        r"\banio\b": "año",
        r"\bNacihn\b": "Nación",
        r"\bnacion\b": "Nación",
        r"\bconstitucihn\b": "Constitución",
        r"\bconstituci[oó]n\b": "Constitución",
        r"\bRepubiica\b": "República",
        r"\brepublica\b": "República",
        r"\b10s\b": "los",
        r"\bl0s\b": "los",
        r"\b1a\b": "la",
        r"\b([a-záéíóúñ]+)cihn\b": r"\1ción"
        
    }
    for pattern, right in replacements.items():
        text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    return text

def limpiar_texto_profundo(texto: str) -> str:
    """
    Función de limpieza centralizada:
    - Remueve ruido de escaneo/marcas de agua/cabeceras.
    - Une letras sueltas dentro de palabras.
    - Estructura saltos de línea para numerales y elimina repeticiones.
    """
    if not texto:
        return ""

    texto = normalize_spanish(texto)
    texto = fix_common_errors(texto)

    # 1. Eliminar ruido de marcas, páginas y cabeceras de Asamblea Nacional
    texto = re.sub(r'\b\d{1,3}\s*[-–—_]*\s*ASAMBLEA NACIONAL\b', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'[-_]{5,}', '', texto)
    texto = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', texto, re.MULTILINE)
    texto = re.sub(r'^(Página|Pag\.|Pág\.)\s*\d+(\s*de\s*\d+)?$', '', texto, re.IGNORECASE | re.MULTILINE)
    
    # 2. Recomponer palabras cortadas por guiones o letras separadas por espacios (ej: "i gualdad")
    texto = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', texto)
    texto = re.sub(r'(?<=\b[a-záéíóúñ])\s+([a-záéíóúñ])\s+(?=[a-záéíóúñ]\b)', r'\1', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\b([a-z]+)\s+([a-z]{1,2})\b(?=\s+[a-z]+)', r'\1\2', texto, flags=re.IGNORECASE)
    # Busca una letra que simule el '1' seguida de un 9 y dos caracteres más (dígitos o letras)
    texto = re.sub(r'\b[klI1][9g]\s*([0-9a-z]{2})\b', r'19\1', texto, flags=re.IGNORECASE)
    
    # Correcciones específicas para terminaciones comunes como 'k9 a' (1940) o similares
    texto = re.sub(r'\bk9\s*a\b', '1940', texto, flags=re.IGNORECASE) # Ajusta según el año base si es constante
    
    texto_limpio = re.sub(r'\s+', ' ', texto)
    # Patrón flexible para capturar variaciones comunes de OCR en los años 19xx (ej. k9 a, l950, etc.)
    
    texto_estructurado = re.sub(r'\s+(\d{1,2}\))', r'\n\1', texto_limpio)


    # 3. Unir saltos de línea de párrafos rotos y normalizar espacios
    lineas = [ l.strip() for l in texto_estructurado.splitlines() if l.strip()]
    # lineas = [  l.strip() for l in texto.splitlines() if l.strip()]
    texto_unido = " ".join(lineas)
   
    # 4. Dar formato de línea a los numerales (1), 2), 3), etc.)
    
    # 5. Desduplicar fragmentos de texto repetidos al final
    lineas_finales = []
    vistas = set()
    for linea in texto_unido.splitlines():
        l_str = linea.strip()
        if not l_str:
            continue
        clave = l_str[:40] if len(l_str) > 40 else l_str
        if clave in vistas and len(l_str) < 90:
            continue
        vistas.add(clave)
        lineas_finales.append(l_str)

    return "\n".join(lineas_finales)


# ==========================================
# 3. PARSER JERÁRQUICO
# ==========================================
def parse_hierarchical_structure(text: str) -> list:
    """Parsea la estructura jerárquica: Títulos -> Secciones -> Artículos."""
    estructura = []
    patron_titulo = r'(?i)\b(T[IÍ]TULO\s+[IVXLCDM]+\b[^.\n]*)'
    fragmentos_titulos = re.split(patron_titulo, text)
    
    if len(fragmentos_titulos) <= 1:
        return [{"titulo": "General", "secciones": parsear_secciones(text)}]
    
    preambulo = fragmentos_titulos[0].strip()
    if preambulo:
        estructura.append({
            "titulo": "Preámbulo / Disposiciones Preliminares",
            "secciones": [{"seccion": "General", "articulos": [{"numero_articulo": "S/N", "texto_completo": preambulo}]}]
        })
    
    for i in range(1, len(fragmentos_titulos), 2):
        nombre_titulo = fragmentos_titulos[i].strip()
        contenido_titulo = fragmentos_titulos[i+1] if (i+1) < len(fragmentos_titulos) else ""
        estructura.append({
            "titulo": nombre_titulo,
            "secciones": parsear_secciones(contenido_titulo)
        })
        
    return estructura

def parsear_secciones(texto_titulo: str) -> list:
    """Divide el contenido de un Título en Secciones."""
    patron_seccion = r'(?i)\b(SECCI[OÓ]N\s+(?:[IVXLCDM]+|[A-ZÁÉÍÓÚÑ]+))\b'
    fragmentos_secciones = re.split(patron_seccion, texto_titulo)
    
    if len(fragmentos_secciones) <= 1:
        return [{"seccion": "General", "articulos": extraer_articulos(texto_titulo)}]
    
    secciones = []
    texto_inicial = fragmentos_secciones[0].strip()
    if texto_inicial:
        secciones.append({"seccion": "General", "articulos": extraer_articulos(texto_inicial)})
        
    for i in range(1, len(fragmentos_secciones), 2):
        nombre_seccion = fragmentos_secciones[i].strip()
        contenido_seccion = fragmentos_secciones[i+1] if (i+1) < len(fragmentos_secciones) else ""
        secciones.append({"seccion": nombre_seccion, "articulos": extraer_articulos(contenido_seccion)})
        
    return secciones

def extraer_articulos(bloque_texto: str) -> list:
    """Extrae artículos dividiendo el texto por la palabra clave Artículo."""
    articulos = []
    patron_articulo = r'(?i)\b(?:Art[ií]culo|Art\.?)\s*(\d+[°ºa-zA-Z]*)\s*[\.\-:]*'
    fragmentos_arts = re.split(patron_articulo, bloque_texto)
    
    if len(fragmentos_arts) <= 1:
        if bloque_texto.strip():
            return [{"numero_articulo": "S/N", "texto_completo": bloque_texto.strip()}]
        return []
        
    texto_suelto = fragmentos_arts[0].strip()
    if texto_suelto:
        articulos.append({"numero_articulo": "Introducción", "texto_completo": texto_suelto})
        
    for i in range(1, len(fragmentos_arts), 2):
        num_art = fragmentos_arts[i].strip()
        texto_art = fragmentos_arts[i+1].strip() if (i+1) < len(fragmentos_arts) else ""
        articulos.append({
            "numero_articulo": f"Artículo {num_art}",
            "texto_completo": texto_art
        })
        
    return articulos


# ==========================================
# 4. PIPELINE DE PROCESAMIENTO E INDEXACIÓN
# ==========================================
PDF_DIRECTORY = Path("data/pdfs")
RUTA_PERSISTENCIA = os.path.join("data", "chroma_db")
NOMBRE_COLECCION = "constituciones_dominicanas"

def indexar_constituciones():
    """Procesa, limpia e indexa las constituciones de forma limpia e higiénica."""
    if not PDF_DIRECTORY.exists():
        print(f"⚠️ El directorio '{PDF_DIRECTORY}' no existe. Creándolo...")
        PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
        return

    anios = ["1994", "2002", "2010", "2015", "2024"]
    archivos_pdf = []
    for año in anios:
        archivos_pdf.extend(list(PDF_DIRECTORY.glob(f"*{año}*.pdf")))

    if not archivos_pdf:
        print(f"⚠️ No se encontraron archivos PDF de constituciones en {PDF_DIRECTORY}")
        return

    print(f"Se encontraron {len(archivos_pdf)} archivos PDF para procesar.")
    print("Cargando modelo de embeddings (SentenceTransformers)...")
    modelo_embeddings = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    cliente = chromadb.PersistentClient(path=RUTA_PERSISTENCIA)
    coleccion = cliente.get_or_create_collection(name=NOMBRE_COLECCION)

    ids, textos, metadatos = [], [], []
    doc_counter = 0

    for file_path in archivos_pdf:
        año_encontrado = next((a for a in anios if a in file_path.name), "Desconocido")
        print(f"Procesando: {file_path.name} (Constitución {año_encontrado})...")

        try:
            raw_text = extract_text(str(file_path))
            if not raw_text or len(raw_text.strip()) < 10:
                print(f"⚠️ Advertencia: No se pudo extraer texto de {file_path.name}")
                continue

            # 1. Limpieza inicial del documento entero
            texto_preprocesado = limpiar_texto_profundo(raw_text)
            
            # 2. Parseo de la jerarquía
            estructura_documento = parse_hierarchical_structure(texto_preprocesado)

            for titulo_elem in estructura_documento:
                titulo_nombre = titulo_elem.get("titulo", "General")
                
                for seccion_elem in titulo_elem.get("secciones", []):
                    seccion_nombre = seccion_elem.get("seccion", "General")
                    
                    for articulo_elem in seccion_elem.get("articulos", []):
                        articulo_num = articulo_elem.get("numero_articulo", "S/N")
                        
                        # 3. Limpieza final individual del artículo (sin concatenaciones duplicadas)
                        articulo_texto = limpiar_texto_profundo(articulo_elem.get("texto_completo", ""))

                        if not articulo_texto.strip():
                            continue

                        doc_id = f"const_{año_encontrado}_art_{articulo_num}_{doc_counter}"
                        doc_counter += 1

                        ids.append(doc_id)
                        textos.append(articulo_texto)
                        metadatos.append({
                            "constitucion": str(año_encontrado),
                            "titulo": str(titulo_nombre),
                            "seccion": str(seccion_nombre),
                            "articulo": str(articulo_num),
                            "source": file_path.name
                        })

        except Exception as e:
            print(f"❌ Error procesando {file_path.name}: {e}")

    if textos:
        print(f"Generando vectores en lote para {len(textos)} fragmentos normativos...")
        embeddings = modelo_embeddings.encode(textos).tolist()

        print(f"Guardando en ChromaDB (colección: '{NOMBRE_COLECCION}')...")
        coleccion.upsert(
            ids=ids,
            documents=textos,
            embeddings=embeddings,
            metadatas=metadatos
        )
        print(f"¡Proceso completado con éxito! Base de datos actualizada en '{RUTA_PERSISTENCIA}'.")
    else:
        print("No hay texto válido para indexar en ChromaDB.")

if __name__ == "__main__":
    indexar_constituciones()