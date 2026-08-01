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


def extract_text_with_ocr(file_path: str) -> str:
    """
    Extrae texto de PDFs escaneados usando OCR con Tesseract.
    
    Args:
        file_path: Ruta al archivo PDF
    
    Returns:
        Texto extraído mediante OCR o cadena vacía si falla
    """
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
    """
    Extrae texto de PDF con múltiples métodos en orden de preferencia.
    
    Prioriza pdfminer, luego OCR, y finalmente PyPDF2 como fallback.
    
    Args:
        file_path: Ruta al archivo PDF
    
    Returns:
        Texto extraído del PDF
    """
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
    """
    Normaliza caracteres Unicode a forma NFC para español.
    
    Args:
        text: Texto a normalizar
    
    Returns:
        Texto normalizado en forma NFC
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)

def fix_common_errors(text: str) -> str:
    """
    Corrige errores frecuentes de OCR en textos jurídicos en español.
    
    Args:
        text: Texto con posibles errores de OCR
    
    Returns:
        Texto con errores corregidos
    """
    replacements = {
        r"\benseniianza\b": "enseñanza",
        r"\benesianza\b": "enseñanza",
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
    Limpia profundamente texto legal eliminando ruido de escaneo y errores OCR.
    
    Args:
        texto: Texto a limpiar
    
    Returns:
        Texto limpio sin artefactos de escaneo
    """
    if not texto or not isinstance(texto, str):
        return ""

    texto = normalize_spanish(texto) or ""
    texto = fix_common_errors(texto) or ""

    if not isinstance(texto, str):
        texto = str(texto)

    texto = re.sub(r'\b\d{1,3}\s*[-–—_]*\s*ASAMBLEA NACIONAL\b', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'[-_]{5,}', '', texto)
    texto = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', texto, re.MULTILINE)
    texto = re.sub(r'^(Página|Pag\.|Pág\.)\s*\d+(\s*de\s*\d+)?$', '', texto, re.IGNORECASE | re.MULTILINE)
    
    texto = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', texto)
    
    RANGO_ANIO = "1940" 
    texto = re.sub(r'\bk9\s*a\b', RANGO_ANIO, texto, flags=re.IGNORECASE)
    
    texto = re.sub(r'[ \t]+', ' ', texto)
    
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    texto_unido = " ".join(lineas)
    
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

def parse_hierarchical_structure(text: str) -> list:
    """
    Parsea la estructura jerárquica de documentos legales tolerando errores de OCR.
    
    Args:
        text: Texto del documento legal a parsear
    
    Returns:
        Lista de diccionarios con estructura jerárquica (títulos, secciones, artículos)
    """
    if not text or not isinstance(text, str):
        return [{"titulo": "General", "secciones": [{"seccion": "General", "articulos": []}]}]

    estructura = []
    
    patron_titulo = r'(?i)\b(T[IÍ]TULO\s+[IVXLCDM0-9]+)([\s\S]*?)(?=\bT[IÍ]TULO\s+[IVXLCDM0-9]+|\Z)'
    
    coincidencias = list(re.finditer(patron_titulo, text))
    
    if not coincidencias:
        return [{
            "titulo": "General", 
            "secciones": [{"seccion": "General", "articulos": extraer_articulos_seguro(text)}]
        }]
    
    primer_inicio = coincidencias[0].start()
    preambulo = text[:primer_inicio].strip()
    if preambulo:
        estructura.append({
            "titulo": "Preámbulo / Disposiciones Preliminares",
            "secciones": [{"seccion": "General", "articulos": extraer_articulos_seguro(preambulo)}]
        })
    
    for match in coincidencias:
        nombre_titulo = match.group(1).strip()
        contenido_titulo = match.group(2).strip()
        
        estructura.append({
            "titulo": nombre_titulo,
            "secciones": parsear_secciones_seguro(contenido_titulo)
        })
        
    return estructura

def parsear_secciones_seguro(texto_titulo: str) -> list:
    """
    Parsea secciones dentro de un título tolerando errores de OCR.
    
    Args:
        texto_titulo: Texto contenido en un título
    
    Returns:
        Lista de diccionarios con secciones y sus artículos
    """
    if not texto_titulo or not isinstance(texto_titulo, str):
        return [{"seccion": "General", "articulos": extraer_articulos_seguro(texto_titulo)}]

    patron_seccion = r'(?i)\b(SECCI[OÓ]N\s+[IVXLCDM0-9]+)([\s\S]*?)(?=\bSECCI[OÓ]N\s+[IVXLCDM0-9]+|\Z)'
    
    coincidencias_secc = list(re.finditer(patron_seccion, texto_titulo))
    
    if not coincidencias_secc:
        return [{
            "seccion": "General",
            "articulos": extraer_articulos_seguro(texto_titulo)
        }]
    
    secciones = []
    
    primer_inicio = coincidencias_secc[0].start()
    texto_inicial = texto_titulo[:primer_inicio].strip()
    if texto_inicial:
        secciones.append({
            "seccion": "General",
            "articulos": extraer_articulos_seguro(texto_inicial)
        })
        
    for match in coincidencias_secc:
        nombre_seccion = match.group(1).strip()
        contenido_seccion = match.group(2).strip()
        
        secciones.append({
            "seccion": nombre_seccion,
            "articulos": extraer_articulos_seguro(contenido_seccion)
        })
        
    return secciones

def extraer_articulos_seguro(bloque_texto: str) -> list:
    """
    Extrae artículos de un bloque de texto con numeración estándar.
    
    Args:
        bloque_texto: Texto que contiene artículos
    
    Returns:
        Lista de diccionarios con número y texto de cada artículo
    """
    articulos = []
    patron_articulo = r'(?i)\b(?:Art[ií]culo|Art\.?)\s*(\d+[°ºa-zA-Z]*)\s*[\.\-:]*'
    
    matches = list(re.finditer(patron_articulo, bloque_texto))
    
    if not matches:
        if bloque_texto.strip():
            return [{"numero_articulo": "S/N", "texto_completo": bloque_texto.strip()}]
        return []
    
    texto_suelto = bloque_texto[:matches[0].start()].strip()
    if texto_suelto:
        articulos.append({"numero_articulo": "Introducción", "texto_completo": texto_suelto})
        
    for idx, match in enumerate(matches):
        num_art = match.group(1).strip()
        inicio_texto = match.end()
        
        if idx + 1 < len(matches):
            fin_texto = matches[idx + 1].start()
        else:
            fin_texto = len(bloque_texto)
            
        texto_art = bloque_texto[inicio_texto:fin_texto].strip()
        
        if texto_art:
            articulos.append({
                "numero_articulo": f"Artículo {num_art}",
                "texto_completo": texto_art
            })
            
    return articulos
     

PDF_DIRECTORY = Path("data/pdfs")
RUTA_PERSISTENCIA = os.path.join("data", "chroma_db")
NOMBRE_COLECCION = "constituciones_dominicanas"

def indexar_constituciones():
    """
    Procesa, limpia e indexa constituciones en ChromaDB.
    
    Lee PDFs de constituciones, extrae texto, limpia artefactos de OCR,
    parsea estructura jerárquica y genera embeddings para búsqueda semántica.
    """
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

            texto_preprocesado = limpiar_texto_profundo(raw_text)
            estructura_documento = parse_hierarchical_structure(texto_preprocesado)

            for titulo_elem in estructura_documento:
                titulo_nombre = titulo_elem.get("titulo", "General")
                
                for seccion_elem in titulo_elem.get("secciones", []):
                    seccion_nombre = seccion_elem.get("seccion", "General")
                    
                    for articulo_elem in seccion_elem.get("articulos", []):
                        articulo_num = articulo_elem.get("numero_articulo", "S/N")
                        
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
    """
    Ejecuta el pipeline de indexación desde línea de comandos.
    """
    indexar_constituciones()