import re
import unicodedata
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import language_tool_python

# Intentamos importar PaddleOCR de forma segura por si no está instalado aún
try:
    from paddleocr import PaddleOCR
    # Inicializamos PaddleOCR en español una sola vez
    _paddle_ocr_engine = PaddleOCR(use_angle_cls=True, lang='es')
except Exception:
    _paddle_ocr_engine = None

def extract_text_paddle(file_path: str) -> str:
    """Extrae texto utilizando PaddleOCR (ideal para PDFs escaneados o con ruido visual)."""
    if _paddle_ocr_engine is None:
        raise RuntimeError("PaddleOCR no está disponible o inicializado.")
    
    # Nota: Si usas pdf2image previamente para convertir páginas PDF a imágenes, puedes pasarlas. 
    # Aquí asumimos el flujo estándar de PaddleOCR sobre el archivo o imágenes.
    resultado = _paddle_ocr_engine.ocr(file_path, cls=True)
    lineas_texto = []
    if resultado:
        for idx in range(len(resultado)):
            res = resultado[idx]
            if res:
                for line in res:
                    if line and len(line) > 1 and len(line[1]) > 0:
                        lineas_texto.append(line[1][0])
    return "\n".join(lineas_texto)

def extract_text(file_path: str, usar_ocr: bool = True) -> str:
    """
    Extrae texto de un PDF priorizando PaddleOCR (si está activo), 
    luego pdfminer y por último PyPDF2.
    """
    # 1. Primera opción: PaddleOCR (Excelente para escaneos y conservación de orden)
    if usar_ocr and _paddle_ocr_engine is not None:
        try:
            texto_ocr = extract_text_paddle(file_path)
            if texto_ocr and len(texto_ocr.strip()) > 50:  # Validar que extrajo contenido real
                return texto_ocr
        except Exception:
            pass

    # 2. Segunda opción: pdfminer (Para PDFs digitales estándar)
    try:
        return pdfminer_extract_text(file_path)
    except Exception:
        pass
    
    # 3. Tercera opción: PyPDF2 (Fallback seguro)
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

# -------------------------------
# Limpieza y normalización de ruido
# -------------------------------
def limpiar_ruido_paginacion(text: str) -> str:
    """Elimina números de página aislados, líneas de guiones separadores y ruido común."""
    if not text:
        return ""
    
    text = re.sub(r'^[-\—_=\*]{3,}\s*$', '', text, re.MULTILINE)
    text = re.sub(r'^\s*[-–—]\s*\d+\s*[-–—]\s*$', '', text, re.MULTILINE)
    text = re.sub(r'^(Página|Pag\.|Pág\.)\s*\d+(\s*de\s*\d+)?$', '', text, re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text    
def clean_text(text: str) -> str:
    """Normaliza espacios en blanco."""
    return re.sub(r'\s+', ' ', text).strip()

# Inicializamos la herramienta una sola vez globalmente para optimizar velocidad
_lt_tool = None

def get_language_tool():
    global _lt_tool
    if _lt_tool is None:
        try:
            print("Inicializando motor de LanguageTool para español...")
            _lt_tool = language_tool_python.LanguageTool('es')
        except Exception as e:
            print(f"Advertencia: No se pudo inicializar LanguageTool (¿Falta Java?): {e}")
            _lt_tool = False
    return _lt_tool if _lt_tool else None

def normalize_spanish(text: str, usar_corrector: bool = True) -> str:
    """Normaliza caracteres Unicode a forma NFC y corrige ortografía/gramática con LanguageTool."""
    if not text:
        return ""
        
    text = unicodedata.normalize("NFC", text)
    
    if usar_corrector:
        tool = get_language_tool()
        if tool:
            text = tool.correct(text)
            
    return text

def fix_common_errors(text: str) -> str:
    """Corrige errores frecuentes de OCR/encoding en los PDFs."""
    replacements = {
        "Nacihn": "Nación",
        "constitucihn": "Constitución",
        "Repubiica": "República",
        "10s": "los",
        "10 s": "los",
        "econhmica": "económica",
        "representacihn": "representación",
        "intervencihn": "intervención",
        "cas0": "caso",
        "a1": "al",
        "cihn": "ción",
        "l a": "la"
    }
    for wrong, right in replacements.items():
        text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text: str) -> str:
    """Aplica todas las limpiezas en orden."""
    text = limpiar_ruido_paginacion(text)
    text = normalize_spanish(text)
    text = fix_common_errors(text)
    text = clean_text(text)
    return text

# -------------------------------
# División en jerarquía real (Títulos -> Secciones -> Artículos -> Partes)
# -------------------------------
def split_titulos(text: str):
    """Divide el texto buscando encabezados de 'TÍTULO'."""
    pattern = r'(T[ÍI]TULO\s+[IVXLC\d]+)'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    titulos = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        titulo_nombre = match.group().strip()
        titulo_texto = text[start:end].strip()
        titulos.append((titulo_nombre, titulo_texto))
    return titulos

def split_sections(text: str):
    """Divide el texto en secciones usando encabezados 'SECCION', 'SECCIÓN'."""
    pattern = r'(SECCI[ÓO]?N\s+[IVXLC\d]+)'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    secciones = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        seccion_nombre = match.group().strip()
        seccion_texto = text[start:end].strip()
        secciones.append((seccion_nombre, seccion_texto))
    return secciones

def split_articles(text: str):
    """Divide el texto en artículos usando encabezados 'Artículo', 'Art.', etc."""
    pattern = r'(?:Artículo|Articulo|Art\.|ART\.)\s*\d+[\.\-]*'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    articulos = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        articulo_num = match.group().strip()
        articulo_texto = text[start:end].strip()
        articulos.append((articulo_num, articulo_texto))
    return articulos

def split_parts(text: str):
    """Divide el texto de un artículo en incisos/partes usando patrones como '(a)', 'a)', '1)', '1.-'."""
    pattern = r'(\([A-Za-z]\)|[A-Za-z]\)|\d+\)|\d+\.-)'
    matches = list(re.finditer(pattern, text))
    partes = []
    if not matches:
        return [{"id": "Único", "texto": text}]
    
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        partes.append({
            "id": match.group().strip(),
            "texto": text[start:end].strip()
        })
    return partes

def parsear_articulos_con_partes(texto_bloque: str):
    """Extrae los artículos y desglosa sus partes internas."""
    articulos_crudos = split_articles(texto_bloque)
    articulos_finales = []
    
    if not articulos_crudos:
        return [{"numero_articulo": "Único", "texto_completo": texto_bloque, "partes": split_parts(texto_bloque)}]
        
    for art_num, art_texto in articulos_crudos:
        partes = split_parts(art_texto)
        articulos_finales.append({
            "numero_articulo": art_num,
            "texto_completo": art_texto,
            "partes": partes
        })
    return articulos_finales

def parse_hierarchical_structure(text: str):
    """
    Organiza la estructura completa en cascada:
    Títulos -> Secciones -> Artículos -> Partes.
    """
    if not text:
        return []
        
    titulos = split_titulos(text)
    estructura = []
    
    if titulos:
        for titulo_nombre, titulo_texto in titulos:
            secciones = split_sections(titulo_texto)
            secciones_procesadas = []
            
            if secciones:
                for sec_nombre, sec_texto in secciones:
                    articulos = parsear_articulos_con_partes(sec_texto)
                    secciones_procesadas.append({
                        "seccion": sec_nombre,
                        "articulos": articulos
                    })
            else:
                articulos = parsear_articulos_con_partes(titulo_texto)
                secciones_procesadas.append({
                    "seccion": "General",
                    "articulos": articulos
                })
                
            estructura.append({
                "titulo": titulo_nombre,
                "secciones": secciones_procesadas
            })
    else:
        secciones = split_sections(text)
        secciones_procesadas = []
        
        if secciones:
            for sec_nombre, sec_texto in secciones:
                articulos = parsear_articulos_con_partes(sec_texto)
                secciones_procesadas.append({
                    "seccion": sec_nombre,
                    "articulos": articulos
                })
        else:
            articulos = parsear_articulos_con_partes(text)
            secciones_procesadas.append({
                "seccion": "General",
                "articulos": articulos
            })
            
        estructura.append({
            "titulo": "General",
            "secciones": secciones_procesadas
        })
        
    return estructura