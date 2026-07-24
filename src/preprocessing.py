import re
import unicodedata
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text

def extract_text(file_path: str, use_pdfminer: bool = True) -> str:
    """
    Extrae texto de un PDF usando pdfminer (default) o PyPDF2.
    
    Args:
        file_path: Ruta al archivo PDF
        use_pdfminer: Si True usa pdfminer, si False usa PyPDF2
    
    Returns:
        Texto extraído del PDF
    """
    if use_pdfminer:
        try:
            return extract_text(file_path)
        except Exception:
            pass
    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

# -------------------------------
# Limpieza y normalización
# -------------------------------
def clean_text(text: str) -> str:
    """Normaliza espacios en blanco."""
    return re.sub(r'\s+', ' ', text).strip()

def normalize_spanish(text: str) -> str:
    """Normaliza caracteres Unicode a forma NFC (acentos correctos)."""
    return unicodedata.normalize("NFC", text)

def fix_common_errors(text: str) -> str:
    """Corrige errores frecuentes de OCR/encoding en los PDFs."""
    replacements = {
        "Nacihn": "Nación",
        "constitucihn": "Constitución",
        "Repubiica": "República",
        "10s": "los",
        "econhmica": "económica",
        "representacihn": "representación",
        "intervencihn": "intervención",
        "cas0":"caso",
        'cihn':'ción'
    }
    for wrong, right in replacements.items():
        text = re.sub(rf"\b{wrong}\b", right, text)
    return text

def preprocess_text(text: str) -> str:
    """Aplica todas las limpiezas en orden."""
    text = normalize_spanish(text)
    text = fix_common_errors(text)
    text = clean_text(text)
    return text

# -------------------------------
# División en secciones, artículos y partes
# -------------------------------
def split_sections(text: str):
    """
    Divide el texto en secciones usando encabezados 'TÍTULO', 'SECCION', 'SECCIÓN'.
    """
    pattern = r'(T[ÍI]TULO\s+[IVXLC\d]+|SECCI[ÓO]?N\s+[IVXLC\d]+)'
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
    """
    Divide el texto en artículos usando encabezados 'Artículo', 'Art.', 'Articulo', 'ART.'.
    """
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
    """
    Divide el texto en incisos/partes usando patrones como '(a)', 'a)', '1)', '1.-'.
    """
    pattern = r'(\([A-Za-z]\)|[A-Za-z]\)|\d+\)|\d+\.-)'
    matches = list(re.finditer(pattern, text))
    partes = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        partes.append({
            "id": match.group(),
            "texto": text[start:end].strip()
        })
    return partes