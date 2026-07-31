import json
import re
import unicodedata
from pathlib import Path
import language_tool_python

def fix_common_errors(text: str) -> str:
    """Corrige errores frecuentes de OCR/encoding mediante reglas y regex."""
    if not text:
        return ""
        
    replacements = {
        "Nacihn": "Nación",
        "constitucihn": "Constitución",
        "Constitucihn": "Constitución",
        "Repubiica": "República",
        "econhmica": "económica",
        "representacihn": "representación",
        "intervencihn": "intervención",
        "cas0": "caso",
        "a1": "al",
        "10 s": "los",
        "cihn": "ción",
        "sihn": "sión",
        "l a": "la",
        "1as": "las"

        
    }
    
    for wrong, right in replacements.items():
        text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)

    text = re.sub(r'([a-záéíóúñ]+)cihn\b', r'\1ción', text, flags=re.IGNORECASE)
    text = re.sub(r'([a-záéíóúñ]+)sihn\b', r'\1sión', text, flags=re.IGNORECASE)
    text = re.sub(r'\beconh(micas?)\b', r'econó\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b10s\b', 'los', text)
    text = re.sub(r'\b1as\b', 'las', text)
    text = re.sub(r'\bl a\b', 'la', text)
    
    return text

def preprocess_text(text: str, tool=None) -> str:
    """Aplica normalización, corrección de OCR y opcionalmente LanguageTool."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = fix_common_errors(text)
    
    if tool:
        text = tool.correct(text)
        
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def corregir_dataset_jsonl(input_path="data/constituciones.jsonl", output_path="data/constituciones_corregidas.jsonl", usar_language_tool=True):
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    tool = None
    if usar_language_tool:
        print("Inicializando LanguageTool para español (requiere Java)...")
        tool = language_tool_python.LanguageTool('es')
    
    lines_processed = 0
    print(f"Procesando y corrigiendo texto de {input_path}...")
    
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            
            articulo_obj = obj.get("articulo", {})
            partes = articulo_obj.get("partes", articulo_obj.get("parte", []))
            
            if isinstance(partes, list):
                for p in partes:
                    if isinstance(p, dict) and "texto" in p and p["texto"]:
                        p["texto"] = preprocess_text(p["texto"], tool)
                    elif isinstance(p, str) and p.strip():
                        idx = partes.index(p)
                        partes[idx] = preprocess_text(p, tool)
            elif isinstance(partes, str) and partes.strip():
                if "parte" in articulo_obj:
                    articulo_obj["parte"] = preprocess_text(partes, tool)
                elif "partes" in articulo_obj:
                    articulo_obj["partes"] = preprocess_text(partes, tool)
                
            if "texto_completo" in articulo_obj and articulo_obj["texto_completo"]:
                articulo_obj["texto_completo"] = preprocess_text(articulo_obj["texto_completo"], tool)
                
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            lines_processed += 1
            if lines_processed % 50 == 0:
                print(f"{lines_processed} artículos procesados y corregidos...")

    if tool:
        tool.close()
        
    print(f"¡Proceso completado! Archivo corregido guardado en: {output_path}")

if __name__ == "__main__":
    # Cambia a False si prefieres omitir LanguageTool y usar únicamente tus reglas rápidas
    corregir_dataset_jsonl(usar_language_tool=True) 