from pathlib import Path
import json
from .preprocessing import extract_text, preprocess_text, split_sections, split_articles, split_parts

def procesar_constitucion(file_path, year):
    """Procesa un archivo PDF de constitución y extrae artículos estructurados."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    try:
        text = extract_text(file_path)
        if not text or len(text.strip()) < 10:
            raise ValueError(f"No se pudo extraer texto del archivo: {file_path}")
        
        text = preprocess_text(text)
        secciones = split_sections(text)
        entries = []

        for seccion_nombre, seccion_texto in secciones:
            articulos = split_articles(seccion_texto)
            for articulo_num, articulo_texto in articulos:
                partes = split_parts(articulo_texto)
                if len(partes) == 0:
                    partes = [{"id": "", "texto": articulo_texto}]

                entry = {
                    "constitucion": str(year),
                    "seccion": seccion_nombre,
                    "articulo": {
                        "numero": articulo_num,
                        "partes": partes
                    }
                }
                entries.append(entry)
        return entries
    except Exception as e:
        raise RuntimeError(f"Error procesando {file_path}: {str(e)}")

def exportar_jsonl(entries, output_file):
    """Exporta entries a archivo JSONL."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    folder = Path("data/CONSTITUCION")
    anios = ["1994", "2002", "2010", "2015",'2024']

    all_entries = []

    for año in anios:
        for file_path in folder.glob(f"*{año}*.pdf"):
            print(f"Procesando {file_path.name} ({año})...")
            entries = procesar_constitucion(file_path, año)
            all_entries.extend(entries)

    exportar_jsonl(all_entries, "data/constituciones.jsonl")
    print("✅ Dataset exportado en data/constituciones.jsonl")
