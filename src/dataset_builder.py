from pathlib import Path
import json
from preprocessing import extract_text, preprocess_text, parse_hierarchical_structure, split_parts,fix_common_errors


def procesar_constitucion(file_path, year):
    """Procesa un archivo PDF de constitución y extrae artículos estructurados con jerarquía completa."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    try:
        text = extract_text(file_path)
        if not text or len(text.strip()) < 10:
            raise ValueError(f"No se pudo extraer texto del archivo: {file_path}")
        
        text = preprocess_text(text)
        
        # Procesamiento mediante estructura jerárquica en cascada
        estructura_documento = parse_hierarchical_structure(text)
        entries = []

        # Recorremos la jerarquía completa: Títulos -> Secciones -> Artículos
        for titulo_elem in estructura_documento:
            titulo_nombre = titulo_elem.get("titulo", "General")
            
            for seccion_elem in titulo_elem.get("secciones", []):
                seccion_nombre = seccion_elem.get("seccion", "General")
                
                for articulo_elem in seccion_elem.get("articulos", []):
                    articulo_num = articulo_elem.get("numero_articulo", "S/N")
                    partes = articulo_elem.get("partes", [])
                    
                    if not partes:
                        articulo_texto = articulo_elem.get("texto_completo", "")
                        partes = [{"id": "Único", "texto": articulo_texto}]

                    entry = {
                        "constitucion": str(year),
                        "titulo": titulo_nombre,
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
    """Exporta las entradas procesadas a un archivo JSONL."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    output_path = Path("data/constituciones.jsonl")
    
    # Asegúrate de eliminar el archivo JSONL previo si deseas regenerarlo desde cero
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"⚠️ El archivo {output_path} ya existe. Bórralo si deseas regenerarlo con la nueva estructura jerárquica.")
    else:
        folder = Path("data/CONSTITUCION")
        anios = ["1994", "2002", "2010", "2015", "2024"]

        all_entries = []

        for año in anios:
            for file_path in folder.glob(f"*{año}*.pdf"):
                print(f"Procesando {file_path.name} ({año})...")
                entries = procesar_constitucion(file_path, año)
                all_entries.extend(entries)

        exportar_jsonl(all_entries, output_path)
        print(f"✅ Dataset jerárquico exportado exitosamente en {output_path}")