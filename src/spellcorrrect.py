import os
import urllib.request
from symspellpy import SymSpell, Verbosity
import re
# 1. URL directa al corpus de frecuencias en español (50,000 palabras más usadas)
URL_CORPUS_ES = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"
ARCHIVO_DICCIONARIO = "data/frequency_dictionary_es.txt"

def descargar_corpus_espanol():
    """Descarga el diccionario de frecuencias en español si no existe localmente."""
    if not os.path.exists(ARCHIVO_DICCIONARIO):
        print("📥 Descargando corpus de frecuencias en español (es_50k.txt)...")
        try:
            # Descargar archivo directamente desde el repo
            urllib.request.urlretrieve(URL_CORPUS_ES, ARCHIVO_DICCIONARIO)
            print("✅ Corpus descargado y guardado como 'frequency_dictionary_es.txt'.")
        except Exception as e:
            print(f"❌ Error al descargar el corpus: {e}")
import re

class CorrectorConsultas:
    def __init__(self):
        # Aseguramos que el archivo existe
        descargar_corpus_espanol()
        
        # Inicializamos SymSpell
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Limpiamos o filtramos el archivo antes de cargarlo para evitar anomalías de OCR (como 'k9')
        self._cargar_diccionario_filtrado(ARCHIVO_DICCIONARIO)

    def _cargar_diccionario_filtrado(self, ruta_archivo):
        """
        Carga el diccionario en memoria omitiendo tokens sospechosos de OCR 
        (por ejemplo, palabras que mezclan letras y dígitos como 'k9', 'l9').
        Luego los carga en SymSpell de forma segura.
        """
        import os
        if not os.path.exists(ruta_archivo):
            return

        with open(ruta_archivo, "r", encoding="utf-8") as f:
            for linea in f:
                partes = linea.split()
                if not partes:
                    continue
                palabra = partes[0].lower()
                frecuencia = int(partes[1]) if len(partes) > 1 else 1
                
                # Excluir tokens de OCR que mezclan letras y números (ej. 'k9', '1a', etc.)
                if re.search(r'^[a-z]+\d+|\d+[a-z]+$', palabra):
                    continue
                
                # Cargar término válido uno a uno en el diccionario de SymSpell
                self.sym_spell.create_dictionary_entry(palabra, frecuencia)

    def corregir_y_separar(self, texto: str) -> str:
        """
        Corrige oraciones completas (incluyendo palabras pegadas por error del usuario) 
        utilizando lookup_compound, que respeta la estructura de la frase.
        """
        # lookup_compound utiliza max_edit_distance en lugar de max_dictionary_edit_distance
        sugerencias = self.sym_spell.lookup_compound(
            texto, 
            max_edit_distance=2
        )
        
        if sugerencias:
            return sugerencias[0].term
        
        # Si por alguna razón no devuelve nada, devolvemos el texto original para no perderlo
        return texto
# ==========================================
# PRUEBA DE INTEGRACIÓN
# ==========================================
if __name__ == "__main__":
    corrector = CorrectorConsultas()
    