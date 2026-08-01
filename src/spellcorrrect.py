import os
import urllib.request
from symspellpy import SymSpell, Verbosity
import re

URL_CORPUS_ES = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"
ARCHIVO_DICCIONARIO = "data/frequency_dictionary_es.txt"

def descargar_corpus_espanol():
    """
    Descarga el diccionario de frecuencias en español si no existe localmente.
    
    Descarga el corpus de las 50,000 palabras más usadas en español
    desde GitHub y lo guarda localmente para uso posterior.
    """
    if not os.path.exists(ARCHIVO_DICCIONARIO):
        print("📥 Descargando corpus de frecuencias en español (es_50k.txt)...")
        try:
            urllib.request.urlretrieve(URL_CORPUS_ES, ARCHIVO_DICCIONARIO)
            print("✅ Corpus descargado y guardado como 'frequency_dictionary_es.txt'.")
        except Exception as e:
            print(f"❌ Error al descargar el corpus: {e}")

class CorrectorConsultas:
    """
    Corrector ortográfico para consultas en español usando SymSpell.
    
    Utiliza un diccionario de frecuencias del español para corregir
    errores ortográficos y separar palabras pegadas en las consultas del usuario.
    """
    
    def __init__(self):
        """
        Inicializa el corrector descargando el corpus y cargando el diccionario.
        """
        descargar_corpus_espanol()
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self._cargar_diccionario_filtrado(ARCHIVO_DICCIONARIO)

    def _cargar_diccionario_filtrado(self, ruta_archivo):
        """
        Carga el diccionario filtrando tokens sospechosos de OCR.
        
        Args:
            ruta_archivo: Ruta al archivo del diccionario de frecuencias
        
        Note:
            Excluye tokens que mezclan letras y números (ej: 'k9', '1a')
            que suelen ser artefactos de OCR.
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
                
                if re.search(r'^[a-z]+\d+|\d+[a-z]+$', palabra):
                    continue
                
                self.sym_spell.create_dictionary_entry(palabra, frecuencia)

    def corregir_y_separar(self, texto: str) -> str:
        """
        Corrige oraciones completas incluyendo palabras pegadas por error.
        
        Args:
            texto: Texto a corregir
        
        Returns:
            Texto corregido con palabras separadas y ortografía corregida
        """
        sugerencias = self.sym_spell.lookup_compound(
            texto, 
            max_edit_distance=2
        )
        
        if sugerencias:
            return sugerencias[0].term
        
        return texto

if __name__ == "__main__":
    """
    Prueba de inicialización del módulo spellcorrrect.py.
    """
    corrector = CorrectorConsultas()
    