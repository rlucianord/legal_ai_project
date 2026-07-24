# Legal AI Project - Sistema de Consultas Constitucionales

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Sistema de IA para consultas sobre la Constitución Dominicana utilizando procesamiento de lenguaje natural y modelos de transformers.

## 🚀 Características

- **Extracción de PDFs**: Procesamiento automático de constituciones en formato PDF
- **Preprocesamiento Inteligente**: Normalización de texto y corrección de errores OCR
- **Fine-tuning de BERT**: Entrenamiento de modelo multilingüe para Question Answering
- **Búsqueda Semántica**: Indexación con FAISS y embeddings multilingües
- **Chat Interactivo**: Sistema de consultas legales en lenguaje natural

## 🛠 Tech Stack

### Core Technologies
- **Python 3.8+**: Lenguaje principal del proyecto
- **PyTorch**: Framework de deep learning para entrenamiento de modelos
- **Transformers (Hugging Face)**: Modelos pre-entrenados BERT para NLP
- **FAISS**: Biblioteca de búsqueda similitud vectorial eficiente

### NLP & Text Processing
- **sentence-transformers**: Embeddings multilingües para búsqueda semántica
- **pdfminer.six**: Extracción de texto desde archivos PDF
- **PyPDF2**: Biblioteca alternativa para procesamiento de PDF
- **Regex**: Pattern matching para división de artículos y secciones

### Data & ML
- **pandas**: Manipulación de datos estructurados
- **numpy**: Computación numérica
- **scikit-learn**: Herramientas de machine learning
- **datasets (Hugging Face)**: Gestión de datasets para entrenamiento

### Architecture Pattern
- **Question Answering**: Extractive QA con BERT multilingüe
- **Retrieval-Augmented Generation**: Búsqueda semántica + respuesta precisa
- **Modular Design**: Separación clara entre procesamiento, entrenamiento e inferencia

## 📋 Requisitos del Sistema

- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows, Linux, o macOS
- **Hardware**: 
  - CPU: Procesador moderno (mínimo 4 cores recomendado)
  - RAM: 8GB mínimo, 16GB recomendado
  - GPU: Opcional pero recomendada para entrenamiento (NVIDIA con CUDA)
- **Espacio en Disco**: 5GB+ para modelos y datos

## 🔧 Configuración del Entorno

### 1. Clonar el Repositorio

```bash
git clone https://github.com/rlucianord/legal_ai_project/edit/main/README.md
cd legal_ai_project
```

### 2. Crear Entorno Virtual

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar Instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 5. Configuración Opcional (GPU)

Si tienes GPU NVIDIA, instala CUDA y cuDNN según tu versión de PyTorch:

```bash
# Verificar disponibilidad de GPU
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## 📁 Estructura del Proyecto

```
legal_ai_project/
├── data/
│   ├── CONSTITUCION/          # Archivos PDF de constituciones (1994, 2002, 2010, 2015, 2024)
│   └── constituciones.jsonl   # Dataset procesado en formato JSONL
├── models/
│   └── checkpoints/           # Modelos fine-tuned de BERT para QA
├── src/
│   ├── __init__.py            # Init del paquete
│   ├── dataset_builder.py     # Extracción y procesamiento de PDFs
│   ├── preprocessing.py       # Limpieza y normalización de texto
│   ├── modeltrainer.py        # Entrenamiento y sistema de chat
│   └── utils.py               # Utilidades generales
├── notebooks/                 # Jupyter notebooks para análisis exploratorio
├── .venv/                     # Entorno virtual Python (gitignore)
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación del proyecto
```

## 🚀 Uso del Proyecto

### Paso 1: Preparar Datos

Coloca los archivos PDF de las constituciones en `data/CONSTITUCION/` con el año en el nombre (ej: `constitucion_1994.pdf`, `constitucion_2002.pdf`).

```bash
cd src
python dataset_builder.py
```

Este script:
- Procesa todos los PDFs encontrados en el directorio
- Extrae y estructura artículos por secciones
- Genera `data/constituciones.jsonl` con el dataset procesado

### Paso 2: Entrenar Modelo

```bash
python modeltrainer.py
```

El proceso de entrenamiento incluye:
1. **Preparación del Dataset**: Carga y tokenización para fine-tuning
2. **Entrenamiento BERT**: Fine-tuning de modelo multilingüe para QA
3. **Indexación FAISS**: Construcción de índice semántico con embeddings
4. **Guardado de Modelo**: Almacenamiento en `models/checkpoints/`

**Nota**: El entrenamiento puede tomar varias horas dependiendo del hardware.

### Paso 3: Usar el Sistema de Chat

```python
from src.modeltrainer import LegalChatBot

# Inicializar el chatbot
chatbot = LegalChatBot()

# Realizar consultas
respuesta = chatbot.chat("¿Quién ejerce la soberanía nacional?")
print(respuesta)

# Otra consulta
respuesta = chatbot.chat("¿Cuáles son los derechos del trabajador?")
print(respuesta)
```

## 📚 Documentación de Funciones

### dataset_builder.py

- **`procesar_constitucion(file_path, year)`**: Procesa un PDF individual y extrae artículos estructurados
- **`exportar_jsonl(entries, output_file)`**: Exporta el dataset procesado a formato JSONL

### preprocessing.py

- **`extract_text(file_path, use_pdfminer=True)`**: Extrae texto de PDF con fallback automático
- **`preprocess_text(text)`**: Aplica limpieza, normalización y corrección de errores
- **`split_sections(text)`**: Divide el texto en secciones (Títulos, Secciones)
- **`split_articles(text)`**: Identifica y extrae artículos individuales
- **`split_parts(text)`**: Divide artículos en incisos y partes

### modeltrainer.py

- **`prepare_dataset(data_path)`**: Prepara dataset para fine-tuning de BERT
- **`train_model(dataset, checkpoint_dir)`**: Entrena modelo BERT para QA
- **`build_index(data_path)`**: Construye índice FAISS con embeddings multilingües
- **`load_qa_pipeline(checkpoint_dir)`**: Carga pipeline de question answering
- **`LegalChatBot`**: Clase principal para el sistema de consultas

## 🔍 Arquitectura del Sistema

```
PDF Input → Text Extraction → Preprocessing → Structured Data (JSONL)
                                                    ↓
                                            Dataset Preparation
                                                    ↓
                                            BERT Fine-tuning
                                                    ↓
                                            Embedding Generation
                                                    ↓
                                            FAISS Indexing
                                                    ↓
                                            Query Processing
                                                    ↓
                                            Semantic Search + QA
                                                    ↓
                                            Response Generation
```

## ⚙️ Configuración Avanzada

### Ajustar Hiperparámetros de Entrenamiento

Edita `src/modeltrainer.py`:

```python
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=4,  # Aumentar si tienes más GPU RAM
    num_train_epochs=3,              # Más épocas = mejor calidad
    learning_rate=2e-5,              # Tasa de aprendizaje
    warmup_steps=500,                # Warmup para estabilidad
    save_steps=1000,
    save_total_limit=3,
    logging_dir="logs"
)
```

### Cambiar Modelo de Embeddings

```python
# Opciones multilingües:
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Actual
embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")   # Más rápido
embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")  # Más preciso
```

## 🐛 Solución de Problemas

### Error: "CUDA out of memory"
- Reduce `per_device_train_batch_size` en `modeltrainer.py`
- Usa CPU eliminando `.to(device)` o usa `device='cpu'`

### Error: "File not found"
- Verifica que los PDFs estén en `data/CONSTITUCION/`
- Asegúrate de que los nombres contengan el año (ej: `1994`)

### Error: "Module not found"
- Activa el entorno virtual: `.venv\Scripts\activate`
- Reinstala dependencias: `pip install -r requirements.txt`

### Resultados de baja calidad
- Aumenta `num_train_epochs` a 3-5
- Usa un modelo base más grande: `bert-large-multilingual-cased`
- Aumenta el dataset con más constituciones

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 🇩🇴 Invitación a Programadores Dominicanos

¡Se invita a programadores dominicanos a colaborar en este proyecto para introducir todos los documentos legales dominicanos disponibles en el repositorio legal de la República Dominicana!

### Perfil Buscado

- Programadores con experiencia en Python y NLP (Procesamiento de Lenguaje Natural)
- Conocimiento en machine learning, transformers y modelos de lenguaje
- Experiencia con procesamiento de documentos PDF y estructuración de datos
- Interés en democratizar el acceso a la información legal mediante tecnología
- Compromiso con la transparencia y el acceso igualitario a la justicia

### Objetivo del Proyecto

El proyecto busca indexar y hacer consultable toda la documentación legal dominicana disponible en el repositorio oficial, permitiendo que cualquier ciudadano pueda acceder y consultar documentos legales de manera sencilla mediante lenguaje natural.

### Reglas para Commits

**Formato de Commit Messages:**
```
<tipo>: <descripción breve>

<descripción detallada opcional>
```

**Tipos permitidos:**
- `feat`: Nueva funcionalidad
- `fix`: Corrección de bugs
- `docs`: Cambios en documentación
- `style`: Formateo de código (sin lógica)
- `refactor`: Refactorización de código
- `test`: Agregar o modificar tests
- `chore`: Mantenimiento, dependencias, etc.

**Ejemplos:**
```
feat: agregar soporte para constitución de 2024
fix: corregir error en extracción de artículos
docs: actualizar README con instrucciones de instalación
refactor: mejorar estructura de preprocessing.py
```

### Reglas para Pull Requests

**Antes de abrir un PR:**
1. **Sincroniza tu fork** con el repositorio principal
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Crea una rama descriptiva** para tu cambio
   ```bash
   git checkout -b feature/nueva-funcionalidad
   # o
   git checkout -b fix/correcion-bug
   ```

3. **Escribe commits claros** siguiendo el formato anterior
   ```bash
   git add .
   git commit -m "feat: descripción clara del cambio"
   ```

4. **Push tu rama** a tu fork
   ```bash
   git push origin feature/nueva-funcionalidad
   ```

5. **Abre el Pull Request** desde GitHub con:
   - Título descriptivo
   - Descripción detallada del cambio
   - Referencias a issues relacionados (si aplica)
   - Capturas de pantalla para cambios visuales

**Revisión e Integración:**
- Todos los PRs serán revisados antes de integrarse
- Se solicitarán cambios si es necesario
- Los PRs deben pasar pruebas básicas
- Se valorará especialmente el conocimiento legal aplicado al código

### Código de Conducta

- Respeto mutuo entre colaboradores
- Comunicación clara y constructiva
- Disponibilidad para explicar cambios técnicos
- Compromiso con la calidad del código
- Colaboración abierta y transparente

## 📝 Roadmap

- [ ] Soporte para más tipos de documentos legales
- [ ] Interfaz web con Streamlit/Gradio
- [ ] API REST para integración externa
- [ ] Multi-idioma (inglés, francés, portugués)
- [ ] Evaluación automática de calidad de respuestas
- [ ] Sistema de retroalimentación usuario

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autor

**Tu Nombre** - [@tu-usuario](https://github.com/tu-usuario)

## 🙏 Agradecimientos

- Hugging Face por los modelos Transformers y datasets
- Meta AI por la biblioteca FAISS
- La comunidad de open-source NLP

## 📞 Contacto

Para preguntas o soporte:
- Abre un issue en el repositorio
- Email: tu-email@ejemplo.com

--- El autor de este proyecto no es abogado, pero considera que la justicia  debe ser igual para todos sin importar raza o color, todos tenemos el derecho a ser tratados con justicia.

---Puedes ser colaborador de este proyecto.

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!
