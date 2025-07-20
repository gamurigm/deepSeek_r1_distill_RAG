# DeepSeekR1-distill-rag

Este proyecto implementa un agente RAG (Retrieval-Augmented Generation) usando modelos HuggingFace y Ollama, con una interfaz Gradio para interacción conversacional y búsqueda contextual.

## Requisitos
- Python 3.10 o superior
- Recomendado: entorno virtual

## Instalación
1. Clona el repositorio y navega a la carpeta del proyecto.
2. Crea y activa un entorno virtual:
   ```zsh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Instala las dependencias:
   ```zsh
   pip install -r requirements.txt
   ```

## Configuración
1. Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:
   ```env
   REASONING_MODEL_ID=<id_modelo_llm>
   TOOL_MODEL_ID=<id_modelo_llm>
   HUGGINGFACE_API_TOKEN=<tu_token_huggingface>
   USE_HUGGINGFACE=yes  # o no, según tu preferencia
   ```

## Procesamiento de PDFs
Para indexar PDFs y construir la base de datos de vectores:
```zsh
python ingest_pdfs.py
```

## Ejecución de la aplicación
Para lanzar la interfaz conversacional:
```zsh
streamlit run streamlit.py
```
O para usar la interfaz Gradio:
```zsh
python r1_smolagent_rag.py
```

## Notas
- Los modelos Ollama deben estar instalados y configurados si se usan como backend.
- La carpeta `data/` debe contener los PDFs a indexar.
- La base de datos de vectores se almacena en `chroma_db/`.

## Estructura principal
- `r1_smolagent_rag.py`: Agente principal y lógica RAG.
- `ingest_pdfs.py`: Indexación de PDFs.
- `streamlit.py`: Interfaz Streamlit.
- `requirements.txt`: Dependencias.
- `data/`: PDFs y README.
- `chroma_db/`: Base de datos de vectores.

## Preguntas frecuentes
- Si tienes problemas con dependencias, revisa la versión de Python y usa entorno virtual.
- Si la app no encuentra modelos, revisa las variables en `.env` y la instalación de Ollama/HuggingFace.
