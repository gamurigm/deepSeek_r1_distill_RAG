# Cómo crear el proyecto R1 Distill RAG desde cero (sin repo)

Esta guía te permite levantar el sistema RAG desde cero, generando todos los archivos y configuraciones necesarios.

---

## 1. Pre-requisitos
- Python 3.10 o superior
- Git
- [Ollama](https://ollama.ai) instalado y corriendo
- (Opcional) Cuenta y token de Hugging Face

---

## 2. Estructura de carpetas

```bash
mkdir -p nombre_proyecto/{data,ollama_models,chroma_db}
cd nombre_proyecto
```

---

## 3. Inicializa un repositorio Git (opcional)

```bash
git init
```

---

## 4. Crea y activa un entorno virtual

```bash
python -m venv venv
source venv/bin/activate
```

---

## 5. Crea el archivo `requirements.txt`

```
aiofiles==23.2.1
aiohappyeyeballs==2.4.4
aiohttp==3.11.11
aiosignal==1.3.2
altair==5.5.0
annotated-types==0.7.0
anyio==4.8.0
asgiref==3.8.1
attrs==25.1.0
backoff==2.2.1
bcrypt==4.2.1
beautifulsoup4==4.12.3
blinker==1.9.0
build==1.2.2.post1
cachetools==5.5.1
certifi==2025.1.31
charset-normalizer==3.4.1
chroma-hnswlib==0.7.6
chromadb==0.6.3
click==8.1.8
colorama==0.4.6
coloredlogs==15.0.1
dataclasses-json==0.6.7
Deprecated==1.2.18
distro==1.9.0
duckduckgo_search==7.3.0
durationpy==0.9
fastapi==0.115.8
ffmpy==0.5.0
filelock==3.17.0
flatbuffers==25.1.24
frozenlist==1.5.0
fsspec==2024.12.0
gitdb==4.0.12
GitPython==3.1.44
google-auth==2.38.0
googleapis-common-protos==1.66.0
gradio==5.14.0
gradio_client==1.7.0
greenlet==3.1.1
grpcio==1.70.0
h11==0.14.0
httpcore==1.0.7
httptools==0.6.4
httpx==0.27.2
httpx-sse==0.4.0
huggingface-hub==0.28.1
humanfriendly==10.0
idna==3.10
importlib_metadata==8.5.0
importlib_resources==6.5.2
Jinja2==3.1.5
jiter==0.8.2
joblib==1.4.2
jsonpatch==1.33
jsonpointer==3.0.0
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
kubernetes==32.0.0
langchain==0.3.17
langchain-chroma==0.2.1
langchain-community==0.3.16
langchain-core==0.3.33
langchain-huggingface==0.1.2
langchain-text-splitters==0.3.5
langsmith==0.3.3
litellm==1.59.10
lxml==5.3.0
markdown-it-py==3.0.0
markdownify==0.14.1
MarkupSafe==2.1.5
marshmallow==3.26.0
mdurl==0.1.2
mmh3==5.1.0
monotonic==1.6
mpmath==1.3.0
multidict==6.1.0
mypy-extensions==1.0.0
narwhals==1.24.1
networkx==3.4.2
numpy==1.26.4
oauthlib==3.2.2
onnxruntime==1.20.1
openai==1.60.2
opentelemetry-api==1.29.0
opentelemetry-exporter-otlp-proto-common==1.29.0
opentelemetry-exporter-otlp-proto-grpc==1.29.0
opentelemetry-instrumentation==0.50b0
opentelemetry-instrumentation-asgi==0.50b0
opentelemetry-instrumentation-fastapi==0.50b0
opentelemetry-proto==1.29.0
opentelemetry-sdk==1.29.0
opentelemetry-semantic-conventions==0.50b0
opentelemetry-util-http==0.50b0
orjson==3.10.15
overrides==7.7.0
packaging==24.2
pandas==2.2.3
pillow==11.1.0
posthog==3.11.0
primp==0.11.0
propcache==0.2.1
protobuf==5.29.3
pyarrow==19.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.1
pydantic==2.10.6
pydantic-settings==2.7.1
pydantic_core==2.27.2
pydeck==0.9.1
pydub==0.25.1
Pygments==2.19.1
pypdf==5.2.0
PyPika==0.48.9
pyproject_hooks==1.2.0
pyreadline3==3.5.4
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
python-multipart==0.0.20
pytz==2025.1
PyYAML==6.0.2
referencing==0.36.2
regex==2024.11.6
requests==2.32.3
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
rich==13.9.4
rpds-py==0.22.3
rsa==4.9
ruff==0.9.4
safehttpx==0.1.6
safetensors==0.5.2
scikit-learn==1.6.1
scipy==1.15.1
semantic-version==2.10.0
sentence-transformers==3.4.1
setuptools==75.8.0
shellingham==1.5.4
six==1.17.0
smmap==5.0.2
smolagents==1.6.0
sniffio==1.3.1
soupsieve==2.6
SQLAlchemy==2.0.37
starlette==0.45.3
streamlit==1.41.1
sympy==1.13.1
tenacity==9.0.0
threadpoolctl==3.5.0
tiktoken==0.8.0
tokenizers==0.21.0
toml==0.10.2
tomlkit==0.13.2
torch==2.6.0
tornado==6.4.2
tqdm==4.67.1
transformers==4.48.2
typer==0.15.1
typing-inspect==0.9.0
typing_extensions==4.12.2
tzdata==2025.1
urllib3==2.3.0
uvicorn==0.34.0
watchdog==6.0.0
watchfiles==1.0.4
websocket-client==1.8.0
websockets==14.2
wrapt==1.17.2
yarl==1.18.3
zipp==3.21.0
zstandard==0.23.0

```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## 6. Crea el archivo `.env`

```
USE_HUGGINGFACE=no
HUGGINGFACE_API_TOKEN=
REASONING_MODEL_ID=deepseek-r1:8b-llama-distill-q4_K_M
TOOL_MODEL_ID=llama3.1:8b
```

---

## 7. Descarga los modelos en Ollama

```bash
ollama pull deepseek-r1:8b-llama-distill-q4_K_M
ollama pull llama3.1:8b
```

---

## 8. Crea los archivos principales

### a) `ingest_pdfs.py`

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

def load_and_process_pdfs(data_dir: str):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory: str):
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    print("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir)
    print(f"Created {len(chunks)} chunks from PDFs")
    print("Creating vector store...")
    vectordb = create_vector_store(chunks, db_dir)
    print(f"Vector store created and persisted at {db_dir}")

if __name__ == "__main__":
    main()
```

---

### b) `r1_smolagent_rag.py`

```python
from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, tool, GradioUI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")

def get_model(model_id):
    return OpenAIServerModel(
        model_id=model_id,
        api_base="http://localhost:11434/v1",
        api_key="ollama"
    )

reasoning_model = get_model(reasoning_model_id)
reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

@tool
def rag_with_reasoner(user_query: str) -> str:
    docs = vectordb.similarity_search(user_query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Based on the following context, answer the user's question. Be concise and specific.\nIf there isn't sufficient information, give as your answer a better query to perform RAG with.\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"""
    response = reasoner.run(prompt, reset=False)
    return response

tool_model = get_model(tool_model_id)
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner], model=tool_model, add_base_tools=False, max_steps=3)

def main():
    GradioUI(primary_agent).launch()

if __name__ == "__main__":
    main()
```
---


## 9. Coloca tus PDFs en la carpeta `data`

```bash
mkdir -p data
# Copia tus PDFs aquí
```

---

## 10. Procesa los PDFs

```bash
python ingest_pdfs.py
```

---

## 11. Ejecuta la aplicación

```bash
python r1_smolagent_rag.py
```

---

