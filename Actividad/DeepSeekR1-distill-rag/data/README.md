# Data Directory

Este directorio está destinado para almacenar los archivos PDF que serán procesados por el sistema RAG.

## Instrucciones de Uso

1. **Coloca tus archivos PDF aquí** - Puedes agregar uno o más archivos PDF en este directorio
2. **Estructura flexible** - Puedes crear subdirectorios si lo deseas, el sistema los procesará recursivamente
3. **Formatos soportados** - Solo archivos `.pdf` son procesados actualmente

## Ejemplos de Estructura

```
data/
├── documento1.pdf
├── documento2.pdf
├── papers/
│   ├── research_paper1.pdf
│   └── research_paper2.pdf
└── manuales/
    └── manual_usuario.pdf
```

## Procesamiento

Después de agregar tus PDFs, ejecuta:

```bash
python ingest_pdfs.py
```

Este comando:
- Encuentra todos los archivos PDF en este directorio y subdirectorios
- Los procesa y divide en chunks de texto
- Crea una base de datos vectorial para búsqueda semántica

## Notas Importantes

- Los archivos PDF no se incluyen en el repositorio Git por privacidad y tamaño
- Cada vez que agregues nuevos PDFs, debes ejecutar `ingest_pdfs.py` nuevamente
- El procesamiento puede tomar tiempo dependiendo del tamaño y cantidad de archivos
