# Tópicos en *Noches Blancas* (Dostoyevski) con BERTopic

Proyecto de modelado de tópicos sobre una traducción al español de *Noches Blancas*
(1848), segmentada por párrafos.

## Estructura
```
data/noches_blancas_es.txt     corpus (capítulos marcados con "## ")
data/segmentos.csv             505 segmentos (salida de src/segmentar.py)
data/embeddings.npy            embeddings precalculados (505 x 384)
src/embeddings.py              modelo paraphrase-multilingual-MiniLM-L12-v2
src/segmentar.py               Paso 1: segmentación + estadísticas + histograma
src/modelar.py                 Paso 2: comparación de 3 configuraciones
src/final.py                   Paso 3: modelo final, outliers y visualizaciones
resultados/                    tablas (CSV/JSON) y etiquetas de los tópicos
figuras/                       PNG para el reporte + HTML interactivos de BERTopic
noches_blancas_bertopic.ipynb  notebook listo para Google Colab
```

## Cómo reproducir
```bash
pip install -r requirements.txt
python src/segmentar.py
python src/modelar.py
python src/final.py
```
`random_state=42` está fijado en UMAP. Si `data/embeddings.npy` existe se reutiliza,
por lo que los tópicos son idénticos a los del reporte. Si se borra, se recalculan con
`sentence-transformers` (diferencias numéricas mínimas pueden mover algún segmento).

## Fuente del texto
Traducción al español hecha con Claude (IA) a partir de la traducción inglesa de
Constance Garnett (1918, dominio público), párrafo por párrafo y conservando la
división original en párrafos y capítulos.
