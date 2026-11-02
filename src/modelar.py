"""Pasos 2 y 3: modelado con BERTopic y comparación de configuraciones.

- Embeddings: paraphrase-multilingual-MiniLM-L12-v2 (se calculan una vez y se
  guardan en data/embeddings.npy para que todas las configuraciones usen los
  mismos vectores).
- UMAP con random_state=42 → resultados reproducibles.
- CountVectorizer con stopwords en español.
- Se comparan 3 configuraciones (min_cluster_size, ngram_range, nr_topics).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embeddings import cargar_modelo  # noqa: E402

SEED = 42

# Stopwords: lista estándar del español (paquete stop-words, 652 términos)
# + interjecciones y muletillas del diálogo que no aportan contenido temático,
# y el vocativo «Nástenka» (aparece en ~21 % de los segmentos y dominaba
# la representación de casi todos los tópicos sin distinguirlos).
EXTRA = """oh ah ay uf eh vaya vamos mire mira oiga escuche escucha
dígame dijo dije decir digo diga sé verdad cosa cosas momento vez veces
señor querido querida ahora entonces después pronto siempre nunca nada
algo todo toda todos todas aquí allí ser estar haber hacer
nástenka""".split()
STOPWORDS = sorted(set(get_stop_words("spanish")) | set(EXTRA))

CONFIGS = {
    # C1: grano grueso, solo unigramas
    "C1_gruesa": dict(min_cluster_size=15, min_samples=None, ngram=(1, 1), nr_topics=None),
    # C2: grano fino, bigramas y min_samples=3 (HDBSCAN menos conservador)
    "C2_fina": dict(min_cluster_size=10, min_samples=3, ngram=(1, 2), nr_topics=None),
    # C3: C2 + reducción a 7 tópicos (fusión jerárquica de tópicos similares)
    "C3_reducida": dict(min_cluster_size=10, min_samples=3, ngram=(1, 2), nr_topics=8),
}


def construir(cfg, representation=None):
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=SEED)
    hdb = HDBSCAN(min_cluster_size=cfg["min_cluster_size"], min_samples=cfg["min_samples"],
                  metric="euclidean", cluster_selection_method="eom",
                  prediction_data=True)
    vec = CountVectorizer(stop_words=STOPWORDS, ngram_range=cfg["ngram"], min_df=2,
                          token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    return BERTopic(embedding_model=None, umap_model=umap_model, hdbscan_model=hdb,
                    vectorizer_model=vec, nr_topics=cfg["nr_topics"],
                    representation_model=representation, language="multilingual",
                    calculate_probabilities=False, verbose=False)


def npmi_coherencia(topic_model, docs, topn=10):
    """NPMI medio de los pares de palabras top de cada tópico (co-ocurrencia
    a nivel de segmento). Rango [-1, 1]; > 0 indica palabras que co-ocurren."""
    vec = CountVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2), min_df=1, binary=True,
                          token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    X = vec.fit_transform(docs).tocsc()
    vocab = vec.vocabulary_
    N = X.shape[0]
    p = lambda cols: X[:, cols].multiply(X[:, cols[1]]).sum() if len(cols) == 2 else None  # noqa
    res = {}
    for t, words in topic_model.get_topics().items():
        if t == -1:
            continue
        ws = [w for w, _ in words[:topn] if w in vocab]
        vals = []
        for a, b in combinations(ws, 2):
            ia, ib = vocab[a], vocab[b]
            pa = X[:, ia].sum() / N
            pb = X[:, ib].sum() / N
            pab = X[:, ia].multiply(X[:, ib]).sum() / N
            if pab == 0:
                vals.append(-1.0)
            else:
                vals.append(np.log(pab / (pa * pb)) / -np.log(pab))
        if vals:
            res[t] = float(np.mean(vals))
    return res


def diversidad(topic_model, topn=10):
    tops = [w for t, ws in topic_model.get_topics().items() if t != -1 for w, _ in ws[:topn]]
    return len(set(tops)) / max(len(tops), 1)


def main():
    df = pd.read_csv(ROOT / "data" / "segmentos.csv")
    docs = df.texto.tolist()

    emb_path = ROOT / "data" / "embeddings.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        emb = cargar_modelo().encode(docs, batch_size=32, show_progress_bar=True)
        np.save(emb_path, emb)
    print("embeddings", emb.shape)

    filas = []
    modelos = {}
    for nombre, cfg in CONFIGS.items():
        tm = construir(cfg)
        topics, _ = tm.fit_transform(docs, embeddings=emb)
        topics = np.array(topics)
        coh = npmi_coherencia(tm, docs)
        filas.append({
            "configuracion": nombre,
            "min_cluster_size": cfg["min_cluster_size"],
            "ngram_range": str(cfg["ngram"]),
            "nr_topics": cfg["nr_topics"] or "—",
            "n_topicos": int(len(set(topics)) - (1 if -1 in topics else 0)),
            "outliers_%": round(100 * (topics == -1).mean(), 1),
            "NPMI_medio": round(np.mean(list(coh.values())), 3),
            "diversidad": round(diversidad(tm), 3),
        })
        modelos[nombre] = tm
        info = tm.get_topic_info()
        info.to_csv(ROOT / "resultados" / f"topic_info_{nombre}.csv", index=False)
        print(nombre, filas[-1])
        print(info[["Topic", "Count", "Name"]].to_string())

    comp = pd.DataFrame(filas)
    comp.to_csv(ROOT / "resultados" / "comparacion_configuraciones.csv", index=False)
    print(comp.to_string())


if __name__ == "__main__":
    main()
