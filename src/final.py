"""Modelo final (configuración C2) + análisis: tabla de tópicos, ejemplos
representativos, tratamiento de outliers y visualizaciones.

Uso:
  python src/final.py            # entrena, guarda resultados y figuras
Las etiquetas humanas de cada tópico están en resultados/etiquetas.json
(se asignaron leyendo palabras clave + segmentos representativos).
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic.representation import MaximalMarginalRelevance
from scipy.cluster import hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modelar import CONFIGS, SEED, construir  # noqa: E402

RES, FIG = ROOT / "resultados", ROOT / "figuras"
# Umbral para reasignar outliers: percentil 25 de la similitud coseno de los
# segmentos YA asignados con el centroide de su propio tópico. Así un outlier
# solo se reasigna si es tan parecido al tópico como un miembro "típico".
PCTL_UMBRAL = 25
PALETA = ["#2a6f97", "#c0392b", "#e09f3e", "#3a7d44", "#7b4b94", "#d65db1",
          "#4d908e", "#9c6644", "#577590", "#f3722c", "#6a994e", "#bc4749",
          "#90be6d", "#277da1"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def etiquetas():
    p = RES / "etiquetas.json"
    return {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def main():
    df = pd.read_csv(ROOT / "data" / "segmentos.csv")
    docs = df.texto.tolist()
    emb = np.load(ROOT / "data" / "embeddings.npy")

    cfg = CONFIGS["C2_fina"]
    tm = construir(cfg, representation=MaximalMarginalRelevance(diversity=0.3))
    topics, _ = tm.fit_transform(docs, embeddings=emb)
    topics = np.array(topics)
    df["topico"] = topics
    pct_out = 100 * (topics == -1).mean()

    # ---------- tabla de tópicos y ejemplos representativos ----------
    info = tm.get_topic_info()
    rep = tm.get_representative_docs()
    labels = etiquetas()
    info["Etiqueta"] = info.Topic.map(lambda t: labels.get(t, ""))
    info["Palabras clave"] = info.Topic.map(lambda t: ", ".join(w for w, _ in tm.get_topic(t)[:8]))
    info[["Topic", "Count", "Etiqueta", "Palabras clave", "Name"]].to_csv(
        RES / "tabla_topicos.csv", index=False)
    with open(RES / "segmentos_representativos.json", "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in rep.items()}, f, ensure_ascii=False, indent=1)

    # ---------- outliers: reasignación por similitud de embeddings ----------
    # Se reasigna un outlier solo si su similitud con el centroide del tópico
    # más cercano supera el umbral; el resto se queda como -1 (ruido real).
    ids = [t for t in sorted(set(topics)) if t != -1]
    centroides = np.vstack([emb[topics == t].mean(0) for t in ids])
    sim = cosine_similarity(emb, centroides)
    propia = np.array([sim[i, ids.index(t)] for i, t in enumerate(topics) if t != -1])
    UMBRAL_OUTLIERS = round(float(np.percentile(propia, PCTL_UMBRAL)), 3)
    nuevos = topics.copy()
    m = topics == -1
    mejor = sim.argmax(1)
    mejor_sim = sim.max(1)
    reasignar = m & (mejor_sim >= UMBRAL_OUTLIERS)
    nuevos[reasignar] = np.array(ids)[mejor[reasignar]]
    # comprobación con la función nativa de BERTopic
    nativo = np.array(tm.reduce_outliers(docs, topics, strategy="embeddings",
                                         embeddings=emb, threshold=UMBRAL_OUTLIERS))
    df["topico_final"] = nuevos
    df["sim_topico"] = mejor_sim
    df.to_csv(RES / "segmentos_con_topicos.csv", index=False)
    out_stats = {
        "outliers_iniciales": int(m.sum()),
        "outliers_iniciales_%": round(pct_out, 1),
        "umbral_similitud": UMBRAL_OUTLIERS,
        "criterio_umbral": f"percentil {PCTL_UMBRAL} de la similitud miembro-centroide",
        "reasignados": int(reasignar.sum()),
        "outliers_finales": int((nuevos == -1).sum()),
        "outliers_finales_%": round(100 * (nuevos == -1).mean(), 1),
        "coincidencia_con_reduce_outliers_%": round(100 * (nativo == nuevos).mean(), 1),
        "sim_mediana_outliers": round(float(np.median(mejor_sim[m])), 3),
        "sim_mediana_no_outliers": round(float(np.median(mejor_sim[~m])), 3),
    }
    json.dump(out_stats, open(RES / "outliers.json", "w"), indent=1)
    print(json.dumps(out_stats, indent=1))
    print(info[["Topic", "Count", "Name"]].to_string())

    # Cambios de tamaño por tópico tras reasignar
    tam = pd.DataFrame({"antes": pd.Series(topics).value_counts(),
                        "despues": pd.Series(nuevos).value_counts()}).fillna(0).astype(int)
    tam.sort_index().to_csv(RES / "tamanos_antes_despues.csv")

    # Distribución de los outliers por capítulo
    df.groupby("capitulo", sort=False).apply(
        lambda g: round(100 * (g.topico == -1).mean(), 1), include_groups=False
    ).to_csv(RES / "outliers_por_capitulo.csv", header=["outliers_%"])

    nombre = lambda t: labels.get(t, f"T{t}") if t != -1 else "Outliers"  # noqa: E731
    tids = [t for t in info.Topic if t != -1]
    color = {t: PALETA[i % len(PALETA)] for i, t in enumerate(tids)}

    # ---------- Fig 1: barchart de palabras clave (c-TF-IDF) ----------
    n = len(tids)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.35 * rows))
    for ax, t in zip(axes.flat, tids):
        ws = tm.get_topic(t)[:6][::-1]
        ax.barh([w for w, _ in ws], [s for _, s in ws], color=color[t])
        ax.set_title(textwrap.fill(f"{t}. {nombre(t)}", 30), fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([])
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Palabras más representativas por tópico (c-TF-IDF + MMR)", y=1.0,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "barchart_topicos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 2: mapa 2D de documentos ----------
    xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine",
              random_state=SEED).fit_transform(emb)
    np.save(RES / "umap2d.npy", xy)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.scatter(xy[topics == -1, 0], xy[topics == -1, 1], s=9, c="#cfcfcf",
               label="Outliers (−1)", zorder=1)
    for t in tids:
        p = xy[topics == t]
        ax.scatter(p[:, 0], p[:, 1], s=14, color=color[t], alpha=.85,
                   label=f"{t}. {nombre(t)}", zorder=2)
        cx, cy = np.median(p, 0)
        ax.text(cx, cy, str(t), fontsize=10, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=color[t], lw=1.2), zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.legend(fontsize=7.2, frameon=False, loc="center left", bbox_to_anchor=(1, .5))
    ax.set_title("Mapa de segmentos (UMAP 2D de los embeddings)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "mapa_documentos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 3: jerarquía de tópicos ----------
    temb = np.vstack([emb[topics == t].mean(0) for t in tids])
    Z = sch.linkage(temb, method="ward", optimal_ordering=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sch.dendrogram(Z, labels=[f"{t}. {nombre(t)}" for t in tids], orientation="left",
                   color_threshold=0.7 * Z[:, 2].max(), ax=ax, leaf_font_size=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Distancia (Ward sobre centroides de embeddings)")
    ax.set_title("Jerarquía de tópicos", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "jerarquia_topicos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Jerarquía nativa de BERTopic (texto) para el anexo
    hier = tm.hierarchical_topics(docs)
    (RES / "arbol_jerarquico.txt").write_text(tm.get_topic_tree(hier), encoding="utf-8")

    # ---------- Fig 4: tópicos a lo largo de la novela (por capítulo) ----------
    caps = list(dict.fromkeys(df.capitulo))
    tab = pd.crosstab(df.topico, df.capitulo)[caps]
    tab = tab.div(tab.sum(0), axis=1) * 100
    tab = tab.loc[tids]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    im = ax.imshow(tab.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(caps)), [c.replace(" De ", " de ") for c in caps], rotation=20, ha="right")
    ax.set_yticks(range(len(tids)), [f"{t}. {nombre(t)}" for t in tids])
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 25 else "#222")
    fig.colorbar(im, ax=ax, label="% de segmentos del capítulo", shrink=.8)
    ax.set_title("Peso de cada tópico por capítulo (%)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "topicos_por_capitulo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    tab.round(1).to_csv(RES / "topicos_por_capitulo.csv")

    # ---------- visualizaciones interactivas nativas de BERTopic ----------
    if labels:
        tm.set_topic_labels({t: f"{t}. {nombre(t)}" for t in tids})
    try:
        tm.visualize_barchart(top_n_topics=len(tids), n_words=6, custom_labels=bool(labels)
                              ).write_html(FIG / "interactivo_barchart.html")
        tm.visualize_documents(docs, reduced_embeddings=xy, custom_labels=bool(labels),
                               hide_annotations=False).write_html(FIG / "interactivo_documentos.html")
        tm.visualize_hierarchy(hierarchical_topics=hier, custom_labels=bool(labels)
                               ).write_html(FIG / "interactivo_jerarquia.html")
        tm.visualize_heatmap(custom_labels=bool(labels)).write_html(FIG / "interactivo_heatmap.html")
    except Exception as e:  # noqa: BLE001
        print("Aviso: no se pudieron exportar las figuras interactivas:", e)


if __name__ == "__main__":
    main()
