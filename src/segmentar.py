"""Paso 1: segmentación de *Noches Blancas* (traducción al español).

Unidad de análisis: el PÁRRAFO (criterio para libro/novela).
Excepción: si un párrafo supera el límite del modelo de embeddings
(128 tokens en paraphrase-multilingual-MiniLM-L12-v2), se divide en
bloques consecutivos de 3–5 oraciones que no superen ese límite, para
que el modelo no trunque el texto.

Filtros: segmentos vacíos, < 5 palabras y duplicados exactos.
Salida: data/segmentos.csv + figuras/histograma_longitudes.png
"""
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embeddings import contar_tokens  # noqa: E402

MAX_TOKENS = 128          # límite de secuencia del modelo de embeddings
MIN_PALABRAS = 5
MIN_ORAC, MAX_ORAC = 3, 5  # tamaño de bloque al partir párrafos largos

# Divide en oraciones tras . ! ? … (con comillas/paréntesis de cierre opcionales)
# seguidos de espacio y una mayúscula, raya de diálogo o signo de apertura.
_SENT_RE = re.compile(r'([\.\!\?…][»"\)]?)\s+(?=[—«¿¡\(A-ZÁÉÍÓÚÑ])')


def oraciones(texto: str) -> list[str]:
    marcado = _SENT_RE.sub(lambda m: m.group(1) + "\x00", texto)
    return [s.strip() for s in marcado.split("\x00") if s.strip()]


def _frases(s: str) -> list[str]:
    """Parte una oración demasiado larga (> MAX_TOKENS) por ; : o rayas,
    agrupando las piezas para que cada una quepa en el modelo."""
    piezas = [x.strip() for x in re.split(r'(?<=[;:])\s+|\s+(?=—)', s) if x.strip()]
    out, actual = [], ""
    for x in piezas:
        cand = (actual + " " + x).strip()
        if actual and contar_tokens(cand) > MAX_TOKENS:
            out.append(actual)
            actual = x
        else:
            actual = cand
    if actual:
        out.append(actual)
    return out


def partir_parrafo(p: str) -> list[str]:
    """Devuelve [p] si cabe en el modelo; si no, bloques consecutivos de
    hasta MIN_ORAC (3) oraciones, sin superar MAX_TOKENS. Las oraciones que
    por sí solas exceden el límite se parten en cláusulas (; : —)."""
    if contar_tokens(p) <= MAX_TOKENS:
        return [p]
    unidades = []
    for s in oraciones(p):
        unidades.extend(_frases(s) if contar_tokens(s) > MAX_TOKENS else [s])
    bloques, actual = [], []
    for s in unidades:
        cand = " ".join(actual + [s])
        if actual and (len(actual) >= MIN_ORAC or contar_tokens(cand) > MAX_TOKENS):
            bloques.append(" ".join(actual))
            actual = [s]
        else:
            actual.append(s)
    if actual:
        resto = " ".join(actual)
        # Un resto muy corto se une al bloque anterior si cabe
        if bloques and contar_tokens(resto) < 20 and \
                contar_tokens(bloques[-1] + " " + resto) <= MAX_TOKENS:
            bloques[-1] += " " + resto
        else:
            bloques.append(resto)
    return bloques


def main():
    texto = (ROOT / "data" / "noches_blancas_es.txt").read_text(encoding="utf-8")
    capitulo, filas = None, []
    n_parrafos = 0
    for bloque in texto.split("\n\n"):
        bloque = bloque.strip()
        if not bloque:
            continue
        if bloque.startswith("## "):
            capitulo = bloque[3:].strip().title()
            continue
        n_parrafos += 1
        partes = partir_parrafo(bloque)
        for j, seg in enumerate(partes):
            filas.append({"capitulo": capitulo, "parrafo_id": n_parrafos,
                          "sub_id": j, "partido": len(partes) > 1, "texto": seg})

    df = pd.DataFrame(filas)
    n_bruto = len(df)
    df["palabras"] = df.texto.str.split().str.len()
    df["tokens"] = df.texto.map(contar_tokens)
    vacios_cortos = (df.palabras < MIN_PALABRAS).sum()
    df = df[df.palabras >= MIN_PALABRAS]
    dups = df.duplicated("texto").sum()
    df = df.drop_duplicates("texto").reset_index(drop=True)
    df.insert(0, "seg_id", range(len(df)))
    df.to_csv(ROOT / "data" / "segmentos.csv", index=False)

    stats = {
        "parrafos_originales": n_parrafos,
        "parrafos_partidos": int(df.groupby("parrafo_id").partido.first().sum()),
        "segmentos_brutos": n_bruto,
        "eliminados_<5_palabras": int(vacios_cortos),
        "eliminados_duplicados": int(dups),
        "segmentos_finales": len(df),
        "palabras_media": round(df.palabras.mean(), 1),
        "palabras_mediana": float(df.palabras.median()),
        "palabras_min": int(df.palabras.min()),
        "palabras_max": int(df.palabras.max()),
        "tokens_media": round(df.tokens.mean(), 1),
        "segmentos_>128_tokens": int((df.tokens > MAX_TOKENS).sum()),
    }
    pd.Series(stats).to_csv(ROOT / "resultados" / "estadisticas_segmentacion.csv", header=["valor"])
    print(pd.Series(stats).to_string())
    print(df.groupby("capitulo", sort=False).size().to_string())

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(df.palabras, bins=30, color="#2a6f97", edgecolor="white")
    ax.axvline(df.palabras.mean(), color="#c0392b", ls="--", lw=1.2,
               label=f"media = {df.palabras.mean():.1f}")
    ax.set_xlabel("Longitud del segmento (palabras)")
    ax.set_ylabel("Número de segmentos")
    ax.set_title("Distribución de longitudes de los segmentos")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figuras" / "histograma_longitudes.png", dpi=200)


if __name__ == "__main__":
    main()
