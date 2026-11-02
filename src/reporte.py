"""Genera el reporte PDF (reporte_bertopic_noches_blancas.pdf)."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "resultados", ROOT / "figuras"
F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Serif", F + "DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("Serif-B", F + "DejaVuSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Serif-I", F + "DejaVuSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Sans", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-B", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I",
                              boldItalic="Serif-B")

AZUL = colors.HexColor("#1b3a5c")
S = {
    "body": ParagraphStyle("body", fontName="Serif", fontSize=9.6, leading=13.6,
                           alignment=TA_JUSTIFY, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=13.5, leading=17, textColor=AZUL,
                         spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Sans-B", fontSize=10.8, leading=14, textColor=AZUL,
                         spaceBefore=8, spaceAfter=4),
    "title": ParagraphStyle("title", fontName="Sans-B", fontSize=19, leading=24,
                            alignment=TA_CENTER, textColor=AZUL),
    "sub": ParagraphStyle("sub", fontName="Serif-I", fontSize=11, leading=15, alignment=TA_CENTER),
    "cap": ParagraphStyle("cap", fontName="Serif-I", fontSize=8.4, leading=11, alignment=TA_CENTER,
                          textColor=colors.HexColor("#444444"), spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Serif", fontSize=8, leading=10.2),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8, leading=10.2,
                            textColor=colors.white),
    "quote": ParagraphStyle("quote", fontName="Serif-I", fontSize=8.4, leading=11.2,
                            leftIndent=12, spaceAfter=3, textColor=colors.HexColor("#333333")),
    "bib": ParagraphStyle("bib", fontName="Serif", fontSize=8.8, leading=12, leftIndent=14,
                          firstLineIndent=-14, spaceAfter=4),
}
P = lambda t, s="body": Paragraph(t, S[s])  # noqa: E731


def tabla(datos, anchos, cab=True):
    filas = [[Paragraph(str(c), S["cellb" if (cab and i == 0) else "cell"]) for c in fila]
             for i, fila in enumerate(datos)]
    t = Table(filas, colWidths=anchos, repeatRows=1 if cab else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d1db")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if cab:
        st.append(("BACKGROUND", (0, 0), (-1, 0), AZUL))
        for i in range(2, len(filas), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f2f5f8")))
    t.setStyle(TableStyle(st))
    return t


def figura(nombre, ancho_cm, pie):
    from PIL import Image as PI
    w, h = PI.open(FIG / nombre).size
    return KeepTogether([Image(str(FIG / nombre), width=ancho_cm * cm,
                               height=ancho_cm * cm * h / w), P(pie, "cap")])


def pie_pagina(c, d):
    c.saveState()
    c.setFont("Sans", 7.5)
    c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, 'Tópicos en "Noches Blancas" con BERTopic')
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{d.page}")
    c.restoreState()


def recortar(t, n=290):
    t = t.replace("»»", "").lstrip("»").strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " […]"


def main():
    seg = pd.read_csv(RES / "estadisticas_segmentacion.csv", index_col=0)["valor"]
    comp = pd.read_csv(RES / "comparacion_configuraciones.csv")
    info = pd.read_csv(RES / "tabla_topicos.csv")
    rep = json.load(open(RES / "segmentos_representativos.json"))
    out = json.load(open(RES / "outliers.json"))
    tam = pd.read_csv(RES / "tamanos_antes_despues.csv", index_col=0)
    ocap = pd.read_csv(RES / "outliers_por_capitulo.csv", index_col=0)["outliers_%"]
    segs = pd.read_csv(RES / "segmentos_con_topicos.csv")
    caps = segs.groupby("capitulo", sort=False).size()
    corpus = (ROOT / "data" / "noches_blancas_es.txt").read_text(encoding="utf-8")
    n_pal = len([w for b in corpus.split("\n\n") if not b.startswith("##") for w in b.split()])

    st = []
    st += [Spacer(1, 1.2 * cm), P("Tópicos en <i>Noches Blancas</i> de Dostoyevski", "title"),
           Spacer(1, 4), P("Modelado de tópicos con BERTopic sobre un corpus literario en español", "sub"),
           Spacer(1, 6), P("Braulio Fernando Antero Díaz · Módulo 6 - Herramientas para el proceso de lenguaje natural · Noviembre de 2026", "sub"),
           Spacer(1, 14)]

    # 1. Introducción
    st += [P("1. Introducción y descripción del corpus", "h1"),
           P("El objetivo de este trabajo es segmentar un texto literario según su tipo y descubrir, "
             "con BERTopic, los tópicos que lo componen. Se eligió <i>Noches Blancas</i> (Fiódor "
             "Dostoyevski, 1848), una novela corta narrada en primera persona por un joven soñador "
             "solitario de San Petersburgo que, a lo largo de cuatro noches y una mañana, conoce a "
             "Nástenka, escucha su historia y se enamora de ella mientras ella espera el regreso de "
             "otro hombre. Es un buen caso de estudio porque alterna registros muy distintos: "
             "descripción urbana, monólogos introspectivos, una historia enmarcada (la de Nástenka) "
             "y mucho diálogo."),
           tabla([["Característica", "Valor"],
                  ["Obra", "<i>Noches Blancas</i> (Белые ночи), F. M. Dostoyevski, 1848"],
                  ["Fuente", "Traducción al español realizada con IA (Claude, Anthropic) a partir de la "
                             "traducción inglesa de Constance Garnett (1918, dominio público), párrafo "
                             "por párrafo, conservando párrafos y capítulos del original."],
                  ["Idioma", "Español"],
                  ["Tamaño", f"{n_pal:,} palabras · {int(seg['parrafos_originales'])} párrafos · "
                             "6 capítulos (Primera, Segunda, Tercera y Cuarta Noche, Historia de "
                             "Nástenka y La Mañana)".replace(",", " ")]],
                 [3.2 * cm, 13.2 * cm]),
           Spacer(1, 6),
           P("<b>Nota sobre la fuente.</b> No fue posible acceder a una edición española de dominio "
             "público desde el entorno de trabajo, por lo que el corpus es una traducción de una "
             "traducción (ruso → inglés → español). Para el modelado de tópicos esto es aceptable "
             "—el contenido semántico y la estructura de párrafos se conservan—, pero el léxico "
             "concreto no coincide con el de ninguna edición publicada en español. Esto se retoma en "
             "la discusión.")]

    # 2. Segmentación
    st += [P("2. Segmentación: criterio y estadísticas", "h1"),
           P("Siguiendo la tabla de unidades de análisis, para un libro/novela <b>un documento = un "
             "párrafo</b>. El párrafo es la unidad natural del texto: en la narración agrupa una idea "
             "o escena, y en el diálogo corresponde a una intervención de un personaje. Se aplicaron "
             "además tres reglas:"),
           P("• <b>Límite del modelo de embeddings.</b> El modelo elegido procesa como máximo 128 tokens; "
             "lo que excede se trunca. El monólogo del soñador de la Segunda Noche ocupa un solo "
             "párrafo de más de 1 900 palabras y otros pasan de 600, así que los párrafos de más de 128 tokens se dividieron en bloques "
             "consecutivos de hasta 3 oraciones sin rebasar el límite (criterio de la tabla para "
             "textos largos: bloque de 3–5 oraciones). Las oraciones que por sí solas lo excedían se "
             "partieron por punto y coma, dos puntos o raya."),
           P(f"• <b>Limpieza.</b> Se eliminaron los segmentos vacíos o de menos de 5 palabras "
             f'({int(seg["eliminados_<5_palabras"])} réplicas del tipo "—¡Sí, sí!" o "—De acuerdo.") '
             f"y los duplicados exactos ({int(seg['eliminados_duplicados'])}). Los títulos de capítulo "
             "no se tratan como documentos, pero se guardan como metadato."),
           P(f"• <b>Tamaño.</b> El resultado son <b>{int(seg['segmentos_finales'])} segmentos</b>, por "
             "encima del mínimo recomendado de 500 para que HDBSCAN tenga suficientes documentos."),
           tabla([["Estadística", "Valor", "Estadística", "Valor"],
                  ["Párrafos originales", int(seg["parrafos_originales"]),
                   "Longitud media", f"{seg['palabras_media']} palabras"],
                  ["Párrafos divididos (> 128 tokens)", int(seg["parrafos_partidos"]),
                   "Mediana", f"{seg['palabras_mediana']:.0f} palabras"],
                  ["Segmentos antes de filtrar", int(seg["segmentos_brutos"]),
                   "Mínimo / máximo", f"{int(seg['palabras_min'])} / {int(seg['palabras_max'])} palabras"],
                  ["Eliminados (< 5 palabras / duplicados)",
                   f"{int(seg['eliminados_<5_palabras'])} / {int(seg['eliminados_duplicados'])}",
                   "Tokens medios por segmento", seg["tokens_media"]],
                  ["<b>Segmentos finales</b>", f"<b>{int(seg['segmentos_finales'])}</b>",
                   "Segmentos > 128 tokens", f"{int(seg['segmentos_>128_tokens'])} (truncado mínimo)"]],
                 [5.2 * cm, 2.4 * cm, 4.6 * cm, 4.2 * cm]),
           Spacer(1, 6),
           P("Por capítulo: " + ", ".join(f"{k.replace(' De ', ' de ')} ({v})" for k, v in caps.items())
             + ". La distribución de longitudes (figura 1) es asimétrica a la derecha: muchos "
             "segmentos cortos de diálogo (20–40 palabras) y una cola de bloques narrativos que "
             "llegan al tope de 100 palabras impuesto por el límite de tokens."),
           figura("histograma_longitudes.png", 11.5, "Figura 1. Histograma de longitudes de los 505 segmentos.")]

    # 3. Metodología
    st += [P("3. Metodología", "h1"), P("3.1 Modelo de embeddings", "h2"),
           P("Se usó <b>paraphrase-multilingual-MiniLM-L12-v2</b> (sentence-transformers). Las razones: "
             "(1) es multilingüe (más de 50 idiomas, incluido el español) y fue entrenado por "
             "destilación de conocimiento sobre datos paralelos para que paráfrasis en distintas lenguas "
             "queden cerca en el espacio vectorial (Reimers y Gurevych, 2020); (2) está entrenado "
             "específicamente para similitud semántica de oraciones y párrafos, que es justo lo que "
             "BERTopic necesita para agrupar; (3) es ligero (384 dimensiones, 12 capas) y es el modelo "
             "multilingüe recomendado por defecto en la documentación de BERTopic. Su limitación "
             "principal es la ventana de 128 tokens, que motivó la regla de segmentación anterior. Una "
             "prueba de control dio una similitud coseno de 0,83 entre una oración en español y su "
             "traducción al inglés, y valores cercanos a 0 entre oraciones sin relación. Se descartó un "
             "modelo solo en español (p. ej. BETO sin ajuste) porque sus embeddings sin entrenamiento "
             "de similitud agrupan peor. Los embeddings se calcularon una sola vez y se reutilizaron "
             "en todas las configuraciones, para que las diferencias se deban solo a los parámetros."),
           P("3.2 Componentes de BERTopic", "h2"),
           tabla([["Componente", "Configuración", "Justificación"],
                  ["UMAP", "n_neighbors=15, n_components=5, min_dist=0, métrica coseno, "
                           "<b>random_state=42</b>",
                   "Valores por defecto de BERTopic; la semilla fija hace el resultado reproducible."],
                  ["HDBSCAN", "min_cluster_size y min_samples variables (ver 3.3), EOM",
                   "Agrupa por densidad y marca como outlier (−1) lo que no encaja en ningún grupo."],
                  ["CountVectorizer", "Stopwords en español (lista <i>stop-words</i>, 652 términos) + "
                                      "47 interjecciones y muletillas del diálogo (oh, ay, mire, "
                                      "dijo, usted…) y el vocativo \"Nástenka\"; min_df=2; solo palabras "
                                      "de 3+ letras",
                   "\"Nástenka\" aparece en el 21 % de los segmentos y encabezaba casi todos los tópicos "
                   "sin distinguirlos."],
                  ["Representación", "c-TF-IDF + Maximal Marginal Relevance (diversity=0,3)",
                   "MMR reduce sinónimos repetidos en las palabras clave de cada tópico."]],
                 [2.6 * cm, 6.6 * cm, 7.2 * cm]),
           P("3.3 Configuraciones comparadas", "h2"),
           P("Se probaron tres configuraciones que varían el tamaño mínimo de clúster, el rango de "
             "n-gramas y el número de tópicos. Para compararlas se midió el número de tópicos, el "
             "porcentaje de outliers, la coherencia NPMI media (co-ocurrencia de las 10 palabras top de "
             "cada tópico dentro de los segmentos) y la diversidad de tópicos (proporción de palabras "
             "únicas entre los top-10 de todos los tópicos).")]
    datos = [["Config.", "min_cluster _size", "min_ samples", "ngram_ range", "nr_ topics", "Tópicos",
              "Outliers", "NPMI", "Divers."]]
    ms = {"C1_gruesa": "auto (=15)", "C2_fina": "3", "C3_reducida": "3"}
    for _, r in comp.iterrows():
        nombre = r.configuracion.replace("_", " ")
        if r.configuracion == "C2_fina":
            nombre = "<b>C2 fina (final)</b>"
        datos.append([nombre, r.min_cluster_size, ms[r.configuracion], r.ngram_range, r.nr_topics,
                      r.n_topicos, f"{r['outliers_%']} %", r.NPMI_medio, r.diversidad])
    st += [tabla(datos, [2.9 * cm, 1.9 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm, 1.4 * cm, 1.6 * cm,
                         1.4 * cm, 1.7 * cm]),
           Spacer(1, 6),
           P("<b>C1 (gruesa)</b> produce 7 tópicos amplios con 27,7 % de outliers: casi todo el diálogo "
             "de la segunda mitad cae en un único tópico de 124 segmentos (\"mañana, carta, quizá\"), que "
             "mezcla la espera de la carta, las citas y las declaraciones. <b>C2 (fina)</b> baja "
             "min_cluster_size a 10 y fija min_samples=3, con lo que HDBSCAN es menos conservador: "
             "salen 12 tópicos, los outliers bajan a 20,6 % y aparecen temas que en C1 estaban "
             "ocultos (la carta, las citas, la felicidad frente al futuro, la declaración de amor). "
             "<b>C3</b> parte de C2 y fusiona tópicos hasta quedar en 7: vuelve a crear un tópico "
             "\"cajón de sastre\" de 140 segmentos (\"dios, corazón, amor, feliz\")."),
           P("La NPMI es negativa en las tres, algo esperable con segmentos cortos (con pocas palabras, "
             "dos términos rara vez aparecen en el mismo segmento) y favorece a los tópicos grandes, por "
             "eso C1 y C3 puntúan algo mejor. No se usó como único criterio. La diversidad es alta "
             "(≈0,9) en todas, es decir, los tópicos no repiten palabras clave. Se eligió <b>C2</b> "
             "porque ofrece la mejor interpretabilidad al leer los segmentos, el menor porcentaje de "
             "outliers y tópicos que corresponden a momentos reconocibles de la novela; el costo es "
             "una coherencia NPMI menor, compensada por la validación cualitativa de la sección 4.")]

    # 4. Resultados
    st += [PageBreak(), P("4. Resultados", "h1"), P("4.1 Tabla de tópicos (modelo C2)", "h2"),
           P("La tabla resume <font face='Sans'>get_topic_info()</font> con un nombre propio asignado a cada "
             "tópico tras leer sus palabras clave y sus segmentos representativos. \"n\" es el número de "
             "segmentos antes de tratar los outliers; \"n final\", después (sección 4.3).")]
    datos = [["#", "Nombre propio", "Palabras clave (c-TF-IDF + MMR)", "n", "n final"]]
    for _, r in info.iterrows():
        t = int(r.Topic)
        datos.append([t, r.Etiqueta if t != -1 else "<i>Outliers</i>", r["Palabras clave"],
                      r.Count, int(tam.loc[t, "despues"])])
    st += [tabla(datos, [0.8 * cm, 5.2 * cm, 7.8 * cm, 1.2 * cm, 1.4 * cm]),
           Spacer(1, 6),
           P("4.2 Segmentos representativos", "h2"),
           P("Para cada tópico se muestran dos de los segmentos que BERTopic considera más "
             "representativos (los más cercanos al c-TF-IDF del tópico), recortados por espacio.")]
    for _, r in info.iterrows():
        t = int(r.Topic)
        if t == -1:
            continue
        docs = sorted(rep[str(t)], key=len, reverse=True)[:2]
        bloque = [P(f"<b>Tópico {t} · {r.Etiqueta}</b>", "body")]
        bloque += [P("«" + recortar(d) + "»", "quote") for d in docs]
        st.append(KeepTogether(bloque))

    st += [P("4.3 Outliers (tópico −1)", "h2"),
           P(f"El modelo C2 dejó <b>{out['outliers_iniciales']} segmentos ({out['outliers_iniciales_%']} %)"
             "</b> como outliers. Al leerlos se ven dos tipos: réplicas breves de diálogo que dependen "
             "del contexto (\"—Pero yo no lo conocía; pensé que usted también…\", \"—¡Un poco! Por ejemplo, "
             "¿por qué tiembla usted?\") y pasajes descriptivos aislados (el anciano de la Fontanka, la "
             "muchacha que se asoma a la ventana). Los outliers se concentran en la Segunda Noche "
             f"({ocap.iloc[1]} %) y en la Historia de Nástenka ({ocap.iloc[2]} %), los capítulos con "
             "más variedad de temas dentro del mismo monólogo."),
           P("<b>Qué se hizo.</b> Se reasignaron con <font face='Sans'>reduce_outliers(strategy=\"embeddings\")"
             "</font>, pero con un umbral exigente y basado en los datos: un outlier pasa a un tópico solo "
             "si su similitud coseno con el centroide de ese tópico es al menos igual al percentil 25 de "
             f"la similitud de los miembros ya asignados (umbral = {out['umbral_similitud']}). Así solo se "
             f"recuperan segmentos tan típicos como un miembro normal. Se reasignaron "
             f"<b>{out['reasignados']}</b> segmentos y quedaron <b>{out['outliers_finales']} "
             f"({out['outliers_finales_%']} %)</b> como ruido, que se conservan como −1 en lugar de forzarlos. "
             "Con un umbral laxo (0,35) se reasignaba el 99 % de los outliers, lo que habría "
             "contaminado los tópicos con réplicas sin contenido. Las palabras clave de los tópicos se "
             "calcularon antes de la reasignación, para no alterar su representación.")]

    st += [P("4.4 Visualizaciones", "h2"),
           figura("barchart_topicos.png", 16.4, "Figura 2. Palabras con mayor peso c-TF-IDF (re-ordenadas "
                  "con MMR) en cada tópico."),
           P("<b>Interpretación de la figura 2.</b> Los tópicos con vocabulario propio son los más claros: "
             "el 3 (abuela, inquilino, alfiler, viejos) y el 10 (carta, mañana, venir) se reconocen sin "
             "leer los textos, y el 4 (Petersburgo, ciudad, verano) y el 5 (vida, soñador, sueños, "
             "mundo) recogen los dos grandes temas del soñador: la ciudad que lo abandona y su vida "
             "imaginaria. En cambio, el 0 (mano, ojos, repente) y el 1 (corazón, llore, exclamé) están "
             "dominados por verbos de acción y de habla: describen <i>cómo</i> se narra (gestos, "
             "llanto) más que <i>de qué</i> se habla."),
           figura("mapa_documentos.png", 16, "Figura 3. Proyección UMAP 2D de los embeddings; cada punto "
                  "es un segmento coloreado por tópico (gris = outliers)."),
           P("<b>Interpretación de la figura 3.</b> El mapa separa bien la historia de la abuela (3), el "
             "monólogo del soñador (5) y Petersburgo (4), que forman islas compactas en los bordes. En "
             "el centro, los tópicos de diálogo (2, 6, 9, 11) se solapan y los outliers grises se "
             "acumulan entre ellos: es la zona de conversación entre los dos personajes, donde el "
             "contenido depende del intercambio y no de un tema estable. Abajo a la derecha, la carta "
             "(10), las citas (7) y los planes (8) quedan juntos, coherentes con la trama de la "
             "espera."),
           figura("jerarquia_topicos.png", 15, "Figura 4. Dendrograma (Ward) de los centroides de los "
                  "tópicos en el espacio de embeddings."),
           P("<b>Interpretación de la figura 4.</b> La jerarquía separa dos grandes ramas. Una agrupa "
             "el <b>drama de la relación</b>: declaraciones de amor (6), llanto y confesión (1), "
             "fórmulas de diálogo (11), planes (8) y, en un subgrupo, la mecánica de la espera "
             "(citas 7 y carta 10). La otra agrupa la <b>narración y la introspección</b>: los gestos "
             "del encuentro (0) junto a la timidez del soñador (2), la felicidad y el miedo al futuro "
             "(9) junto a Petersburgo (4), la historia de la abuela (3) y, al final y bastante "
             "separado, el monólogo del soñador (5), el tópico más singular del libro. Si se quisieran "
             "menos tópicos, los primeros candidatos a fusionarse serían 1+11 y 0+2."),
           figura("topicos_por_capitulo.png", 15, "Figura 5. Porcentaje de segmentos de cada capítulo "
                  "asignados a cada tópico (modelo C2, sin contar outliers)."),
           P("<b>Interpretación de la figura 5.</b> Aunque BERTopic no recibió información de capítulos, "
             "los tópicos reproducen el arco narrativo: la Primera Noche está dominada por Petersburgo y "
             "la soledad (23 %) y la timidez del soñador (18 %); la Segunda Noche, por el monólogo del "
             "soñador (26 %), que no vuelve a aparecer; la Historia de Nástenka, por la abuela y el "
             "inquilino (33 %); la Cuarta Noche, por el llanto y la confesión (22 %); y La Mañana, por "
             "la carta de Nástenka pidiendo perdón (33 % en el tópico 6). Es la mejor evidencia de que "
             "los tópicos tienen sentido para este corpus.")]

    # 5. Discusión
    st += [P("5. Discusión", "h1"), P("5.1 Tópicos que tienen sentido", "h2"),
           P("Los tópicos 3 (abuela e inquilino), 4 (Petersburgo), 5 (el soñador), 6 (amor y perdón) y "
             "10 (la carta) son claramente temáticos: tienen vocabulario distintivo, sus segmentos "
             "representativos tratan un mismo asunto y se concentran en el capítulo donde ese asunto "
             "ocurre. El 7 (citas y \"hasta mañana\") y el 9 (felicidad presente frente al futuro) son más "
             "pequeños pero coherentes: el primero reúne la logística de los encuentros nocturnos y el "
             "segundo, la reflexión del soñador sobre lo efímero de su felicidad, un tema central de "
             "la obra. El 2 resulta interesante: empezó pareciendo \"la timidez ante las mujeres\" "
             "(Primera Noche), pero incluye también los pasajes de la Tercera Noche en los que el "
             "soñador interpreta mal la ternura de Nástenka; por eso se le dio un nombre más amplio."),
           P("5.2 Tópicos que no tienen tanto sentido y por qué", "h2"),
           P("• <b>Tópico 0 (gestos narrados) y tópico 1 (llanto y confesión)</b> son los más grandes y "
             "agrupan por <i>registro</i> y no por tema: narración de acciones físicas (tomar la mano, "
             "bajar los ojos) y estallidos emocionales en diálogo. El modelo de paráfrasis capta bien el "
             "tipo de discurso, y en una novela con dos personajes y un solo conflicto el registro "
             "varía más que el tema."),
           P("• <b>Tópico 11 (fórmulas de diálogo)</b> se define por su función conversacional (\"dígamelo\", "
             "\"basta\", \"continúe\"), no por su contenido. Es un artefacto de segmentar el diálogo por "
             "intervención: réplicas cortas casi sin palabras con contenido."),
           P("• <b>Tópico 8 (propósitos y planes)</b> es el menos coherente: mezcla los planes de "
             "Nástenka de dar lecciones, la decisión de \"comportarnos con más sensatez\" y el pasaje "
             "en que las casas de Petersburgo \"hablan\" con el narrador. Comparten un tono de "
             "intención o anuncio, pero no un tema; por eso el nombre lo marca como mixto."),
           P("• <b>Limitaciones generales.</b> (1) Un único texto breve da pocos documentos "
             "por tópico y obliga a usar tamaños de clúster pequeños; (2) los segmentos "
             "partidos de un mismo monólogo heredan su contexto y pueden caer en tópicos distintos "
             "aunque formen una sola idea; (3) el corpus es una traducción de una traducción hecha con "
             "IA, de modo que el léxico (y por tanto las palabras clave) refleja esas decisiones de "
             "traducción y no las de una edición publicada; (4) la NPMI, calculada sobre los propios "
             "segmentos cortos, es una medida ruidosa en este contexto."),
           P("6. Conclusiones", "h1"),
           P("BERTopic, con un modelo multilingüe de paráfrasis y una segmentación por párrafos ajustada "
             "al límite de 128 tokens, encontró 12 tópicos en <i>Noches Blancas</i>, de los cuales unos "
             "ocho son claramente temáticos y siguen el arco de la novela: la ciudad que se vacía, la "
             "vida imaginaria del soñador, la historia de la abuela y el inquilino, la espera de la "
             "carta y la despedida final. Los demás reflejan el registro (narración de gestos, llanto, "
             "fórmulas de diálogo) más que el contenido, algo esperable en un corpus literario con poco "
             "vocabulario acerca del contexto y muchos diálogos implícitos. Comparar configuraciones mostró que un "
             "min_cluster_size menor usando un min_samples=3 reduce los outliers y separa temas que en una "
             "configuración gruesa quedaban mezclados, y que forzar pocos tópicos (nr_topics) vuelve a "
             "crear tópicos \"cajón de sastre\". Para trabajos futuros sería útil comparar con una edición "
             "publicada en español, probar modelos de embeddings con ventanas más largas (p. ej. "
             "multilingual-e5) que permitan párrafos completos, y aplicar el mismo método a varias "
             "obras de Dostoyevski para tener un corpus mayor.")]

    # 7. Bibliografía
    st += [P("7. Bibliografía", "h1")]
    for b in [
        "Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. "
        "<i>Proceedings of GSCL</i>, 31–40.",
        "Campello, R. J. G. B., Moulavi, D. y Sander, J. (2013). Density-based clustering based on "
        "hierarchical density estimates. <i>PAKDD 2013</i>, LNCS 7819, 160–172.",
        "Carbonell, J. y Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering "
        "documents and producing summaries. <i>SIGIR '98</i>, 335–336.",
        "Dostoyevski, F. M. (1848/1918). <i>White Nights</i> (trad. C. Garnett). En <i>White Nights and "
        "Other Stories</i>. Londres: Heinemann. Dominio público.",
        "Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. "
        "<i>arXiv:2203.05794</i>.",
        "McInnes, L., Healy, J. y Melville, J. (2018). UMAP: Uniform Manifold Approximation and "
        "Projection for dimension reduction. <i>arXiv:1802.03426</i>.",
        "Reimers, N. y Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese "
        "BERT-networks. <i>EMNLP-IJCNLP 2019</i>, 3982–3992.",
        "Reimers, N. y Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using "
        "knowledge distillation. <i>EMNLP 2020</i>, 4512–4525.",
        "Wang, W., Wei, F., Dong, L., Bao, H., Yang, N. y Zhou, M. (2020). MiniLM: Deep self-attention "
        "distillation for task-agnostic compression of pre-trained transformers. <i>NeurIPS 2020</i>.",
    ]:
        st.append(P(b, "bib"))

    st += [P("8. Declaración de uso de IA", "h1"),
           P("Se utilizó Claude (Anthropic) como asistente en este trabajo para: traducir al español "
             "la traducción inglesa de Constance Garnett, párrafo por párrafo, ya que no se contaba con "
             "una edición española accesible; y escribir gran parte del código de segmentación, modelado y "
             "visualización. La elección de parámetros, los nombres de los tópicos y la interpretación se basaron en la lectura de "
             "los resultados reales (tablas, segmentos representativos y figuras), y el autor revisó "
             "y es responsable del contenido final"),
           P("9. Código", "h1"),
           P("Repositorio / notebook: <b>https://github.com/BrawlStereo/BERTopic-Test.git</b>"),
           P("El repositorio contiene el corpus, los scripts <font face='Sans'>segmentar.py</font>, "
             "<font face='Sans'>modelar.py</font> y <font face='Sans'>final.py</font>, el notebook "
             "<font face='Sans'>noches_blancas_bertopic.ipynb</font> (listo para Colab), los embeddings "
             "precalculados para reproducir exactamente los resultados (<font face='Sans'>random_state=42"
             "</font> en UMAP) y las versiones interactivas de las visualizaciones de BERTopic (HTML).")]

    doc = SimpleDocTemplate(str(ROOT / "reporte_bertopic_noches_blancas.pdf"), pagesize=A4,
                            leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title="Tópicos en Noches Blancas con BERTopic",
                            author="Braulio Fernando Antero Díaz")
    doc.build(st, onFirstPage=pie_pagina, onLaterPages=pie_pagina)


if __name__ == "__main__":
    main()
