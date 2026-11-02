"""Modelo de embeddings: paraphrase-multilingual-MiniLM-L12-v2.

Por defecto se usa sentence-transformers (descarga el modelo de Hugging Face).
Si no está disponible (p. ej. sin acceso a Hugging Face), se usa una copia
ONNX del mismo modelo con mean pooling + normalización L2, que es exactamente
lo que hace sentence-transformers para este modelo.
"""
from pathlib import Path

import numpy as np

MODELO_HF = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "models" / "onnx"

_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(ONNX_DIR / "tokenizer.json"))
        _tok.no_truncation()
        _tok.no_padding()
    return _tok


def contar_tokens(texto: str) -> int:
    """Número de tokens (incluye <s> y </s>) según el tokenizador del modelo."""
    try:
        return len(_tokenizer().encode(texto).ids)
    except Exception:  # fallback con sentence-transformers / transformers
        from transformers import AutoTokenizer
        global _tok_hf
        try:
            _tok_hf
        except NameError:
            _tok_hf = AutoTokenizer.from_pretrained(MODELO_HF)
        return len(_tok_hf(texto)["input_ids"])


class OnnxMiniLM:
    """Embebedor ONNX compatible con la interfaz `.encode()` de sentence-transformers."""

    def __init__(self, model_dir=ONNX_DIR, max_len=128):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.sess = ort.InferenceSession(str(Path(model_dir) / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")

    def encode(self, textos, batch_size=32, show_progress_bar=False, **_):
        out = []
        for i in range(0, len(textos), batch_size):
            enc = self.tok.encode_batch(list(textos[i:i + batch_size]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            h = self.sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                     "token_type_ids": np.zeros_like(ids)})[0]
            v = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.vstack(out)


def cargar_modelo():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODELO_HF)
    except Exception:
        return OnnxMiniLM()
