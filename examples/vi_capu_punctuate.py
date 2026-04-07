"""
Dau cau + viet hoa cho ban ghi ASR bang dragonSwing/xlm-roberta-capu (token classification).

Mo hinh: https://huggingface.co/dragonSwing/xlm-roberta-capu
- Them dau: . , : ?
- Viet hoa dau cau / thuc the (xem model card).

Canh bao: PyTorch + ~1.1GB tai ve (HF Hub). Khong bat mac dinh.

Bat:
  set GIPFORMER_CAPU=1
Tuy chon:
  set GIPFORMER_CAPU_REPO=dragonSwing/xlm-roberta-capu
  set GIPFORMER_CAPU_CACHE=   # cache_dir cho huggingface_hub.snapshot_download

Cai them:
  pip install -r python/requirements-capu.txt
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

# Windows: tranh WinError 1314 khi huggingface_hub tao symlink vao cache (can Developer Mode).
if sys.platform == "win32":
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Any = None
_model_error: Optional[str] = None


def capu_requested() -> bool:
    return os.environ.get("GIPFORMER_CAPU", "").strip().lower() in ("1", "true", "yes", "on")


def _default_capu_flat_snapshot_dir() -> Path:
    h = os.environ.get("HF_HOME", "").strip()
    if h:
        base = Path(h)
    else:
        base = Path.home() / ".cache" / "huggingface"
    return base / "vivibevoice_capu_flat_snapshot"


def _snapshot_dir() -> Path:
    if sys.platform == "win32":
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    snap_dir = os.environ.get("GIPFORMER_CAPU_SNAPSHOT_DIR", "").strip()
    if snap_dir:
        p = Path(snap_dir)
        if not (p / "gec_model.py").is_file():
            raise RuntimeError(f"GIPFORMER_CAPU_SNAPSHOT_DIR khong hop le: {p}")
        if not (p / "verb-form-vocab.txt").is_file():
            raise RuntimeError(
                f"{p} thieu verb-form-vocab.txt — ban snapshot day du hoac dung GIPFORMER_CAPU_SNAPSHOT_LOCAL_DIR."
            )
        return p
    from huggingface_hub import snapshot_download

    repo = os.environ.get("GIPFORMER_CAPU_REPO", "dragonSwing/xlm-roberta-capu").strip()
    cache = os.environ.get("GIPFORMER_CAPU_CACHE", "").strip() or None
    local_flat = os.environ.get("GIPFORMER_CAPU_SNAPSHOT_LOCAL_DIR", "").strip()
    if not local_flat:
        local_flat = str(_default_capu_flat_snapshot_dir())
    lp = Path(local_flat)
    lp.mkdir(parents=True, exist_ok=True)
    kw: dict = {"repo_id": repo, "cache_dir": cache, "local_dir": str(lp.resolve())}
    return Path(snapshot_download(**kw))


def _load_model() -> None:
    global _model, _model_error
    with _lock:
        if _model is not None or _model_error is not None:
            return
        try:
            import torch  # noqa: F401
        except ImportError:
            _model_error = "Thieu torch. Cai: pip install -r python/requirements-capu.txt"
            logger.warning("[capu] %s", _model_error)
            return
        try:
            snap = _snapshot_dir()
            if not (snap / "gec_model.py").is_file():
                _model_error = f"Snapshot HF khong day du: {snap}"
                logger.error("[capu] %s", _model_error)
                return
            root = str(snap.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            vocab = snap / "vocabulary"
            if not vocab.is_dir():
                _model_error = f"Thieu thu muc vocabulary: {vocab}"
                logger.error("[capu] %s", _model_error)
                return

            from gec_model import GecBERTModel  # type: ignore

            # max_len < tokenizer; ASR doan dai — split_chunk + max_len 128
            _model = GecBERTModel(
                vocab_path=str(vocab),
                model_paths=root,
                split_chunk=True,
                chunk_size=64,
                overlap_size=16,
                max_len=128,
                min_len=2,
                log=False,
            )
            # return_dict=False tra ve tuple; can dict nhu Seq2LabelsOutput (xem modeling_seq2labels).
            # max_error_probability phai shape (batch,) — neu (batch, seq) thi _convert + postprocess_batch sai.
            for _m in _model.models:
                _orig = _m.forward

                def _fwd(*a, _o=_orig, **kw):
                    kw = dict(kw)
                    kw["return_dict"] = False
                    out = _o(*a, **kw)
                    if isinstance(out, tuple) and len(out) >= 2:
                        logits, detect_logits = out[0], out[1]
                        mep = logits.new_ones(logits.size(0))
                        return {
                            "logits": logits,
                            "detect_logits": detect_logits,
                            "max_error_probability": mep,
                        }
                    return out

                _m.forward = _fwd  # type: ignore[method-assign]

            logger.info("[capu] Da nap xlm-roberta-capu (dau cau/tach cau).")
        except Exception as e:
            _model_error = str(e)
            logger.exception("[capu] Khoi tao that bai: %s", e)


def maybe_apply_capu_text(text: str) -> str:
    if not text or not text.strip():
        return text
    # ONNX truoc (nhe hon PyTorch weights) neu GIPFORMER_CAPU_ONNX=1
    if os.environ.get("GIPFORMER_CAPU_ONNX", "").strip().lower() in ("1", "true", "yes", "on"):
        from vi_capu_onnx import apply_onnx_capu_text, onnx_capu_ready

        if onnx_capu_ready():
            try:
                return apply_onnx_capu_text(text)
            except Exception as e:
                logger.warning("[capu] ONNX loi, thu PyTorch: %s", e)
        else:
            from vi_capu_onnx import onnx_capu_last_error

            err = onnx_capu_last_error()
            if err:
                logger.warning("[capu] ONNX chua san sang (%s), thu PyTorch neu bat GIPFORMER_CAPU", err)
    if not capu_requested():
        return text
    _load_model()
    global _model, _model_error
    if _model_error or _model is None:
        return text
    try:
        out = _model(text)
        if isinstance(out, list) and out and isinstance(out[0], str):
            return out[0].strip()
        return text
    except Exception as e:
        logger.warning("[capu] Suy luan loi: %s", e)
        return text


def capu_status_line() -> str:
    if os.environ.get("GIPFORMER_CAPU_ONNX", "").strip().lower() in ("1", "true", "yes", "on"):
        from vi_capu_onnx import onnx_capu_status

        return onnx_capu_status()
    if not capu_requested():
        return "capu=off"
    if _model is not None:
        return "capu=ready"
    if _model_error:
        return f"capu=error:{_model_error[:120]}"
    return "capu=pending"
