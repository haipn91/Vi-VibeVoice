"""
Inference CAPU (xlm-roberta-capu) qua ONNX Runtime — khong nap pytorch_model.bin vao RAM.

Can:
  pip install torch          # van can cho tien xu ly (tensor, GecBERTModel.preprocess)
  pip install onnxruntime
  + python scripts/export_capu_onnx.py  -> capu-seq2labels.onnx (FP32)
  + python scripts/quantize_capu_onnx.py -> capu-seq2labels.int8.onnx (nhanh hon, nho hon)

Khi GIPFORMER_CAPU_ONNX=1 va tim thay .onnx, vi_capu_punctuate se uu tien backend nay.
Mac dinh uu tien capu-seq2labels.int8.onnx neu co; tat: GIPFORMER_CAPU_PREFER_INT8=0.

Toc do (CPU mac dinh rat cham neu khong tinh chinh):
  ORT_CPU_NUM_THREADS — so luong thread intra-op ONNX Runtime (mac dinh: min(8, CPU)).
  GIPFORMER_CAPU_ITERATIONS — so vong GEC (mac dinh 1; HF mac dinh 3 vong, cham ~3x).
  GIPFORMER_CAPU_CHUNK_SIZE / GIPFORMER_CAPU_OVERLAP — doan tu (words); chunk lon => it lan ORT hon.
  GIPFORMER_CAPU_MAX_LEN — gioi han do dai (words) moi lan encode (mac dinh 128).
  GIPFORMER_CAPU_ORT_CPU_ONLY=1 — chi CPUExecutionProvider (tat CUDA ORT).
  Tren GPU: cai onnxruntime-gpu, torch.cuda + device cuda => ORT uu tien CUDAExecutionProvider.
  GIPFORMER_CAPU_SNAPSHOT_DIR — thu muc snapshot day du (co verb-form-vocab.txt).
  GIPFORMER_CAPU_SNAPSHOT_LOCAL_DIR — tai snapshot phang vao day (tranh loi symlink cache HF + utils.py).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

# Windows: tranh WinError 1314 khi snapshot_download tao symlink (ONNX can vocab HF).
if sys.platform == "win32":
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_onnx_gec: Any = None
_onnx_err: Optional[str] = None


def onnx_capu_enabled() -> bool:
    return os.environ.get("GIPFORMER_CAPU_ONNX", "").strip().lower() in ("1", "true", "yes", "on")


def _default_capu_flat_snapshot_dir() -> Path:
    """Thu muc snapshot phang (khong symlink blobs) — tranh loi verb-form-vocab trong utils.py."""
    h = os.environ.get("HF_HOME", "").strip()
    if h:
        base = Path(h)
    else:
        base = Path.home() / ".cache" / "huggingface"
    return base / "vivibevoice_capu_flat_snapshot"


def _snapshot_download_capu_flat(repo: str, cache: str | None, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    kw: dict = {"repo_id": repo, "cache_dir": cache, "local_dir": str(local_dir.resolve())}
    return Path(snapshot_download(**kw))


def _resolve_capu_onnx_path() -> Path:
    """FP32 mac dinh capu-seq2labels.onnx; uu tien .int8.onnx neu ton tai (quantize_capu_onnx.py)."""
    env = os.environ.get("GIPFORMER_CAPU_ONNX_PATH", "").strip()
    if env:
        return Path(env)
    root = Path(__file__).resolve().parent / "onnx"
    fp32 = root / "capu-seq2labels.onnx"
    int8 = root / "capu-seq2labels.int8.onnx"
    prefer = os.environ.get("GIPFORMER_CAPU_PREFER_INT8", "1").strip().lower()
    prefer_int8 = prefer not in ("0", "false", "no", "off")
    if prefer_int8 and int8.is_file():
        return int8
    return fp32


def _onnx_file_looks_int8(onnx_p: Path) -> bool:
    n = onnx_p.name.lower()
    return ".int8." in n or n.endswith("int8.onnx")


def _capu_fp32_onnx_sibling(onnx_p: Path) -> Optional[Path]:
    """Thu capu-seq2labels.onnx cung thu muc khi ban int8 hong / khong doc duoc."""
    alt = onnx_p.parent / "capu-seq2labels.onnx"
    return alt if alt.is_file() else None


def _onnx_load_failure_is_corrupt_model(e: BaseException) -> bool:
    msg = str(e)
    return "INVALID_PROTOBUF" in msg or "Protobuf parsing" in msg or "InvalidProtobuf" in type(e).__name__


def _parse_positive_int(key: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return max(lo, min(hi, default))
    try:
        v = int(raw)
    except ValueError:
        return max(lo, min(hi, default))
    return max(lo, min(hi, v))


def _gec_chunk_params() -> tuple[int, int, int, int]:
    """iterations, chunk_size, overlap_size, max_len — dieu chinh toc do vs chat luong."""
    iterations = _parse_positive_int("GIPFORMER_CAPU_ITERATIONS", 1, 1, 5)
    chunk_size = _parse_positive_int("GIPFORMER_CAPU_CHUNK_SIZE", 96, 32, 256)
    overlap_size = _parse_positive_int("GIPFORMER_CAPU_OVERLAP", 16, 0, 64)
    max_len = _parse_positive_int("GIPFORMER_CAPU_MAX_LEN", 128, 32, 256)
    if overlap_size * 2 > chunk_size:
        overlap_size = max(0, chunk_size // 2 - 1)
    return iterations, chunk_size, overlap_size, max_len


def _ort_execution_providers(device: torch.device) -> list[str]:
    """ORT CPU hoac CUDA (can onnxruntime-gpu + driver)."""
    import onnxruntime as ort

    force_cpu = os.environ.get("GIPFORMER_CAPU_ORT_CPU_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if force_cpu or device.type != "cuda":
        return ["CPUExecutionProvider"]
    try:
        avail = ort.get_available_providers()
    except Exception:
        avail = []
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.warning(
        "[capu-onnx] torch.device=cuda nhung ONNX Runtime khong co CUDAExecutionProvider "
        "(cai onnxruntime-gpu dong bo voi CUDA? Hoac dat GIPFORMER_CAPU_ORT_CPU_ONLY=1)."
    )
    return ["CPUExecutionProvider"]


class _OrtSeq2LabelsCore(nn.Module):
    """Thay the Seq2LabelsModel: forward ONNX, tra dict nhu output PyTorch."""

    def __init__(self, onnx_path: Path, device: torch.device) -> None:
        super().__init__()
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        raw_threads = os.environ.get("ORT_CPU_NUM_THREADS", "").strip()
        n_threads = int(raw_threads) if raw_threads.isdigit() else 0
        if n_threads <= 0:
            n_threads = min(8, os.cpu_count() or 4)
        so.intra_op_num_threads = n_threads
        so.inter_op_num_threads = 1
        providers = _ort_execution_providers(device)
        self._sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        self._device = device
        logger.info("[capu-onnx] ORT providers=%s, torch.device=%s", providers, device)

        out_meta = self._sess.get_outputs()
        names = [o.name for o in out_meta]
        if "logits" in names and "detect_logits" in names:
            self._ort_out = ("logits", "detect_logits")
        elif len(names) >= 2:
            self._ort_out = tuple(names[:2])
            logger.warning(
                "[capu-onnx] ONNX thieu ten logits/detect_logits chuan, dung thu tu dau ra: %s",
                names,
            )
        else:
            raise RuntimeError(f"[capu-onnx] Can 2 dau ra ONNX, co: {names}")

        nlow = onnx_path.name.lower()
        if ".int8." in nlow or nlow.endswith("int8.onnx"):
            logger.info(
                "[capu-onnx] Dang nap INT8 — dynamic quant co the lam lech dau cau; "
                "neu sai nhieu, dat GIPFORMER_CAPU_PREFER_INT8=0 va dung capu-seq2labels.onnx (FP32)."
            )

    def forward(self, input_ids=None, attention_mask=None, input_offsets=None, **kw):
        if input_ids is None or attention_mask is None or input_offsets is None:
            raise ValueError("ORT core needs input_ids, attention_mask, input_offsets")
        feeds = {
            "input_ids": input_ids.detach().cpu().numpy(),
            "attention_mask": attention_mask.detach().cpu().numpy(),
            "input_offsets": input_offsets.detach().cpu().numpy(),
        }
        lo, dd = self._sess.run(list(self._ort_out), feeds)
        logits = torch.from_numpy(lo).to(self._device)
        detect_logits = torch.from_numpy(dd).to(self._device)
        # Giong modeling_seq2labels Seq2LabelsOutput (dung cho _convert / postprocess)
        max_error_probability = torch.ones(logits.size(0), device=self._device)
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "detect_logits": detect_logits,
            "max_error_probability": max_error_probability,
        }
        return out


def _build_gec_with_onnx_core(snap: Path, onnx_path: Path) -> Any:
    root = str(snap.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    import modeling_seq2labels as msl  # type: ignore

    from gec_model import GecBERTModel  # type: ignore
    from configuration_seq2labels import Seq2LabelsConfig  # type: ignore

    real_from_pretrained = msl.Seq2LabelsModel.from_pretrained

    class _ConfigShell(nn.Module):
        def __init__(self, cfg: Any) -> None:
            super().__init__()
            self.config = cfg

    def _fake_from_pretrained(cls: Any, pretrained_model_name_or_path: str, *a: Any, **kw: Any):
        cfg = Seq2LabelsConfig.from_pretrained(str(pretrained_model_name_or_path))
        return _ConfigShell(cfg)

    msl.Seq2LabelsModel.from_pretrained = classmethod(_fake_from_pretrained)  # type: ignore[method-assign]
    try:
        vocab = snap / "vocabulary"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        iterations, chunk_size, overlap_size, max_len = _gec_chunk_params()
        gec = GecBERTModel(
            vocab_path=str(vocab),
            model_paths=str(snap),
            split_chunk=True,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            max_len=max_len,
            min_len=2,
            log=False,
            iterations=iterations,
            device=device,
        )
    finally:
        msl.Seq2LabelsModel.from_pretrained = real_from_pretrained  # type: ignore[assignment]

    gec.models[0] = _OrtSeq2LabelsCore(onnx_path, device).to(device)
    gec.models[0].eval()
    return gec


def load_onnx_capu() -> None:
    global _onnx_gec, _onnx_err
    with _lock:
        if _onnx_gec is not None or _onnx_err is not None:
            return
        onnx_p = _resolve_capu_onnx_path()
        if not onnx_p.is_file():
            _onnx_err = f"Thieu ONNX: {onnx_p} (chay scripts/export_capu_onnx.py)"
            logger.warning("[capu-onnx] %s", _onnx_err)
            return
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            _onnx_err = "Thieu onnxruntime. pip install onnxruntime"
            logger.warning("[capu-onnx] %s", _onnx_err)
            return
        try:
            if sys.platform == "win32":
                os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
            snap_dir = os.environ.get("GIPFORMER_CAPU_SNAPSHOT_DIR", "").strip()
            if snap_dir:
                snap = Path(snap_dir)
                if not (snap / "gec_model.py").is_file() or not (snap / "vocabulary").is_dir():
                    raise RuntimeError(
                        f"GIPFORMER_CAPU_SNAPSHOT_DIR khong hop le: {snap} (can gec_model.py + vocabulary/)"
                    )
                if not (snap / "verb-form-vocab.txt").is_file():
                    raise RuntimeError(
                        f"{snap} thieu verb-form-vocab.txt — can ban day du repo HF (khong phai cache symlink loi). "
                        f"Hoac de trong thu muc va dat GIPFORMER_CAPU_SNAPSHOT_LOCAL_DIR de tai phang."
                    )
                repo_label = str(snap)
            else:
                repo = os.environ.get("GIPFORMER_CAPU_REPO", "dragonSwing/xlm-roberta-capu").strip()
                cache = os.environ.get("GIPFORMER_CAPU_CACHE", "").strip() or None
                local_flat = os.environ.get("GIPFORMER_CAPU_SNAPSHOT_LOCAL_DIR", "").strip()
                if not local_flat:
                    local_flat = str(_default_capu_flat_snapshot_dir())
                snap = _snapshot_download_capu_flat(repo, cache, Path(local_flat))
                repo_label = f"{repo} (flat_dir={snap})"

            try:
                _onnx_gec = _build_gec_with_onnx_core(snap, onnx_p)
            except Exception as build_err:
                fp32_alt = _capu_fp32_onnx_sibling(onnx_p)
                if (
                    fp32_alt is not None
                    and fp32_alt != onnx_p
                    and _onnx_file_looks_int8(onnx_p)
                    and _onnx_load_failure_is_corrupt_model(build_err)
                ):
                    logger.warning(
                        "[capu-onnx] File INT8 khong hop le (%s), thu FP32: %s",
                        build_err,
                        fp32_alt,
                    )
                    _onnx_gec = _build_gec_with_onnx_core(snap, fp32_alt)
                    onnx_p = fp32_alt
                else:
                    raise

            it, cs, ov, ml = _gec_chunk_params()
            ort_t = os.environ.get("ORT_CPU_NUM_THREADS", "").strip() or "auto"
            logger.info(
                "[capu-onnx] San sang (ORT + tokenizer/vocab tu %s) onnx=%s | iter=%s chunk=%s overlap=%s max_len=%s ORT_threads=%s",
                repo_label,
                onnx_p.name,
                it,
                cs,
                ov,
                ml,
                ort_t,
            )
        except Exception as e:
            _onnx_err = str(e)
            logger.exception("[capu-onnx] Loi khoi tao: %s", e)


def apply_onnx_capu_text(text: str) -> str:
    if not text or not text.strip():
        return text
    global _onnx_gec, _onnx_err
    load_onnx_capu()
    if _onnx_err or _onnx_gec is None:
        return text
    try:
        out = _onnx_gec(text)
        if isinstance(out, list) and out and isinstance(out[0], str):
            return out[0].strip()
        return text
    except Exception as e:
        logger.warning("[capu-onnx] Suy luan: %s", e)
        return text


def onnx_capu_ready() -> bool:
    load_onnx_capu()
    return _onnx_gec is not None


def onnx_capu_last_error() -> Optional[str]:
    return _onnx_err


def onnx_capu_status(onnx_path: Optional[Path] = None) -> str:
    if not onnx_capu_enabled():
        return "capu_onnx=off"
    p = onnx_path or _resolve_capu_onnx_path()
    if not p.is_file():
        return f"capu_onnx=missing:{p.name}"
    if _onnx_gec is not None:
        return "capu_onnx=ready"
    if _onnx_err:
        return f"capu_onnx=error:{_onnx_err[:100]}"
    return "capu_onnx=pending"
