"""
ASR cuc bo: g-group-ai-lab/gipformer-65M-rnnt qua sherpa-onnx (giong infer_onnx.py cua gipformer).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from vi_spoken_itn import apply_vi_spoken_number_rules

REPO_ID = "g-group-ai-lab/gipformer-65M-rnnt"
SAMPLE_RATE = 16000
FEATURE_DIM = 80

ONNX_FILES = {
    "fp32": {
        "encoder": "encoder-epoch-35-avg-6.onnx",
        "decoder": "decoder-epoch-35-avg-6.onnx",
        "joiner": "joiner-epoch-35-avg-6.onnx",
    },
    "int8": {
        "encoder": "encoder-epoch-35-avg-6.int8.onnx",
        "decoder": "decoder-epoch-35-avg-6.int8.onnx",
        "joiner": "joiner-epoch-35-avg-6.int8.onnx",
    },
}

_recognizer = None
_model_lock = threading.Lock()


def _sentence_case_vi(text: str) -> str:
    """Chu thuong, roi viet hoa chu dau moi doan sau . ! ? (khong nhan dien ten rieng giua cau)."""
    text = text.strip().lower()
    if not text:
        return text
    chunks = re.split(r"(?<=[.!?])\s+", text)
    fixed = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        fixed.append(chunk[0].upper() + chunk[1:] if len(chunk) > 1 else chunk.upper())
    return " ".join(fixed)


def _normalize_recognized_text(text: str) -> str:
    """
    Tu vung BPE trong tokens.txt cua gipformer la chu HOA co dau — decoder khong phan biet ten rieng.
    GIPFORMER_SPOKEN_ITN (mac dinh bat): chuan hoa thang/nam noi bang chu -> so (vi_spoken_itn).
    GIPFORMER_OUTPUT_CASE:
      sentence (mac dinh) — chu thuong + dau cau dau cau; ten rieng giua cau van thuong.
      lower          — toan chu thuong.
      original       — giu nhu decoder (thuong toan HOA).
    """
    if not text:
        return text
    _itn = os.environ.get("GIPFORMER_SPOKEN_ITN", "1").strip().lower()
    if _itn not in ("0", "false", "no", "off"):
        text = apply_vi_spoken_number_rules(text)
    mode = os.environ.get("GIPFORMER_OUTPUT_CASE", "sentence").strip().lower()
    if mode in ("original", "none", "keep"):
        return text
    if mode == "lower":
        return text.lower()
    return _sentence_case_vi(text)


def _ensure_deps() -> None:
    if sherpa_onnx is None:
        raise RuntimeError('Thieu sherpa-onnx. Chay: pip install sherpa-onnx')
    if hf_hub_download is None:
        raise RuntimeError('Thieu huggingface_hub. Chay: pip install huggingface_hub')


def _resolve_ffmpeg_exe() -> str:
    """ffmpeg tren PATH, hoac binary di kem imageio-ffmpeg (pip)."""
    system = shutil.which('ffmpeg')
    if system:
        return system
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            logger.info('Dung ffmpeg tu imageio-ffmpeg: %s', bundled)
            return bundled
    except Exception as e:
        logger.debug('imageio-ffmpeg khong khoi tao duoc: %s', e)
    raise RuntimeError(
        'Can ffmpeg de doi WebM/Opus -> WAV 16kHz. '
        'Chon mot: (1) pip install imageio-ffmpeg  (2) cai ffmpeg va them vao PATH '
        '(winget install ffmpeg)'
    )


def _ffmpeg_to_wav_16k_mono(src: Path, dst_wav: Path, timeout_sec: int = 900) -> None:
    ffmpeg = _resolve_ffmpeg_exe()
    cmd = [
        ffmpeg,
        '-nostdin',
        '-y',
        '-i',
        str(src),
        '-ar',
        str(SAMPLE_RATE),
        '-ac',
        '1',
        '-f',
        'wav',
        str(dst_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout_sec, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '')[-2500:]
        raise RuntimeError(f'ffmpeg that bai (exit {result.returncode}): {tail}')


def prepare_wav_from_bytes(data: bytes, suffix: str) -> Path:
    if not suffix.startswith('.'):
        suffix = f'.{suffix}'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f_in:
        f_in.write(data)
        in_path = Path(f_in.name)
    fd_out, out_name = tempfile.mkstemp(suffix='.wav')
    os.close(fd_out)
    out_path = Path(out_name)
    try:
        _ffmpeg_to_wav_16k_mono(in_path, out_path)
        return out_path
    finally:
        in_path.unlink(missing_ok=True)


def download_model(quantize: str) -> dict:
    _ensure_deps()
    files = ONNX_FILES[quantize]
    local_root = os.environ.get('GIPFORMER_LOCAL_MODEL_DIR', '').strip()
    if local_root:
        root = Path(local_root)
        paths = {}
        for key, filename in files.items():
            p = root / filename
            if not p.is_file():
                raise RuntimeError(f'Thong ASR local thieu file: {p} (GIPFORMER_LOCAL_MODEL_DIR={root})')
            paths[key] = str(p.resolve())
        tok = root / 'tokens.txt'
        if not tok.is_file():
            raise RuntimeError(f'Thong ASR local thieu: {tok}')
        paths['tokens'] = str(tok.resolve())
        logger.info('Dung gipformer local tai %s (quantize=%s)', root, quantize)
        return paths

    logger.info('Tai gipformer (%s) tu Hugging Face (%s) …', quantize, REPO_ID)
    paths = {}
    for key, filename in files.items():
        paths[key] = hf_hub_download(repo_id=REPO_ID, filename=filename)
    paths['tokens'] = hf_hub_download(repo_id=REPO_ID, filename='tokens.txt')
    logger.info('Da tai xong ONNX + tokens.')
    return paths


def create_recognizer(model_paths: dict, num_threads: int) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=model_paths['encoder'],
        decoder=model_paths['decoder'],
        joiner=model_paths['joiner'],
        tokens=model_paths['tokens'],
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method='modified_beam_search',
    )


def get_recognizer():
    global _recognizer
    with _model_lock:
        if _recognizer is None:
            _ensure_deps()
            quantize = os.environ.get('GIPFORMER_QUANTIZE', 'int8').lower()
            if quantize not in ONNX_FILES:
                quantize = 'int8'
            threads = int(os.environ.get('GIPFORMER_NUM_THREADS', '4'))
            paths = download_model(quantize)
            _recognizer = create_recognizer(paths, num_threads=threads)
            logger.info('Sherpa OfflineRecognizer da san sang (quantize=%s).', quantize)
        return _recognizer


def _decode_chunk(recognizer, samples: np.ndarray, sample_rate: int) -> str:
    """Giai mot doan waveform (float32 mono)."""
    stream = recognizer.create_stream()
    stream.accept_waveform(int(sample_rate), np.ascontiguousarray(samples, dtype=np.float32))
    recognizer.decode_streams([stream])
    return (stream.result.text or '').strip()


def transcribe_wav_path(wav_path: Path) -> str:
    """
    Offline transducer: mot lan decode cho audio ngan (mac dinh toi ~28s) de tranh cat ranh gioi lam lech cau.
    Audio dai hon: cat doan — GIPFORMER_CHUNK_SECONDS (mac dinh 24, toi da 25).

    GIPFORMER_SINGLE_PASS_MAX_SEC: neu doan <= gia tri nay (gioi han an toan ~30s) thi khong cat.
    """
    recognizer = get_recognizer()
    samples, sample_rate = sf.read(str(wav_path), dtype='float32')
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f'Sample rate {sample_rate} khac {SAMPLE_RATE}. Kiem tra buoc ffmpeg.'
        )

    total = len(samples)
    duration_sec = total / float(SAMPLE_RATE)

    single_max = float(os.environ.get('GIPFORMER_SINGLE_PASS_MAX_SEC', '28'))
    single_max = max(10.0, min(single_max, 29.5))
    single_samples = int(single_max * SAMPLE_RATE)

    if total <= single_samples:
        text = _decode_chunk(recognizer, samples, sample_rate)
        return _normalize_recognized_text(text)

    chunk_sec = float(os.environ.get('GIPFORMER_CHUNK_SECONDS', '24'))
    chunk_sec = max(8.0, min(chunk_sec, 25.0))
    chunk_samples = int(chunk_sec * SAMPLE_RATE)

    pieces = []
    for start in range(0, total, chunk_samples):
        end = min(start + chunk_samples, total)
        chunk = samples[start:end]
        if len(chunk) < int(0.35 * SAMPLE_RATE):
            continue
        seg = _decode_chunk(recognizer, chunk, sample_rate)
        if seg:
            pieces.append(seg)

    logger.info(
        'ASR audio dai %.1fs - xu ly %s doan (moi ~%.0fs, single-pass toi %.0fs)',
        duration_sec,
        len(pieces),
        chunk_sec,
        single_max,
    )
    merged = ' '.join(pieces).strip()
    return _normalize_recognized_text(merged)


def transcribe_upload_bytes(data: bytes, filename: str) -> str:
    ext = Path(filename or 'capture.webm').suffix.lower() or '.webm'
    wav_path = None
    try:
        wav_path = prepare_wav_from_bytes(data, ext)
        return transcribe_wav_path(wav_path)
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)


def warmup() -> None:
    """Goi khi startup de tai model som (khong bat buoc)."""
    get_recognizer()
