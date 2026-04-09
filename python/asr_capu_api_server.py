#!/usr/bin/env python3
"""
API ASR (gipformer / sherpa-onnx) + CAPU tren mot may chu — ung dung Electron gui audio, nhan JSON {text, timing}.

Cai dat (tren may GPU/Linux hoac Windows Server):
  cd <repo>
  pip install -r python/requirements-asr-capu-api.txt
  # Tuy chon CAPU ONNX tren GPU: pip uninstall -y onnxruntime && pip install onnxruntime-gpu

Mo hinh / file:
  - ASR: HF gipformer ONNX (CPU, sherpa-onnx — nhanh voi int8).
  - CAPU: ONNX hoac PyTorch HF (xem env duoi).

Chay API (mac dinh cho phep ket noi mang LAN):
  python python/asr_capu_api_server.py --host 0.0.0.0 --port 8000

Hoac PowerShell:  .\\scripts\\run_asr_capu_api_server.ps1

Bien moi truong (tuy chon):
  GIPFORMER_CAPU_ONNX=1          - CAPU qua ONNX Runtime
  GIPFORMER_CAPU_ONNX_PATH=...   - duong dan .onnx (FP32 hoac .int8.onnx)
  GIPFORMER_CAPU=1               - fallback / chu PyTorch HF (dung GPU neu co torch.cuda)
  GIPFORMER_CAPU_ORT_CPU_ONLY=1  - ep ORT CAPU chi chay CPU
  HF_HOME                        - cache model

Ket noi tu Vi-VibeVoice:
  Settings -> Mode: Remote API
  ASR endpoint: http://<IP-may-GPU>:8000/asr/transcribe
  Audio field: audio
  JSON path: text
  Extra form: {"language":"vi","model":"g-group-ai-lab/gipformer-65M-rnnt"}

Trien khai chi thu muc deploy (doc lap): xem deploy/README.md + deploy/server.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

if sys.platform == "win32":
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import time
import traceback
from pathlib import Path

# ASR/CAPU helpers live in this directory (gipformer_asr.py, vi_capu_punctuate.py, …).
ROOT = Path(__file__).resolve().parent.parent
_PY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PY_DIR))

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gipformer_asr import get_recognizer, transcribe_upload_bytes
from vi_capu_punctuate import capu_status_line, maybe_apply_capu_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vi_asr_capu_api")

app = FastAPI(title="Vi-VibeVoice ASR+CAPU API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ASR_API_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    try:
        import torch

        cuda = torch.cuda.is_available()
        logger.info("Torch CUDA available=%s device_count=%s", cuda, torch.cuda.device_count())
    except Exception as e:
        logger.info("Torch CUDA check: %s", e)

    try:
        import onnxruntime as ort

        logger.info("ONNX Runtime providers: %s", ort.get_available_providers())
    except Exception as e:
        logger.info("ONNX Runtime: %s", e)

    try:
        logger.info("Tai gipformer ASR…")
        get_recognizer()
        logger.info("ASR san sang.")
    except Exception:
        logger.error("Khoi tai ASR that bai:\n%s", traceback.format_exc())
    logger.info("CAPU: %s", capu_status_line())


@app.get("/health")
def health() -> dict:
    cuda = False
    cuda_name = None
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            cuda_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    ort_providers: list[str] = []
    try:
        import onnxruntime as ort

        ort_providers = list(ort.get_available_providers())
    except Exception:
        pass
    return {
        "ok": True,
        "service": "vi-vibevoice-asr-capu",
        "engine": "gipformer-sherpa-onnx",
        "asr_note": "ASR dang dung ten xu ly sherpa-onnx (mac dinh CPU, int8).",
        "capu": capu_status_line(),
        "cuda_available": cuda,
        "cuda_device": cuda_name,
        "onnxruntime_providers": ort_providers,
    }


@app.post("/asr/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default="vi"),
    model: str = Form(default="g-group-ai-lab/gipformer-65M-rnnt"),
):
    t0 = time.perf_counter()
    try:
        t_read0 = time.perf_counter()
        content = await audio.read()
        read_upload_ms = (time.perf_counter() - t_read0) * 1000.0
        if not content:
            raise HTTPException(status_code=400, detail="File audio rong")
        t_asr0 = time.perf_counter()
        text = transcribe_upload_bytes(content, audio.filename or "capture.webm")
        asr_ms = (time.perf_counter() - t_asr0) * 1000.0
        t_capu0 = time.perf_counter()
        text = maybe_apply_capu_text(text)
        capu_ms = (time.perf_counter() - t_capu0) * 1000.0
        server_total_ms = (time.perf_counter() - t0) * 1000.0
        timing = {
            "read_upload_ms": round(read_upload_ms, 2),
            "asr_ms": round(asr_ms, 2),
            "capu_ms": round(capu_ms, 2),
            "server_total_ms": round(server_total_ms, 2),
        }
        return {"text": text, "timing": timing}
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.warning("ASR runtime: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
    except Exception as e:
        logger.exception("ASR loi")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "trace": traceback.format_exc()},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR+CAPU FastAPI for Vi-VibeVoice remote clients")
    parser.add_argument("--host", default=os.environ.get("ASR_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ASR_API_PORT", "8000")))
    args = parser.parse_args()
    logger.info("Listening http://%s:%s  (transcribe POST /asr/transcribe)", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
