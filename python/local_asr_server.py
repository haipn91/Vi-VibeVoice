#!/usr/bin/env python3
"""
ASR cuc bo cho Vi-VibeVoice: gipformer ONNX (sherpa-onnx), chi lang nghe loopback.
Modules ASR/CAPU trong cung thu muc python/ (gipformer_asr.py, vi_capu_punctuate.py, …).

Cai dat: pip install -r python/requirements-local-asr.txt

Dau cau / viet hoa (tuy chon, PyTorch + model ~1.1GB):
  pip install -r python/requirements-capu.txt
  set GIPFORMER_CAPU=1

ONNX (sau khi chay scripts/export_capu_onnx.py):
  set GIPFORMER_CAPU_ONNX=1
  (tuy chon) set GIPFORMER_CAPU_ONNX_PATH=...\\capu-seq2labels.onnx

Chay (thuong do Electron spawn):
  python python/local_asr_server.py --port 18765
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PY_DIR))

import os

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from gipformer_asr import get_recognizer, transcribe_upload_bytes
from vi_capu_punctuate import capu_status_line, maybe_apply_capu_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vi_local_asr")

app = FastAPI(title="Vi-VibeVoice Local ASR", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        logger.info("Dang tai gipformer (cuc bo)…")
        get_recognizer()
        logger.info("Local ASR san sang.")
    except Exception:
        logger.error("Khoi tai model that bai:\n%s", traceback.format_exc())
    if os.environ.get("GIPFORMER_CAPU", "").strip().lower() in ("1", "true", "yes", "on"):
        logger.info(
            "GIPFORMER_CAPU bat — dau cau (xlm-roberta-capu) se tai lan goi /asr/transcribe dau tien."
        )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "engine": "gipformer-local-onnx", "capu": capu_status_line()}


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
