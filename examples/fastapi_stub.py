"""
API ASR that cho Vi-VibeVoice: inference that voi gipformer-65M-rnnt (sherpa-onnx).

Chay:
  cd examples
  ..\\.venv\\Scripts\\activate
  uvicorn fastapi_stub:app --host 127.0.0.1 --port 8000

Bien moi truong:
  GIPFORMER_QUANTIZE=int8|fp32   (mac dinh int8 — nhanh, nhe)
  GIPFORMER_NUM_THREADS=4
  GIPFORMER_OUTPUT_CASE=sentence|lower|original  (mac dinh sentence: thuong + dau cau)

Can: ffmpeg tren PATH (doi WebM tu Electron sang WAV 16kHz).
Lan dau: tu dong tai ONNX tu Hugging Face (~100MB voi int8).
"""
from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from gipformer_asr import get_recognizer, transcribe_upload_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Vi-VibeVoice ASR',
    description='g-group-ai-lab/gipformer-65M-rnnt qua sherpa-onnx + ffmpeg',
)


@app.on_event('startup')
def _startup_warmup() -> None:
    try:
        logger.info('Dang khoi tai gipformer (lan dau co the lau do tai HF)…')
        get_recognizer()
        logger.info('ASR engine san sang.')
    except Exception:
        logger.error('Khoi tai model that bai — lan request dau se thu lai:\n%s', traceback.format_exc())


@app.get('/health')
def health() -> dict:
    return {'ok': True, 'engine': 'gipformer-65M-rnnt-sherpa-onnx'}


@app.post('/asr/transcribe')
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default='vi'),
    model: str = Form(default='g-group-ai-lab/gipformer-65M-rnnt'),
):
    """
    Tra ve JSON: {\"text\": \"...\" } — cung contract ma Electron app mac dinh doi.
    """
    try:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail='File audio rong')

        text = transcribe_upload_bytes(content, audio.filename or 'capture.webm')
        return {'text': text}
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.warning('ASR runtime: %s', e)
        return JSONResponse(status_code=500, content={'detail': str(e)})
    except Exception as e:
        logger.exception('ASR loi')
        return JSONResponse(
            status_code=500,
            content={'detail': str(e), 'trace': traceback.format_exc()},
        )
