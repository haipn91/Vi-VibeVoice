#!/usr/bin/env python3
"""
FastAPI hau xu ly van ban sau ASR bang LLM (vLLM — API tuong thich OpenAI).

Khong dung CAPU/rule-based ITN; goi POST /v1/chat/completions toi vLLM voi prompt chuan hoa:
so (chu -> chu so / dang dung), viet hoa dau cau / ten rieng khi hop ly, dau cau day du.

Cai dat:
  pip install -r python/requirements-llm-postprocess-api.txt

Cau hinh (bien moi truong — ban dien URL va ten model sau):
  VLLM_BASE_URL   - Vi du: http://127.0.0.1:8000/v1  (phai co /v1 neu server OpenAI-compatible)
  VLLM_MODEL      - Ten model da load trong vLLM, vi du: Qwen/Qwen2.5-7B-Instruct
  VLLM_API_KEY    - Tuy chon; neu vLLM bat --api-key thi dat Bearer token o day

Chay API nay (cong khac vLLM), vi du:
  python python/llm_text_postprocess_api.py --host 0.0.0.0 --port 8010

Client gui JSON: POST /text/normalize  {"text": "..."}
Tra ve: {"text": "<van ban da chuan hoa>", "timing": {...}}

Tich hop pipeline: ASR (server khac) -> lay text -> POST toi endpoint nay.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_text_postprocess")

# --- Prompt: chi ra mot phien ban ro rang; LLM chi tra ve text da xu ly ---
SYSTEM_PROMPT_VI = """Bạn là bộ chuẩn hóa văn bản sau nhận dạng giọng nói (ASR) tiếng Việt.

Nhiệm vụ: nhận một đoạn văn thô từ ASR và trả về **chỉ một đoạn văn đã chuẩn hóa**, không giải thích, không tiêu đề, không markdown.

Yêu cầu chuẩn hóa:
1. **Số và ngày tháng năm**: chuyển cách nói bằng chữ sang dạng số/định dạng thông dụng khi ngữ cảnh rõ ràng (ví dụ tháng, năm, số đếm ngắn).
2. **Dấu câu**: thêm hoặc sửa dấu chấm, phẩy, hỏi, chấm than cho đúng ngữ điệu câu.
3. **Viết hoa**: viết hoa đầu câu sau dấu kết thúc câu; viết hoa tên riêng, tên địa danh, tổ chức khi có thể suy luận từ ngữ cảnh; không tự bịa thông tin.
4. **Lỗi ASR nhẹ**: sửa lỗi nhận âm hiển nhiên nếu câu trở nên tự nhiên hơn; không thêm nội dung mới không có trong bản gốc.

**Đầu ra**: chỉ một khối văn bản tiếng Việt đã chỉnh sửa, không dòng trống thừa ở đầu/cuối."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def build_user_message(raw_text: str) -> str:
    return (
        "Dưới đây là văn bản thô từ ASR. Hãy chuẩn hóa theo hệ thống và chỉ trả về kết quả cuối cùng.\n\n"
        f"<<<ASR>>>\n{raw_text}\n<<<END>>>"
    )


class NormalizeRequest(BaseModel):
    text: str = Field(..., description="Van ban dau ra tu ASR can hau xu ly")


class NormalizeResponse(BaseModel):
    text: str
    timing: dict[str, Any]


async def call_vllm_chat(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    api_key: Optional[str],
    user_content: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """POST .../chat/completions (OpenAI-compatible, dung boi vLLM)."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_VI},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = await client.post(url, json=payload, headers=headers, timeout=httpx.Timeout(120.0, connect=30.0))
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM tra ve khong co choices: {data!r}")
    msg = (choices[0].get("message") or {}).get("content")
    if msg is None or not str(msg).strip():
        raise RuntimeError(f"vLLM tra ve noi dung rong: {data!r}")
    return str(msg).strip()


def _default_max_tokens(text_len: int) -> int:
    # Du phong cho mo rong sau chuan hoa; toi da tranh cat ngan.
    return min(4096, max(256, text_len * 3 + 128))


app = FastAPI(
    title="Vi-VibeVoice LLM text post-process (vLLM)",
    version="1.0.0",
    description="Hau xu ly text ASR qua LLM (vLLM OpenAI-compatible API).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_env("LLM_POSTPROCESS_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    base = _env("VLLM_BASE_URL")
    model = _env("VLLM_MODEL")
    return {
        "ok": True,
        "service": "llm-text-postprocess",
        "vllm_configured": bool(base and model),
        "vllm_base_url_set": bool(base),
        "vllm_model_set": bool(model),
        "note": "Dat VLLM_BASE_URL (vi du http://host:8000/v1) va VLLM_MODEL truoc khi goi /text/normalize.",
    }


@app.post("/text/normalize", response_model=NormalizeResponse)
async def normalize_text(body: NormalizeRequest) -> NormalizeResponse:
    raw = (body.text or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="text rong")

    base = _env("VLLM_BASE_URL")
    model = _env("VLLM_MODEL")
    if not base or not model:
        raise HTTPException(
            status_code=503,
            detail="Chua cau hinh VLLM_BASE_URL hoac VLLM_MODEL (bien moi truong).",
        )

    api_key = _env("VLLM_API_KEY") or None
    temperature = float(_env("VLLM_TEMPERATURE", "0.2") or "0.2")
    max_tokens = int(_env("VLLM_MAX_TOKENS", "0") or "0") or _default_max_tokens(len(raw))

    user_msg = build_user_message(raw)
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            out = await call_vllm_chat(
                client,
                base_url=base,
                model=model,
                api_key=api_key,
                user_content=user_msg,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    except httpx.HTTPStatusError as e:
        logger.warning("vLLM HTTP loi: %s %s", e.response.status_code, e.response.text[:500])
        raise HTTPException(
            status_code=502,
            detail={
                "msg": f"vLLM tra loi HTTP {e.response.status_code}",
                "vllm_body": e.response.text[:2000],
            },
        ) from e
    except httpx.RequestError as e:
        logger.exception("vLLM ket noi loi")
        raise HTTPException(status_code=502, detail=f"Khong ket noi vLLM: {e}") from e
    except Exception as e:
        logger.exception("vLLM xu ly loi")
        raise HTTPException(
            status_code=500,
            detail={"msg": str(e), "trace": traceback.format_exc()},
        ) from e

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return NormalizeResponse(
        text=out,
        timing={
            "llm_ms": round(elapsed_ms, 2),
            "vllm_model": model,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FastAPI — hau xu ly text ASR qua vLLM")
    parser.add_argument("--host", default=os.environ.get("LLM_POSTPROCESS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LLM_POSTPROCESS_PORT", "8010")))
    args = parser.parse_args()
    logger.info(
        "Listening http://%s:%s  POST /text/normalize  (vLLM: %s)",
        args.host,
        args.port,
        _env("VLLM_BASE_URL") or "(chua dat VLLM_BASE_URL)",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
