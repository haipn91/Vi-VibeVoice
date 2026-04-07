#!/usr/bin/env python3
"""
openWakeWord bridge for Vi-VibeVoice — stdout JSON lines compatible with electron/main.js wake parser.
Default test phrase: "hey_jarvis" (say "hey jarvis"). Other builtins: alexa, hey_mycroft, hey_rhasspy, timer, weather.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time

# Khóa: callback sounddevice + vòng main đều gọi emit — không khóa sẽ xen kẽ byte trên stdout → Electron parse JSON hỏng.
_emit_lock = threading.Lock()


def emit(event: str, **payload) -> None:
    line = json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"
    with _emit_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="openWakeWord listener for Vi-VibeVoice.")
    p.add_argument(
        "--model-name",
        default="hey_jarvis",
        help="Built-in name, e.g. hey_jarvis, alexa, hey_mycroft (underscore or space).",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Score 0..1 to fire wake-word.")
    p.add_argument(
        "--inference-framework",
        default="onnx",
        choices=["onnx", "tflite"],
        help="Use onnx on Windows (recommended).",
    )
    p.add_argument("--chunk-samples", type=int, default=1280, help="1280 = 80 ms @ 16 kHz (recommended).")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--debounce-sec", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    emit("status", message="openwakeword_bridge: khoi dong…")

    try:
        import numpy as np
    except ImportError as exc:
        emit(
            "error",
            message=f"Thieu numpy: {exc}. Cai: pip install numpy",
        )
        return 1

    try:
        import sounddevice as sd
    except ImportError as exc:
        emit(
            "error",
            message=f"Thieu sounddevice: {exc}. Cai: pip install sounddevice",
        )
        return 1

    try:
        import openwakeword
        from openwakeword.model import Model
    except ImportError as exc:
        emit(
            "error",
            message=(
                f"Thieu openwakeword: {exc}. Cai: pip install -r python/requirements-openwakeword.txt"
            ),
        )
        log_stderr(f"ImportError openwakeword: {exc}")
        return 1

    # Underscore form matches openWakeWord pretrained filenames (hey_jarvis_v0.1.*)
    model_key = args.model_name.strip().replace(" ", "_")
    if not model_key:
        emit("error", message="model-name trong.")
        return 2

    emit("status", message=f"Dang tai openWakeWord (model={model_key}, {args.inference_framework})…")
    oww = None
    try:
        oww = Model(
            wakeword_models=[model_key],
            inference_framework=args.inference_framework,
        )
    except Exception as exc_first:
        emit("status", message=f"Chua co file model — tai ve (lan dau, chi khi can)…")
        log_stderr(f"Model lan 1: {exc_first}")
        try:
            openwakeword.utils.download_models(model_names=[f"{model_key}_v0.1"])
            emit("status", message="download_models xong.")
        except Exception as exc_dl:
            emit("error", message=f"download_models that bai: {exc_dl}")
            log_stderr(f"download_models: {exc_dl}")
            return 3
        try:
            oww = Model(
                wakeword_models=[model_key],
                inference_framework=args.inference_framework,
            )
        except Exception as exc_second:
            emit("error", message=f"Khong khoi tao Model: {exc_second}")
            log_stderr(f"Model lan 2: {exc_second}")
            return 3

    label = list(oww.models.keys())[0] if oww.models else model_key
    emit(
        "status",
        message=(
            f"openWakeWord san sang — noi cum wake tuong ung (vd: hey jarvis). "
            f"Nguong={args.threshold}, nhan={label}"
        ),
    )

    last_fire = 0.0
    chunk = max(1280, int(args.chunk_samples))
    audio_queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        del frames, time_info
        if status:
            log_stderr(f"sounddevice status: {status}")
        audio_queue.put(bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=chunk,
            dtype="int16",
            channels=1,
            callback=on_audio,
        ):
            emit("status", message="Dang lang nghe openWakeWord (cuc bo, ONNX/tflite)…")
            last_debug = 0.0
            frame_n = 0
            while True:
                data = audio_queue.get()
                audio = np.frombuffer(data, dtype=np.int16).copy()
                if audio.size != chunk:
                    continue
                frame_n += 1
                try:
                    scores = oww.predict(audio)
                except Exception as exc:
                    emit("status", message=f"predict loi: {exc}")
                    log_stderr(f"predict: {exc}")
                    continue

                best = 0.0
                best_name = ""
                for name, sc in scores.items():
                    try:
                        v = float(np.asarray(sc).reshape(-1)[0])
                    except (TypeError, ValueError, IndexError):
                        try:
                            v = float(sc)
                        except (TypeError, ValueError):
                            continue
                    if v > best:
                        best = v
                        best_name = str(name)

                now = time.monotonic()
                if now - last_debug >= 3.0:
                    last_debug = now
                    emit(
                        "status",
                        message=f"[debug oww] frame={frame_n} score_max={best:.3f} (nguong {args.threshold})",
                    )

                if best >= args.threshold and (now - last_fire) >= args.debounce_sec:
                    last_fire = now
                    emit("wake-word", text=best_name or model_key, score=round(best, 4))
    except KeyboardInterrupt:
        emit("status", message="openWakeWord dung boi nguoi dung.")
        return 0
    except Exception as exc:
        emit("error", message=str(exc))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
