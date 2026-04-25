import argparse
import io
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from omnivoice import OmniVoice


app = FastAPI(title="OmniVoice API")
logger = logging.getLogger("omnivoice_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DEFAULT_CHECKPOINT = "k2-fsa/OmniVoice"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

model_lock = threading.Lock()
model: OmniVoice | None = None
model_checkpoint: str | None = None
model_load_args: dict[str, Any] = {}
generation_defaults: dict[str, Any] = {
    "num_step": 32,
    "guidance_scale": 2.0,
    "speed": 1.0,
    "duration": None,
    "t_shift": 0.1,
    "denoise": True,
    "preprocess_prompt": True,
    "postprocess_output": True,
    "layer_penalty_factor": 5.0,
    "position_temperature": 5.0,
    "class_temperature": 0.0,
    "audio_chunk_duration": 15.0,
    "audio_chunk_threshold": 30.0,
}

AudioArray = np.ndarray | torch.Tensor


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda") or device == "mps":
        return torch.float16
    return torch.float32


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    return str(dtype)


def _ensure_model() -> OmniVoice:
    if model is None:
        raise RuntimeError("Model has not been initialized. Start the server via `uv run python server.py`.")
    return model


def _as_float32_audio(audio: AudioArray) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        pcm = audio.squeeze().detach().cpu().numpy()
    else:
        pcm = np.asarray(audio).squeeze()
    return np.clip(pcm.astype(np.float32), -1.0, 1.0)


def _audio_to_pcm16le_bytes(audio: AudioArray) -> bytes:
    pcm = _as_float32_audio(audio)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    return pcm_i16.tobytes()


def _audio_to_wav_bytes(audio: AudioArray, sample_rate: int) -> bytes:
    wav = _as_float32_audio(audio)

    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _load_model(checkpoint: str, device: str | None) -> None:
    global model, model_checkpoint, model_load_args

    resolved_device = device or _select_device()
    resolved_dtype = _select_dtype(resolved_device)

    logger.info(
        "Loading OmniVoice checkpoint=%s device=%s dtype=%s",
        checkpoint,
        resolved_device,
        _dtype_name(resolved_dtype),
    )
    tts = OmniVoice.from_pretrained(
        checkpoint,
        device_map=resolved_device,
        dtype=resolved_dtype,
    )

    model = tts
    model_checkpoint = checkpoint
    model_load_args = {
        "device": resolved_device,
        "dtype": _dtype_name(resolved_dtype),
    }


@app.get("/health")
def health() -> JSONResponse:
    tts = _ensure_model()
    return JSONResponse(
        {
            "status": "ok",
            "checkpoint": model_checkpoint,
            "sample_rate": tts.sampling_rate,
        }
    )


@app.get("/model")
def get_model_info() -> JSONResponse:
    tts = _ensure_model()
    return JSONResponse(
        {
            "checkpoint": model_checkpoint,
            "load_args": model_load_args,
            "sample_rate": tts.sampling_rate,
            "supported_language_ids": sorted(tts.supported_language_ids()),
            "supported_language_names": sorted(tts.supported_language_names()),
            "generation_defaults": generation_defaults,
        }
    )


@app.get("/config")
def get_config() -> JSONResponse:
    return get_model_info()


@app.post("/tts")
async def tts(
    text: str = Form(...),
    language: str | None = Form(default=None),
    instruct: str | None = Form(default=None),
    ref_text: str | None = Form(default=None),
    reference_wav: UploadFile | None = File(default=None),
    num_step: int | None = Form(default=None),
    guidance_scale: float | None = Form(default=None),
    speed: float | None = Form(default=None),
    duration: float | None = Form(default=None),
    t_shift: float | None = Form(default=None),
    denoise: bool | None = Form(default=None),
    preprocess_prompt: bool | None = Form(default=None),
    postprocess_output: bool | None = Form(default=None),
    layer_penalty_factor: float | None = Form(default=None),
    position_temperature: float | None = Form(default=None),
    class_temperature: float | None = Form(default=None),
    audio_chunk_duration: float | None = Form(default=None),
    audio_chunk_threshold: float | None = Form(default=None),
) -> Response:
    tts_model = _ensure_model()

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must be non-empty.")

    normalized_language = (language or "").strip() or None
    normalized_instruct = (instruct or "").strip() or None
    normalized_ref_text = (ref_text or "").strip() or None

    if reference_wav is not None and normalized_instruct is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either `reference_wav` for voice cloning or `instruct` for voice design, not both.",
        )

    generate_kwargs: dict[str, Any] = {
        key: value
        for key, value in {
            "language": normalized_language,
            "instruct": normalized_instruct,
            "ref_text": normalized_ref_text,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "speed": speed,
            "duration": duration,
            "t_shift": t_shift,
            "denoise": denoise,
            "preprocess_prompt": preprocess_prompt,
            "postprocess_output": postprocess_output,
            "layer_penalty_factor": layer_penalty_factor,
            "position_temperature": position_temperature,
            "class_temperature": class_temperature,
            "audio_chunk_duration": audio_chunk_duration,
            "audio_chunk_threshold": audio_chunk_threshold,
        }.items()
        if value is not None
    }

    temp_wav_path: str | None = None
    if reference_wav is not None:
        uploaded = await reference_wav.read()
        if not uploaded:
            raise HTTPException(status_code=400, detail="`reference_wav` was provided but is empty.")
        if reference_wav.filename and not reference_wav.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="`reference_wav` must be a .wav file.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded)
            temp_wav_path = tmp.name
        generate_kwargs["ref_audio"] = temp_wav_path

    logger.info(
        "Received /tts request text_length=%s language=%s mode=%s has_ref_text=%s",
        len(text),
        normalized_language,
        "voice_clone" if reference_wav is not None else ("voice_design" if normalized_instruct else "auto"),
        bool(normalized_ref_text),
    )

    try:
        with model_lock:
            audios = tts_model.generate(
                text=text,
                **generate_kwargs,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        if temp_wav_path is not None:
            Path(temp_wav_path).unlink(missing_ok=True)

    pcm_bytes = _audio_to_pcm16le_bytes(audios[0])
    return Response(
        content=pcm_bytes,
        media_type="audio/pcm",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": str(tts_model.sampling_rate),
            "X-Channels": "1",
        },
    )


@app.post("/voice-design")
async def voice_design(
    prompt: str = Form(...),
    voice_description: str = Form(...),
    language: str | None = Form(default=None),
) -> Response:
    tts_model = _ensure_model()

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise HTTPException(status_code=400, detail="`prompt` must be non-empty.")

    normalized_voice_description = voice_description.strip()
    if not normalized_voice_description:
        raise HTTPException(
            status_code=400,
            detail="`voice_description` must be non-empty.",
        )

    normalized_language = (language or "").strip() or None

    logger.info(
        "Received /voice-design request prompt_length=%s language=%s voice_description=%s",
        len(normalized_prompt),
        normalized_language,
        normalized_voice_description,
    )

    try:
        with model_lock:
            audios = tts_model.generate(
                text=normalized_prompt,
                language=normalized_language,
                instruct=normalized_voice_description,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Voice design generation failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    wav_bytes = _audio_to_wav_bytes(audios[0], tts_model.sampling_rate)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Format": "wav_pcm_s16le",
            "X-Sample-Rate": str(tts_model.sampling_rate),
            "X-Channels": "1",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python server.py",
        description="Launch a FastAPI server for OmniVoice.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Model checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device. Defaults to cuda:0, then mps, then cpu.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_model(checkpoint=args.checkpoint, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
