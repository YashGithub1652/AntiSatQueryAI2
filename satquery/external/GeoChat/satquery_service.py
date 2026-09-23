from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------
# GeoChat source
# ---------------------------------------------------------------------

GEoCHAT_ROOT = r"C:\SIH\GeoChat"

if GEoCHAT_ROOT not in sys.path:
    sys.path.insert(0, GEoCHAT_ROOT)

from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
)
from geochat.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from geochat.conversation import conv_templates


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery-geochat")


MODEL_PATH = "MBZUAI/GeoChat-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    raise RuntimeError(
        "GeoChat service requires CUDA for the intended SatQuery deployment."
    )


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------

app = FastAPI(
    title="SatQuery GeoChat Inference Host",
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------

class VQARequest(BaseModel):
    image_base64: str
    query: str
    max_new_tokens: int = 256


class VQAResponse(BaseModel):
    success: bool
    answer: str
    model: str
    scientific: bool
    latency_sec: float
    fallback: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------

tokenizer = None
model = None
image_processor = None
context_len = None


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model() -> None:
    global tokenizer
    global model
    global image_processor
    global context_len

    if model is not None:
        return

    logger.info("Loading official GeoChat-7B model...")

    model_name = get_model_name_from_path(MODEL_PATH)

    tokenizer, model, image_processor, context_len = (
        load_pretrained_model(
            MODEL_PATH,
            None,
            model_name,
            load_8bit=False,
            load_4bit=True,
            device=DEVICE,
        )
    )

    model.eval()

    logger.info(
        "GeoChat loaded successfully. "
        "model=%s device=%s context=%s",
        type(model).__name__,
        DEVICE,
        context_len,
    )


@app.on_event("startup")
def startup() -> None:
    load_model()


# ---------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------

def decode_image(encoded: str) -> Image.Image:
    try:
        raw = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image payload: {exc}",
        )


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "MBZUAI/GeoChat-7B",
        "model_loaded": model is not None,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "scientific": True,
    }


@app.post("/v1/vqa", response_model=VQAResponse)
def run_vqa(request: VQARequest):

    started = time.perf_counter()

    try:

        load_model()

        image = decode_image(request.image_base64)

        # -------------------------------------------------------------
        # GeoChat official image preprocessing
        # -------------------------------------------------------------
        preprocess_started = time.perf_counter()
        image_tensor = image_processor.preprocess(
            image,
            crop_size={
                "height": 504,
                "width": 504,
            },
            size={
                "shortest_edge": 504,
            },
            return_tensors="pt",
        )["pixel_values"]

        image_tensor = image_tensor.half().cuda()
        preprocess_elapsed = time.perf_counter() - preprocess_started
        logger.info("GeoChat image preprocessing: %.3fs", preprocess_elapsed)

        # -------------------------------------------------------------
        # Conversation
        # -------------------------------------------------------------

        conv = conv_templates["vicuna_v1"].copy()

        prompt = (
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + request.query.strip()
        )

        conv.append_message(
            conv.roles[0],
            prompt,
        )

        conv.append_message(
            conv.roles[1],
            None,
        )

        final_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            final_prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).cuda()

        # -------------------------------------------------------------
        # Generate
        # -------------------------------------------------------------
        generate_started = time.perf_counter()
        with torch.inference_mode():

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=request.max_new_tokens,
                use_cache=True,
            )
            generate_elapsed = time.perf_counter() - generate_started
            logger.info("GeoChat model.generate: %.3fs", generate_elapsed)


        # Decode only newly generated tokens.
        # GeoChat input_ids contain IMAGE_TOKEN_INDEX (-200), which is
        # intentionally not a SentencePiece vocabulary ID.
        generated_ids = output_ids[:, input_ids.shape[1]:]

        output = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        elapsed = time.perf_counter() - started

        return VQAResponse(
            success=True,
            answer=output,
            model="GeoChat-7B",
            scientific=True,
            latency_sec=elapsed,
            fallback=False,
        )

    except Exception as exc:

        logger.exception(
            "GeoChat inference failed."
        )

        # Decode only newly generated tokens.
        # GeoChat input_ids contain IMAGE_TOKEN_INDEX (-200), which is
        # intentionally not a SentencePiece vocabulary ID.
        generated_ids = output_ids[:, input_ids.shape[1]:]

        output = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        elapsed = time.perf_counter() - started

        return VQAResponse(
            success=False,
            answer="",
            model="GeoChat-7B",
            scientific=False,
            latency_sec=elapsed,
            fallback=False,
            error=str(exc),
        )

