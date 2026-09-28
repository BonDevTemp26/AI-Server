"""Stage 3 — VLM verification of candidate clips with free Hugging Face models.

Two interchangeable backends (``backend:`` in configs/vlm_verifier.yaml), both
consuming the same prompt and emitting the same ``VerificationResult``:

* ``openai_compatible`` — hosted inference. Default endpoint is the Hugging
  Face **Inference Providers router** (OpenAI-compatible; free monthly credits
  with an ``HF_TOKEN``). The same client also talks to OpenRouter's free tier
  or a self-hosted vLLM/LM Studio server — only endpoint/model/key change.
* ``hf_local`` — fully local, unlimited and key-free. Downloads an open-weight
  VLM from the Hub and runs it via transformers' ``image-text-to-text``
  pipeline. Any Hub model supporting that task works: Qwen2.5-VL, SmolVLM2,
  LLaVA-OneVision, InternVL, ... (pick a size that fits your GPU; see the
  ``local:`` section of the config).

The clip is sent as N uniformly-sampled frames plus the verification prompt;
the model must answer with a JSON verdict which is parsed defensively (LLMs
decorate JSON with prose).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path

import cv2

from pipeline.events import CandidateEvent, VerificationResult

logger = logging.getLogger(__name__)


def sample_clip_frames(clip_path: str | Path, max_frames: int = 8,
                       max_side: int = 768) -> list[bytes]:
    """Uniformly sample up to ``max_frames`` JPEG-encoded frames from a clip."""
    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise IOError(f"Unreadable clip: {clip_path}")
    step = max(1, total // max_frames)
    jpegs, idx = [], 0
    while len(jpegs) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            jpegs.append(buf.tobytes())
        idx += step
    cap.release()
    return jpegs


def _parse_verdict(text: str) -> tuple[str, float, str]:
    """Extract {verdict, confidence, description} from a possibly chatty reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "uncertain")).lower().strip()
            if verdict not in ("confirmed", "rejected", "uncertain"):
                verdict = "uncertain"
            conf = float(data.get("confidence", 0.5))
            return verdict, min(max(conf, 0.0), 1.0), str(data.get("description", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Small local models often emit truncated/malformed JSON — regex the fields
    # out even without a parseable object.
    vm = re.search(r"verdict\W{0,4}(confirmed|rejected|uncertain)", text, re.IGNORECASE)
    if vm:
        cm = re.search(r"confidence\W{0,4}([01](?:\.\d+)?)", text, re.IGNORECASE)
        dm = re.search(r'description\W{0,4}"([^"]{3,300})', text, re.IGNORECASE)
        conf = min(max(float(cm.group(1)), 0.0), 1.0) if cm else 0.4
        return vm.group(1).lower(), conf, (dm.group(1) if dm else text[:300])
    lowered = text.lower()          # last-resort keyword fallback
    if "confirmed" in lowered and "not confirmed" not in lowered:
        return "confirmed", 0.5, text[:300]
    if "rejected" in lowered or "no theft" in lowered:
        return "rejected", 0.5, text[:300]
    return "uncertain", 0.3, text[:300]


def _failure_result(cfg: dict, model: str, err: Exception | str) -> VerificationResult:
    fallback = str(cfg.get("policy", {}).get("on_api_failure", "uncertain"))
    return VerificationResult(verdict=fallback, confidence=0.0,
                              description=f"VLM failed: {err}", model=model)


# Non-transient HTTP statuses: retrying won't help, so fail fast with a hint.
_HTTP_HINTS = {
    401: "invalid/missing API key — set HF_TOKEN in .env and restart",
    402: "free inference credits exhausted — wait for the monthly reset or "
         "switch to  backend: hf_local  (unlimited, local)",
    403: "token not authorized for Inference Providers — create a token at "
         "https://huggingface.co/settings/tokens with the 'Make calls to "
         "Inference Providers' permission (or a classic 'Read' token), put it "
         "in .env as HF_TOKEN, and restart",
    404: "model not served by any Inference Provider — pick a served one "
         "(check https://huggingface.co/api/models/<model-id>"
         "?expand=inferenceProviderMapping), e.g. Qwen/Qwen2.5-VL-72B-Instruct",
}


class _PromptMixin:
    def _load_prompt(self, cfg: dict) -> None:
        prompt_path = Path(cfg["prompt_file"])
        if not prompt_path.is_absolute():
            prompt_path = Path(__file__).resolve().parent.parent / prompt_path
        self.prompt = prompt_path.read_text().strip()
        self.max_frames = int(cfg.get("max_frames", 8))
        self.frame_max_side = int(cfg.get("frame_max_side", 768))

    def _prompt_text(self, event: CandidateEvent) -> str:
        return self.prompt.format(camera_id=event.camera_id,
                                  top_action=event.top_action,
                                  score=f"{event.score:.2f}")


class OpenAICompatibleVerifier(_PromptMixin):
    """Hosted VLM over any OpenAI-compatible chat endpoint.

    Default config targets the HF Inference Providers router
    (``https://router.huggingface.co/v1/chat/completions`` +
    ``Qwen/Qwen2.5-VL-7B-Instruct`` + ``$HF_TOKEN``).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.endpoint = cfg["endpoint"]
        self.model = cfg["model"]
        self.api_key = os.environ.get(cfg.get("api_key_env", "HF_TOKEN"), "")
        if not self.api_key:
            logger.warning("No API key in $%s — hosted VLM calls will fail "
                           "(policy.on_api_failure applies). For key-free "
                           "operation use  backend: hf_local",
                           cfg.get("api_key_env", "HF_TOKEN"))
        self._load_prompt(cfg)
        self.timeout_s = float(cfg.get("timeout_s", 120))
        self.max_retries = int(cfg.get("max_retries", 2))
        logger.info("VLM backend: hosted %s via %s", self.model, self.endpoint)

    def verify(self, clip_path: str | Path, event: CandidateEvent) -> VerificationResult:
        import requests

        frames = sample_clip_frames(clip_path, self.max_frames, self.frame_max_side)
        content = [{"type": "text", "text": self._prompt_text(event)}]
        content += [{"type": "image_url", "image_url": {"url":
                     "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                    for j in frames]
        payload = {"model": self.model, "temperature": 0.1, "max_tokens": 400,
                   "messages": [{"role": "user", "content": content}]}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(self.endpoint, json=payload, headers=headers,
                                     timeout=self.timeout_s)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                verdict, conf, desc = _parse_verdict(text)
                logger.info("VLM verdict for %s: %s (%.2f)", event.event_id, verdict, conf)
                return VerificationResult(verdict=verdict, confidence=conf,
                                          description=desc, model=self.model,
                                          raw_response=text[:2000])
            except Exception as exc:
                last_err = exc
                resp_obj = getattr(exc, "response", None)
                status = getattr(resp_obj, "status_code", None)
                body = (getattr(resp_obj, "text", "") or "")[:300]
                logger.warning("VLM call failed (attempt %d/%d): %s%s",
                               attempt + 1, self.max_retries + 1, exc,
                               f" | response: {body}" if body else "")
                if status in _HTTP_HINTS:
                    logger.warning("VLM HTTP %d — %s", status, _HTTP_HINTS[status])
                    break                        # non-transient: don't retry
                time.sleep(2 ** attempt)
        return _failure_result(self.cfg, self.model, last_err)


class HFLocalVerifier(_PromptMixin):
    """Fully local verification with an open-weight VLM from the HF Hub.

    Model-agnostic via ``AutoModelForImageTextToText`` (Qwen2.5-VL, SmolVLM2,
    LLaVA-OneVision, InternVL, ...). Multi-frame verification needs a
    multi-image/video-capable model (all of the above are); single-image
    models should set ``max_frames: 1``.

    Small models tend to autocomplete a long prompt instead of answering it,
    so the assistant turn is **primed with the start of the JSON verdict**
    (``{"verdict": "``) via ``continue_final_message`` — generation then has
    to continue the JSON rather than wander off. Falls back to a normal
    generation prompt for chat templates that can't continue a final message.
    """

    JSON_PRIMER = '{"verdict": "'

    def __init__(self, cfg: dict):
        self.cfg = cfg
        lcfg = cfg.get("local", {})
        self.model = str(lcfg.get("model_id", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"))
        self.max_new_tokens = int(lcfg.get("max_new_tokens", 120))
        self._load_prompt(cfg)
        # Schema-free prompt for small models (they parrot JSON templates);
        # the JSON shape comes from response priming instead.
        if lcfg.get("prompt_file"):
            self.prompt = Path(lcfg["prompt_file"]).read_text().strip()
        self.fixed_confidence = float(lcfg.get("fixed_confidence", 0.65))
        # Tiled local models (e.g. SmolVLM) explode many/large frames into huge
        # image-token sequences that overflow their context. Let the local
        # section override the shared frame budget with leaner values.
        self.max_frames = int(lcfg.get("max_frames", self.max_frames))
        self.frame_max_side = int(lcfg.get("frame_max_side", self.frame_max_side))

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "backend: hf_local requires the local VLM extras:\n"
                "  pip install 'transformers>=4.49' accelerate\n"
                "(plus bitsandbytes if local.quantize_4bit is enabled)"
            ) from exc
        self._torch = torch

        # Default to CPU: on a small GPU (≤4 GB) Stages 1–2 already fill VRAM,
        # and Stage 3 runs off the live hot path so CPU latency is acceptable.
        self.device = str(lcfg.get("device", "cpu"))
        load_kwargs: dict = {}
        move_to_device = False
        if bool(lcfg.get("quantize_4bit", False)):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            load_kwargs["device_map"] = "auto"     # bitsandbytes places layers itself
            self.device = "cuda"
        elif self.device == "auto":
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = "auto"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cpu":
            load_kwargs["torch_dtype"] = torch.float32  # bf16/fp16 is slow on CPU
            move_to_device = True
        else:
            load_kwargs["torch_dtype"] = "auto"
            move_to_device = True

        logger.info("Loading local VLM %s (first run downloads from the Hub)...",
                    self.model)
        self.processor = AutoProcessor.from_pretrained(self.model)
        self.model_obj = AutoModelForImageTextToText.from_pretrained(
            self.model, **load_kwargs)
        if move_to_device:
            self.model_obj.to(self.device)
        self.model_obj.eval()
        logger.info("VLM backend: local %s ready on %s", self.model, self.device)

    @staticmethod
    def _jpegs_to_pil(jpegs: list[bytes]) -> list:
        from PIL import Image
        return [Image.open(io.BytesIO(j)).convert("RGB") for j in jpegs]

    @staticmethod
    def _build_messages(prompt_text: str, images: list) -> list[dict]:
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def _tokenize(self, messages: list[dict]) -> tuple[dict, bool]:
        """Chat-template + tokenize; returns (inputs, primed_with_json_start)."""
        primed_messages = messages + [{"role": "assistant", "content": [
            {"type": "text", "text": self.JSON_PRIMER}]}]
        try:
            inputs = self.processor.apply_chat_template(
                primed_messages, tokenize=True, return_dict=True,
                return_tensors="pt", continue_final_message=True)
            return inputs, True
        except (ValueError, TypeError, KeyError) as exc:
            logger.debug("continue_final_message unsupported (%s) — plain prompt", exc)
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True)
            return inputs, False

    def verify(self, clip_path: str | Path, event: CandidateEvent) -> VerificationResult:
        torch = self._torch
        try:
            frames = sample_clip_frames(clip_path, self.max_frames, self.frame_max_side)
            messages = self._build_messages(self._prompt_text(event),
                                            self._jpegs_to_pil(frames))
            inputs, primed = self._tokenize(messages)
            inputs = inputs.to(self.device) if hasattr(inputs, "to") else {
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                out = self.model_obj.generate(**inputs,
                                              max_new_tokens=self.max_new_tokens,
                                              do_sample=False)
            text = self.processor.decode(out[0][prompt_len:], skip_special_tokens=True)
            if primed:
                text = self.JSON_PRIMER + text.lstrip()
            verdict, conf, desc = _parse_verdict(text)
            # Small models rarely state a numeric confidence; substitute the
            # configured fixed value so policy gating stays meaningful.
            if not re.search(r'confidence"?\s*[:=]\s*[01](?:\.\d+)?', text, re.IGNORECASE):
                conf = self.fixed_confidence
            logger.info("VLM verdict for %s: %s (%.2f)", event.event_id, verdict, conf)
            return VerificationResult(verdict=verdict, confidence=conf,
                                      description=desc, model=self.model,
                                      raw_response=text[:2000])
        except Exception as exc:
            if "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError":
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                logger.error(
                    "Local VLM OOM for %s — the GPU is already used by Stages 1–2. "
                    "Set  local.device: cpu  in configs/vlm_verifier.yaml (Stage 3 "
                    "is off the hot path, CPU is fine), and/or lower local.max_frames "
                    "/ local.frame_max_side.", event.event_id)
            else:
                logger.exception("Local VLM inference failed for %s", event.event_id)
            return _failure_result(self.cfg, self.model, exc)


class MockVerifier:
    """Confirms everything with fixed confidence — pipeline plumbing tests."""

    def __init__(self, cfg: dict | None = None, verdict: str = "confirmed",
                 confidence: float = 0.9):
        self.verdict, self.confidence = verdict, confidence

    def verify(self, clip_path, event: CandidateEvent) -> VerificationResult:
        return VerificationResult(verdict=self.verdict, confidence=self.confidence,
                                  description=f"[mock] auto-{self.verdict} for "
                                              f"{event.top_action}", model="mock")


def build_verifier(cfg: dict, backend: str | None = None):
    backend = backend or cfg.get("backend", "openai_compatible")
    if backend == "mock":
        logger.info("VLM backend: MOCK (auto-confirm)")
        return MockVerifier(cfg)
    if backend == "openai_compatible":
        return OpenAICompatibleVerifier(cfg)
    if backend == "hf_local":
        return HFLocalVerifier(cfg)
    raise ValueError(f"Unknown VLM backend: {backend!r} "
                     f"(openai_compatible | hf_local | mock)")


def apply_policy(result: VerificationResult, policy: dict) -> tuple[bool, VerificationResult]:
    """(should_alert, possibly-downgraded result) per vlm_verifier.yaml policy."""
    min_conf = float(policy.get("min_confidence", 0.6))
    if result.verdict == "confirmed" and result.confidence < min_conf:
        result = VerificationResult(verdict="uncertain", confidence=result.confidence,
                                    description=result.description, model=result.model,
                                    raw_response=result.raw_response)
    return result.verdict in set(policy.get("alert_on", ["confirmed"])), result
