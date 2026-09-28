
"""Pipeline orchestrator — wires the four phases per README_Architec.md.

    ingest → [Stage 1] detect+track → ring buffer
                     └→ persons? → [Stage 2] temporal window → trigger
                                        └→ CandidateEvent → verification queue
    verification worker: wait for T+15s → extract clip → [Stage 3] VLM
                                        └→ policy → [Stage 4] alerts + review queue

Threading model: one camera thread per feed (each with its own detector, since
ByteTrack state is per-stream), a shared temporal model behind a lock, and one
background verification worker so slow VLM calls never stall the live loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import yaml

from alerts.alert_manager import AlertManager
from buffering.clip_extractor import ClipExtractor
from buffering.ring_buffer import BufferManager
from detection.tracker import build_detector
from pipeline.events import CandidateEvent, VerificationResult
from temporal.anomaly_scorer import TriggerAggregator
from temporal.base import WindowAssembler, build_temporal_model
from verification.vlm_client import apply_policy, build_verifier


logger = logging.getLogger(__name__)


class PipelineOrchestrator:

    def __init__(
        self,
        app_cfg: dict,
        overrides: dict | None = None
    ):
        self.cfg = app_cfg

        rt = dict(
            app_cfg.get("runtime", {})
        )

        rt.update(
            {
                k: v
                for k, v in (overrides or {}).items()
                if v
            }
        )

        self.runtime = rt

        # ---------------------------------------------------------------
        # Auto-detect GPU for detector.
        #
        # IMPORTANT:
        # PyTorch ROCm uses "cuda" as the device name on AMD.
        #
        # This GPU device is used by YOLO.
        # VideoMAEv2 below is explicitly forced to CPU.
        # ---------------------------------------------------------------

        try:
            import torch

            if torch.cuda.is_available():
                self.runtime["device"] = "cuda"

                print(
                    "[THIEF-DETECT] GPU detected! "
                    "Auto-switching device to 'cuda'.",
                    flush=True
                )

            else:
                raise RuntimeError(
                    "GPU not detected! "
                    "User strict policy: NEVER trigger thief in CPU."
                )

        except ImportError:
            pass

        # ---------------------------------------------------------------
        # Load component configurations
        # ---------------------------------------------------------------

        comp = app_cfg["component_configs"]

        self.detector_cfg = yaml.safe_load(
            Path(
                comp["detector"]
            ).read_text()
        )

        self.temporal_cfg = yaml.safe_load(
            Path(
                comp["temporal"]
            ).read_text()
        )

        self.vlm_cfg = yaml.safe_load(
            Path(
                comp["vlm"]
            ).read_text()
        )

        self.alerts_cfg = yaml.safe_load(
            Path(
                comp["alerts"]
            ).read_text()
        )

        # ---------------------------------------------------------------
        # Buffer / clip configuration
        # ---------------------------------------------------------------

        paths = app_cfg["paths"]

        bcfg = app_cfg.get(
            "buffering",
            {}
        )

        self.buffers = BufferManager(
            duration_s=float(
                bcfg.get(
                    "buffer_duration_s",
                    60
                )
            ),
            store_fps=float(
                bcfg.get(
                    "store_fps",
                    12.5
                )
            ),
            jpeg_quality=int(
                bcfg.get(
                    "jpeg_quality",
                    85
                )
            )
        )

        self.clip_extractor = ClipExtractor(
            paths.get(
                "clips_dir",
                "data/clips"
            )
        )

        # ---------------------------------------------------------------
        # Temporal window assembler
        # ---------------------------------------------------------------

        wcfg = self.temporal_cfg["window"]

        self.windows = WindowAssembler(
            num_frames=int(
                wcfg["num_frames"]
            ),
            sampling_rate=int(
                wcfg["sampling_rate"]
            ),
            clip_fps=float(
                wcfg["clip_fps"]
            ),
            stride_s=float(
                wcfg["stride_s"]
            ),
            input_size=int(
                wcfg["input_size"]
            )
        )

        # ---------------------------------------------------------------
        # Stage 2 — VideoMAEv2
        #
        # IMPORTANT:
        # VideoMAEv2 ONNX is forced to CPU.
        #
        # Reason:
        # ROCm provider loads successfully, but actual inference on the
        # Radeon 890M fails with:
        #
        # hipErrorNoBinaryForGpu
        #
        # CPU inference was tested successfully.
        #
        # YOLO remains on AMD GPU.
        # ---------------------------------------------------------------

        self.temporal = build_temporal_model(
            self.temporal_cfg,
            mode=rt.get(
                "temporal_mode"
            ),
            device=self.runtime.get(
                "device",
                "cuda"
            )
        )

        self._temporal_lock = threading.Lock()

        self.aggregator = TriggerAggregator.from_config(
            self.temporal_cfg
        )

        # ---------------------------------------------------------------
        # Stage 3 — VLM
        # ---------------------------------------------------------------

        self.verifier = build_verifier(
            self.vlm_cfg,
            backend=rt.get(
                "vlm_backend"
            )
        )

        # ---------------------------------------------------------------
        # Stage 4 — Alerts
        # ---------------------------------------------------------------

        self.alert_manager = AlertManager(
            self.alerts_cfg,
            paths.get(
                "alerts_log",
                "data/alerts/alerts.jsonl"
            ),
            paths.get(
                "review_log",
                "data/review/review.jsonl"
            )
        )

        self._verify_queue: queue.Queue[
            CandidateEvent
        ] = queue.Queue()

        self._stop = threading.Event()

        self._camera_done: dict[
            str,
            bool
        ] = {}

        self.stats = {
            "frames": 0,
            "windows": 0,
            "candidates": 0,
            "alerts": 0
        }

        self._stats_lock = threading.Lock()

    # ───────────────────────────────────────────────────────────────────
    # Camera source
    # ───────────────────────────────────────────────────────────────────

    def _make_source(
        self,
        cam: dict
    ):
        backend = self.runtime.get(
            "ingest_backend",
            "opencv"
        )

        if backend == "gstreamer":
            from ingest.gstreamer_pipeline import GStreamerSource

            return GStreamerSource(
                cam["id"],
                cam["source"]
            )

        from ingest.rtsp_client import FrameSource

        return FrameSource(
            cam["id"],
            cam["source"],
            loop=bool(
                cam.get(
                    "loop",
                    False
                )
            )
        )

    # ───────────────────────────────────────────────────────────────────
    # Camera loop — Stages 0–2
    # ───────────────────────────────────────────────────────────────────

    def _camera_loop(
        self,
        cam: dict
    ) -> None:

        cam_id = cam["id"]

        min_persons = int(
            self.detector_cfg.get(
                "min_persons_for_temporal",
                1
            )
        )

        # ---------------------------------------------------------------
        # Stage 0 — MOG2 Motion Gating
        # ---------------------------------------------------------------

        import cv2

        bg_subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=25,
                detectShadows=False
            )
        )

        motion_threshold = 1500

        # Create the AI queue
        ai_queue_max_size = int(self.runtime.get("ai_queue_max_size", 4))
        ai_queue = queue.Queue(maxsize=ai_queue_max_size)

        # -----------------------------------------------------------
        # FrameReader Thread
        # -----------------------------------------------------------
        def frame_reader_thread_fn():
            logger.info(
                "[FRAME-READER] Camera %s: starting source.frames()",
                cam_id
            )
            try:
                source = self._make_source(cam)
                for ts, frame in source.frames():
                    if self._stop.is_set():
                        break

                    # Always append to RingBuffer immediately!
                    self.buffers.append(cam_id, ts, frame)
                    with self._stats_lock:
                        self.stats["frames"] += 1

                    # Non-blocking push to AI queue (drop oldest if full)
                    try:
                        ai_queue.put_nowait((ts, frame))
                    except queue.Full:
                        try:
                            # Drop the oldest frame to make room
                            ai_queue.get_nowait()
                            ai_queue.put_nowait((ts, frame))
                            logger.info("[AI-QUEUE] Camera %s: dropped stale frame, queue full", cam_id)
                        except (queue.Empty, queue.Full):
                            pass
            except Exception:
                logger.exception("[FRAME-READER] Camera %s crashed", cam_id)
            finally:
                logger.info("[FRAME-READER] Camera %s finished", cam_id)
                # Signal AI worker to stop by pushing None
                try:
                    ai_queue.put(None, timeout=1)
                except queue.Full:
                    pass

        # Start the FrameReader thread
        reader_thread = threading.Thread(
            target=frame_reader_thread_fn,
            name=f"reader-{cam_id}",
            daemon=True
        )
        reader_thread.start()

        try:

            # -----------------------------------------------------------
            # Stage 1 — YOLO
            #
            # AMD GPU / ROCm
            #
            # PyTorch ROCm uses device="cuda".
            # -----------------------------------------------------------

            detector = build_detector(
                self.detector_cfg,
                backend=self.runtime.get(
                    "detector_backend",
                    "yolo"
                ),
                device=self.runtime.get(
                    "device",
                    "cuda"
                )
            )

            # -----------------------------------------------------------
            # Consume from AI Queue
            # -----------------------------------------------------------
            frame_counter = 0
            last_persons_count = 0
            while not self._stop.is_set():
                try:
                    item = ai_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if item is None:
                    break # Reader thread exited
                
                ts, frame = item

                logger.info(
                    "[AI-WORKER] Camera %s: processing frame ts=%s queue_size=%d",
                    cam_id,
                    ts,
                    ai_queue.qsize()
                )

                # -------------------------------------------------------
                # Motion detection
                # -------------------------------------------------------

                fg_mask = bg_subtractor.apply(
                    frame
                )

                motion_area = cv2.countNonZero(
                    fg_mask
                )

                if motion_area < motion_threshold:
                    continue

                frame_counter += 1

                # -------------------------------------------------------
                # YOLO detection
                # -------------------------------------------------------

                if frame_counter % 2 != 0:
                    detections = detector.process(
                        frame
                    )
                    persons = sum(
                        1
                        for d in detections
                        if d.class_name == "person"
                    )
                    last_persons_count = persons
                else:
                    detections = []
                    persons = last_persons_count

                logger.info(
                    "[THIEF-DETECT-DEBUG] Camera %s: YOLO detections=%d, skip=%s",
                    cam_id,
                    len(detections),
                    frame_counter % 2 == 0
                )

                # -------------------------------------------------------
                # Build temporal clip
                # -------------------------------------------------------

                clip = self.windows.push(
                    cam_id,
                    frame,
                    ts
                )

                if clip is None:
                    logger.info(
                        "[THIEF-DETECT-DEBUG] Camera %s: temporal clip not ready",
                        cam_id
                    )
                    continue

                logger.info(
                    "[THIEF-DETECT-DEBUG] Camera %s: temporal clip READY",
                    cam_id
                )

                logger.info(
                    "[THIEF-DETECT-DEBUG] Camera %s: persons=%d min_persons=%d",
                    cam_id,
                    persons,
                    min_persons
                )

                if persons < min_persons:
                    logger.info(
                        "[THIEF-DETECT-DEBUG] Camera %s: skipped Stage 2 because persons=%d < min_persons=%d",
                        cam_id,
                        persons,
                        min_persons
                    )
                    continue
                
                # -------------------------------------------------------
                # Stage 2 — VideoMAEv2
                #
                # CPU
                # -------------------------------------------------------

                with self._temporal_lock:

                    result = (
                        self.temporal.process_window(
                            cam_id,
                            ts,
                            clip
                        )
                    )

                with self._stats_lock:
                    self.stats["windows"] += 1

                event = self.aggregator.update(
                    result
                )

                if event:

                    with self._stats_lock:
                        self.stats["candidates"] += 1

                    print(
                        f"[THIEF-DETECT-STAGE-2] "
                        f"Camera {cam_id}: "
                        "Suspicious activity detected "
                        "(CandidateEvent generated). "
                        "Enqueuing for VLM verification...",
                        flush=True
                    )

                    self._verify_queue.put(
                        event
                    )

        except Exception:

            logger.exception(
                "[%s] camera loop crashed",
                cam_id
            )

        finally:

            self._camera_done[
                cam_id
            ] = True
            
            # Allow reader thread to finish
            reader_thread.join(timeout=1.0)

            logger.info(
                "[%s] camera loop finished",
                cam_id
            )

    # ───────────────────────────────────────────────────────────────────
    # Verification helper
    # ───────────────────────────────────────────────────────────────────

    def _wait_for_window(
        self,
        event: CandidateEvent,
        wall_timeout: float
    ) -> None:

        buffer = self.buffers.get(
            event.camera_id
        )

        deadline = (
            time.monotonic()
            + wall_timeout
        )

        while (
            not buffer.covers(
                event.clip_end_ts
            )
            and not self._camera_done.get(
                event.camera_id,
                False
            )
            and not self._stop.is_set()
            and time.monotonic() < deadline
        ):

            time.sleep(
                0.2
            )

    # ───────────────────────────────────────────────────────────────────
    # Verification loop — Stages 3–4
    # ───────────────────────────────────────────────────────────────────

    def _verification_loop(
        self
    ) -> None:

        post_s = float(
            self.temporal_cfg.get(
                "trigger",
                {}
            ).get(
                "clip_post_s",
                15
            )
        )

        while not (
            self._stop.is_set()
            and self._verify_queue.empty()
        ):

            try:

                event = self._verify_queue.get(
                    timeout=0.5
                )

            except queue.Empty:

                if (
                    all(
                        self._camera_done.values()
                    )
                    and self._camera_done
                ):
                    break

                continue

            try:

                self._wait_for_window(
                    event,
                    wall_timeout=post_s + 30
                )

                print(
                    f"[THIEF-DETECT-STAGE-3] "
                    f"Camera {event.camera_id}: "
                    "Extracting video clip for verification "
                    f"(Event ID: {event.event_id})...",
                    flush=True
                )

                clip_path = (
                    self.clip_extractor.extract(
                        event,
                        self.buffers.get(
                            event.camera_id
                        )
                    )
                )

                if clip_path is None:

                    print(
                        f"[THIEF-DETECT-ERROR] "
                        f"Camera {event.camera_id}: "
                        f"Clip extraction failed for event "
                        f"{event.event_id}",
                        flush=True
                    )

                    result = VerificationResult(
                        verdict="uncertain",
                        confidence=0.0,
                        description=(
                            "clip extraction failed "
                            "— needs human review"
                        )
                    )

                else:

                    print(
                        f"[THIEF-DETECT-STAGE-3] "
                        f"Camera {event.camera_id}: "
                        "VLM Verification starting for clip...",
                        flush=True
                    )

                    result = self.verifier.verify(
                        clip_path,
                        event
                    )

                should_alert, result = apply_policy(
                    result,
                    self.vlm_cfg.get(
                        "policy",
                        {}
                    )
                )

                self.alert_manager.log_review_item(
                    event,
                    result,
                    (
                        str(clip_path)
                        if clip_path
                        else None
                    ),
                    should_alert
                )

                print(
                    f"[THIEF-DETECT-STAGE-4] "
                    f"Camera {event.camera_id}: "
                    f"Verification complete. "
                    f"Verdict: '{result.verdict}', "
                    f"Confidence: {result.confidence:.2f}, "
                    f"Alert Triggered: {should_alert}",
                    flush=True
                )

                if should_alert:

                    self.alert_manager.dispatch(
                        event,
                        result,
                        clip_path
                    )

                    with self._stats_lock:
                        self.stats["alerts"] += 1

                else:

                    logger.info(
                        "Candidate %s not alerted "
                        "(verdict: %s) — "
                        "kept in review queue",
                        event.event_id,
                        result.verdict
                    )

            except Exception as e:

                print(
                    f"[THIEF-DETECT-ERROR] "
                    f"Camera "
                    f"{event.camera_id if 'event' in locals() else 'Unknown'}: "
                    f"Verification failed with error: {str(e)}",
                    flush=True
                )

                logger.exception(
                    "Verification failed for %s",
                    (
                        event.event_id
                        if 'event' in locals()
                        else 'Unknown'
                    )
                )

        print(
            "[THIEF-DETECT] Verification loop finished",
            flush=True
        )

        logger.info(
            "verification loop finished"
        )

    # ───────────────────────────────────────────────────────────────────
    # Run pipeline
    # ───────────────────────────────────────────────────────────────────

    def run(
        self
    ) -> dict:

        cameras = self.cfg.get(
            "cameras",
            []
        )

        if not cameras:

            raise ValueError(
                "No cameras configured "
                "(configs/app.yaml → cameras:)"
            )

        self._camera_done = {
            c["id"]: False
            for c in cameras
        }

        # ---------------------------------------------------------------
        # Verification worker
        # ---------------------------------------------------------------

        verifier_thread = threading.Thread(
            target=self._verification_loop,
            name="verifier",
            daemon=True
        )

        verifier_thread.start()

        # ---------------------------------------------------------------
        # Camera threads
        # ---------------------------------------------------------------

        cam_threads = [
            threading.Thread(
                target=self._camera_loop,
                args=(c,),
                name=f"cam-{c['id']}",
                daemon=True
            )
            for c in cameras
        ]

        for t in cam_threads:
            t.start()

        print(
            f"[THIEF-DETECT-STAGE-1] "
            f"Pipeline initialized for "
            f"{len(cameras)} cameras. "
            "Awaiting motion and persons...",
            flush=True
        )

        logger.info(
            "Pipeline running: %d camera(s) | "
            "detector=%s temporal=%s vlm=%s",
            len(cameras),
            self.runtime.get(
                "detector_backend"
            ),
            self.runtime.get(
                "temporal_mode"
            ),
            self.runtime.get(
                "vlm_backend"
            )
        )

        start = time.monotonic()

        try:

            while any(
                t.is_alive()
                for t in cam_threads
            ):

                time.sleep(
                    0.5
                )

                current_max = float(
                    self.runtime.get(
                        "max_runtime_s",
                        0
                    )
                    or 0
                )

                if (
                    current_max
                    and (
                        time.monotonic()
                        - start
                    ) > current_max
                ):

                    logger.info(
                        "max_runtime_s reached — stopping"
                    )

                    self._stop.set()

                    break

        except KeyboardInterrupt:

            logger.info(
                "Ctrl-C — shutting down"
            )

            self._stop.set()

        # ---------------------------------------------------------------
        # Wait for camera threads
        # ---------------------------------------------------------------

        for t in cam_threads:
            t.join(
                timeout=10
            )

        # ---------------------------------------------------------------
        # Allow queued VLM checks to finish
        # ---------------------------------------------------------------

        verifier_thread.join(
            timeout=120
        )

        self._stop.set()

        self.alert_manager.close()

        logger.info(
            "Done. %s",
            self.stats
        )

        return dict(
            self.stats
        )

