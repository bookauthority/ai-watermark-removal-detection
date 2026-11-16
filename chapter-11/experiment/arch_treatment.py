from __future__ import annotations

import queue
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .util import SimulatedImage, busy_work


@dataclass
class _QueuedItem:
    request: Dict[str, Any]
    future: "ResultFuture"


class ResultFuture:
    """Simple thread-safe future for passing results back to ingress.

    This is a minimal condition-based implementation sufficient for this
    experiment. It supports setting a result or an error and waiting for it.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[Exception] = None
        self._done = False

    def set_result(self, result: Dict[str, Any]) -> None:
        with self._cond:
            if self._done:
                return
            self._result = result
            self._done = True
            self._cond.notify_all()

    def set_error(self, exc: Exception) -> None:
        with self._cond:
            if self._done:
                return
            self._error = exc
            self._done = True
            self._cond.notify_all()

    def result(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        with self._cond:
            if not self._done:
                self._cond.wait(timeout=timeout)
            if not self._done:
                raise TimeoutError("Result not ready in time")
            if self._error is not None:
                raise self._error
            assert self._result is not None
            return self._result


@dataclass
class TreatmentPipeline:
    """Simulated microservice-style detection pipeline.

    Components:
    - ingress enqueues jobs into detect_queue
    - feature workers consume detect_queue, run preprocessing, and
      enqueue into model_queue
    - model workers run inference and set results on futures
    """

    num_feature_workers: int = 3
    num_model_workers: int = 2
    detect_queue_maxsize: int = 1024
    model_queue_maxsize: int = 1024

    detect_queue: queue.Queue[_QueuedItem] = field(init=False)
    model_queue: queue.Queue[_QueuedItem] = field(init=False)
    _threads_started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.detect_queue = queue.Queue(maxsize=self.detect_queue_maxsize)
        self.model_queue = queue.Queue(maxsize=self.model_queue_maxsize)
        self._start_workers()

    # ------------------------------------------------------------------
    # Worker orchestration
    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        if self._threads_started:
            return
        self._threads_started = True

        for i in range(self.num_feature_workers):
            t = threading.Thread(
                target=self._feature_worker,
                name=f"feature-worker-{i}",
                daemon=True,
            )
            t.start()

        for i in range(self.num_model_workers):
            t = threading.Thread(
                target=self._model_worker,
                name=f"model-worker-{i}",
                daemon=True,
            )
            t.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Ingress entry-point.

        This simulates the ingress-gateway behavior, which enqueues the
        request and waits on a future for the result, representing the
        aggregator pulling results from the model service.
        """
        future = ResultFuture()
        item = _QueuedItem(request=request, future=future)

        try:
            # Put into detect_queue; if queue is full, raise an error.
            self.detect_queue.put(item, timeout=1.0)
        except queue.Full as exc:  # pragma: no cover - rare path
            future.set_error(exc)

        # Wait for the pipeline to finish processing this request.
        # The timeout is generous but bounded to avoid deadlocks in tests.
        result = future.result(timeout=10.0)
        return result

    # ------------------------------------------------------------------
    # Internal workers
    # ------------------------------------------------------------------

    def _feature_worker(self) -> None:
        """Simulate feature service worker loop."""
        while True:
            try:
                item = self.detect_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            req = item.request
            future = item.future
            try:
                workload_id = req.get("workload_id", "unknown")
                image: SimulatedImage = req["payload"]
                req_id = str(req.get("id", "0"))

                seed_val = 0
                for ch in "feature" + workload_id + req_id:
                    seed_val = (seed_val * 31 + ord(ch)) & 0xFFFFFFFF
                rng = random.Random(seed_val)

                # Preprocessing and feature extraction work
                _ = busy_work(rng, iterations=120, payload=image.data)

                # Attach pseudo feature vector summary
                feature_score = rng.random()
                req["feature_score"] = float(feature_score)

                # Simulate push to model queue
                self.model_queue.put(item, timeout=1.0)

            except Exception as exc:  # pragma: no cover - robustness
                future.set_error(exc)
            finally:
                self.detect_queue.task_done()

    def _model_worker(self) -> None:
        """Simulate model service worker loop."""
        while True:
            try:
                item = self.model_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            req = item.request
            future = item.future

            try:
                workload_id = req.get("workload_id", "unknown")
                image: SimulatedImage = req["payload"]
                req_id = str(req.get("id", "0"))

                seed_val = 0
                for ch in "model" + workload_id + req_id:
                    seed_val = (seed_val * 31 + ord(ch)) & 0xFFFFFFFF
                rng = random.Random(seed_val)

                # Model inference work; slightly heavier than baseline per
                # request, but processed concurrently across workers.
                _ = busy_work(rng, iterations=260, payload=image.data)

                score = rng.random()
                label = "watermark_removed" if score > 0.5 else "no_issue"

                result = {
                    "id": req_id,
                    "score": float(score),
                    "label": label,
                    "workload_id": workload_id,
                }

                # Simulate audit and moderation sinks with lighter work
                _ = busy_work(rng, iterations=40)

                future.set_result(result)

            except Exception as exc:  # pragma: no cover - robustness
                future.set_error(exc)
            finally:
                self.model_queue.task_done()
