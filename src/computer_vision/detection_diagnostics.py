"""
Face detection diagnostics (Phase 2B audit instrumentation).

Purpose
-------
Measure, at runtime, the detected face bounding-box size and detection
confidence so the real-world "it fails when I stand 2-3 m away" problem can be
localised to a specific stage of the pipeline.

Design constraints
------------------
- **Zero impact on normal behaviour.** Disabled by default
  (`FACE_DIAGNOSTICS_ENABLED=false`). When disabled, `record()` is a single
  boolean check and returns immediately.
- **No change to detection, recognition, or liveness.** This module only reads
  values that already exist and reports them.
- **Optional CSV export** for measurements taken across real distances.
"""

import csv
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionDiagnostics:
    """
    Collects and reports face-detection measurements.

    Usage:
        diag = DetectionDiagnostics()
        diag.record(frame_shape, detected_faces, stage="detection")
        diag.overlay(frame, detected_faces)   # optional, draws annotations
        print(diag.summary())
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        log_every: Optional[int] = None,
        warn_face_px: Optional[int] = None,
        csv_path: Optional[str] = None,
        draw_overlay: Optional[bool] = None,
    ):
        """
        Initialize the diagnostics collector.

        Args:
            enabled: Override `config.FACE_DIAGNOSTICS_ENABLED`.
            log_every: Override `config.FACE_DIAGNOSTICS_LOG_EVERY`.
            warn_face_px: Override `config.FACE_DIAGNOSTICS_WARN_FACE_PX`.
            csv_path: Override `config.FACE_DIAGNOSTICS_CSV_PATH`.
            draw_overlay: Override `config.FACE_DIAGNOSTICS_DRAW_OVERLAY`.
        """
        self.enabled = config.FACE_DIAGNOSTICS_ENABLED if enabled is None else bool(enabled)
        self.log_every = config.FACE_DIAGNOSTICS_LOG_EVERY if log_every is None else int(log_every)
        self.warn_face_px = (
            config.FACE_DIAGNOSTICS_WARN_FACE_PX if warn_face_px is None else int(warn_face_px)
        )
        self.csv_path = config.FACE_DIAGNOSTICS_CSV_PATH if csv_path is None else csv_path
        self.draw_overlay_enabled = (
            config.FACE_DIAGNOSTICS_DRAW_OVERLAY if draw_overlay is None else bool(draw_overlay)
        )

        self.frame_count = 0
        self.detected_frames = 0
        self.missed_frames = 0
        self.total_faces = 0
        self.sizes: List[int] = []
        self.confidences: List[float] = []

        # Downstream-stage counters.
        self.stage_counts: Dict[str, int] = {}
        self.stage_success: Dict[str, int] = {}
        self.embedding_failures = 0
        self.ear_failures = 0
        self.ear_values: List[float] = []
        self.recognition_distances: List[float] = []

        self._lock = threading.Lock()
        self._csv_file = None
        self._csv_writer = None

        if self.enabled:
            logger.info(
                "Face detection diagnostics ENABLED "
                f"(log_every={self.log_every}, warn_face_px={self.warn_face_px}, "
                f"csv={'yes' if self.csv_path else 'no'})"
            )
            self._init_csv()

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        """Open the CSV file and write its header, if configured."""
        if not self.csv_path:
            return
        try:
            path = Path(self.csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists()
            self._csv_file = path.open("a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            if new_file:
                self._csv_writer.writerow([
                    "timestamp", "stage", "frame_w", "frame_h",
                    "faces_detected", "bbox_w", "bbox_h", "confidence",
                ])
                self._csv_file.flush()
        except Exception as exc:
            logger.error(f"Could not open diagnostics CSV {self.csv_path}: {exc}")
            self._csv_file = None
            self._csv_writer = None

    def _write_csv(self, stage: str, frame_shape: Sequence[int],
                   faces: Sequence[object]) -> None:
        """Append one row per detected face (or one empty-detection row)."""
        if self._csv_writer is None:
            return
        try:
            height, width = frame_shape[0], frame_shape[1]
            ts = time.time()
            if not faces:
                self._csv_writer.writerow([ts, stage, width, height, 0, "", "", ""])
            else:
                for face in faces:
                    bx, by, bw, bh = face.bbox
                    self._csv_writer.writerow([
                        ts, stage, width, height, len(faces), bw, bh,
                        f"{float(face.confidence):.4f}",
                    ])
            self._csv_file.flush()
        except Exception as exc:
            logger.debug(f"Diagnostics CSV write failed: {exc}")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        frame_shape: Sequence[int],
        faces: Sequence[object],
        stage: str = "detection",
    ) -> None:
        """
        Record the outcome of one frame.

        Args:
            frame_shape: `frame.shape` of the analysed frame.
            faces: Sequence of `DetectedFace` (may be empty).
            stage: Label such as "detection" or "liveness".
        """
        if not self.enabled:
            return

        try:
            height = int(frame_shape[0])
            width = int(frame_shape[1])

            with self._lock:
                self.frame_count += 1
                if faces:
                    self.detected_frames += 1
                    self.total_faces += len(faces)
                    for face in faces:
                        bw, bh = face.bbox[2], face.bbox[3]
                        longest = max(bw, bh)
                        self.sizes.append(longest)
                        self.confidences.append(float(face.confidence))
                else:
                    self.missed_frames += 1

                count = self.frame_count
                should_log = self.log_every <= 1 or count % self.log_every == 0
                snapshot = self._snapshot_locked()

            self._write_csv(stage, (height, width), faces)

            if should_log:
                logger.info(self.format_line(snapshot, height, width, count, stage))

            if faces and snapshot["max_face_px"] < self.warn_face_px:
                logger.warning(
                    f"Face detected but small: {snapshot['max_face_px']}px "
                    f"(below {self.warn_face_px}px). Detection quality degrades below "
                    "~100px for this detector - the subject is probably too far away."
                )
        except Exception as exc:
            # Diagnostics must never break the pipeline.
            logger.debug(f"Diagnostics record failed: {exc}")

    # ------------------------------------------------------------------
    # Per-stage measurements
    # ------------------------------------------------------------------

    def record_stage(
        self,
        stage: str,
        bbox: Sequence[int],
        detection_confidence: Optional[float] = None,
        embedding_ok: Optional[bool] = None,
        recognized: Optional[bool] = None,
        distance: Optional[float] = None,
        ear: Optional[float] = None,
    ) -> None:
        """
        Record a downstream-stage outcome for one already-detected face.

        These counters are what separate "the detector missed the face" from
        "the face was detected but recognition or liveness dropped it".

        Args:
            stage: "recognition" or "liveness".
            bbox: The detected bounding box.
            detection_confidence: Detector confidence for this face.
            embedding_ok: Whether an embedding could be generated.
            recognized: Whether the identity matched.
            distance: Recognition distance when matched.
            ear: Eye Aspect Ratio when available (None = extraction failed).
        """
        if not self.enabled:
            return

        try:
            with self._lock:
                self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
                if recognized:
                    self.stage_success[stage] = self.stage_success.get(stage, 0) + 1
                    if distance is not None:
                        self.recognition_distances.append(float(distance))
                if embedding_ok is False:
                    self.embedding_failures += 1
                if stage == "liveness" and ear is None:
                    self.ear_failures += 1
                elif ear is not None:
                    self.ear_values.append(float(ear))
        except Exception as exc:
            logger.debug(f"Diagnostics record_stage failed: {exc}")

    def stage_report(self) -> Dict[str, object]:
        """
        Return per-stage counters.

        Useful for deciding whether the bottleneck is detection, recognition,
        or liveness at a given distance.
        """
        with self._lock:
            report: Dict[str, object] = {}
            for stage, count in self.stage_counts.items():
                success = self.stage_success.get(stage, 0)
                report[stage] = {
                    "processed": count,
                    "succeeded": success,
                    "success_rate": (success / count) if count else 0.0,
                }
            report["embedding_failures"] = self.embedding_failures
            report["ear_extraction_failures"] = self.ear_failures
            report["ear_samples"] = len(self.ear_values)
            if self.recognition_distances:
                report["avg_recognition_distance"] = (
                    sum(self.recognition_distances) / len(self.recognition_distances)
                )
            if self.ear_values:
                report["avg_ear"] = sum(self.ear_values) / len(self.ear_values)
            return report

    def _snapshot_locked(self) -> Dict[str, float]:
        """Compute aggregate stats. Caller must hold the lock."""
        avg_conf = sum(self.confidences) / len(self.confidences) if self.confidences else float("nan")
        return {
            "frames": self.frame_count,
            "detected_frames": self.detected_frames,
            "missed_frames": self.missed_frames,
            "detect_rate": (self.detected_frames / self.frame_count) if self.frame_count else 0.0,
            "avg_face_px": (sum(self.sizes) / len(self.sizes)) if self.sizes else float("nan"),
            "max_face_px": max(self.sizes) if self.sizes else 0,
            "min_face_px": min(self.sizes) if self.sizes else 0,
            "avg_confidence": avg_conf,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def format_line(snapshot: Dict[str, float], frame_h: int, frame_w: int,
                    frame_count: int, stage: str) -> str:
        """Format one compact log line."""
        return (
            f"[face-diag] frame={frame_count} stage={stage} "
            f"frame_size={frame_w}x{frame_h} "
            f"detect_rate={snapshot['detect_rate']:.0%} "
            f"faces={snapshot['detected_frames']}/{snapshot['frames']} "
            f"face_px(avg/min/max)={snapshot['avg_face_px']:.0f}/"
            f"{snapshot['min_face_px']}/{snapshot['max_face_px']} "
            f"conf(avg)={snapshot['avg_confidence']:.3f}"
        )

    def summary(self) -> Dict[str, float]:
        """Return the aggregate statistics collected so far."""
        with self._lock:
            return self._snapshot_locked()

    def report(self) -> str:
        """Return a human-readable multi-line summary."""
        s = self.summary()
        lines = [
            "Face Detection Diagnostics",
            "----------------------------",
            f"frames analysed      : {s['frames']}",
            f"frames with a face   : {s['detected_frames']}",
            f"frames without a face: {s['missed_frames']}",
            f"detection rate       : {s['detect_rate']:.1%}",
            f"face size avg/min/max: {s['avg_face_px']:.0f} / {s['min_face_px']} / {s['max_face_px']} px",
            f"avg confidence       : {s['avg_confidence']:.3f}",
            f"warn threshold       : {self.warn_face_px} px",
        ]
        if self.sizes:
            ordered = sorted(self.sizes)
            p10 = ordered[max(0, int(len(ordered) * 0.10) - 1)]
            p50 = ordered[len(ordered) // 2]
            lines.append(f"face size p10/p50    : {p10} / {p50} px")

        stages = self.stage_report()
        if stages:
            lines.append("")
            lines.append("Downstream stages:")
            for stage in ("recognition", "liveness"):
                if stage in stages:
                    info = stages[stage]
                    lines.append(
                        f"  {stage:<12} processed={info['processed']} "
                        f"succeeded={info['succeeded']} "
                        f"({info['success_rate']:.0%})"
                    )
            if stages.get("embedding_failures"):
                lines.append(f"  embedding failures: {stages['embedding_failures']}")
            if stages.get("ear_extraction_failures"):
                lines.append(
                    f"  EAR extraction fails: {stages['ear_extraction_failures']} "
                    f"/ {stages['ear_samples'] + stages['ear_extraction_failures']} samples"
                )
            if "avg_recognition_distance" in stages:
                lines.append(
                    f"  avg match distance: {stages['avg_recognition_distance']:.3f}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    def overlay(self, frame, faces: Sequence[object]):
        """
        Draw a compact annotation on a copy of the frame.

        Returns the annotated frame (a copy; the input is never modified).
        Does nothing when disabled or when drawing is turned off.
        """
        if not self.enabled or not self.draw_overlay_enabled:
            return frame
        try:
            import cv2
        except ImportError:
            return frame

        try:
            out = frame.copy()
            for face in faces:
                x, y, w, h = face.bbox
                conf = float(face.confidence)
                longest = max(w, h)

                if longest < self.warn_face_px:
                    color = (0, 165, 255)   # orange: detected, but small
                else:
                    color = (0, 200, 0)     # green: healthy size

                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    out, f"{longest}px c={conf:.2f}",
                    (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

            s = self.summary()
            banner = (
                f"DIAG frames={s['frames']} rate={s['detect_rate']:.0%} "
                f"avg_px={s['avg_face_px']:.0f} conf={s['avg_confidence']:.2f}"
            )
            cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(
                out, banner, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            return out
        except Exception as exc:
            logger.debug(f"Diagnostics overlay failed: {exc}")
            return frame

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close the CSV file. Safe to call multiple times."""
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception as exc:
                logger.debug(f"Diagnostics CSV close failed: {exc}")
            finally:
                self._csv_file = None
                self._csv_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Shared instance for convenient use from pages/services.
_shared: Optional[DetectionDiagnostics] = None


def get_diagnostics() -> DetectionDiagnostics:
    """Return the process-wide diagnostics instance, creating it on first use."""
    global _shared
    if _shared is None:
        _shared = DetectionDiagnostics()
    return _shared