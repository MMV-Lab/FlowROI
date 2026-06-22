"""FlowROI: fast motion-based ROI extraction from image sequences.

This module estimates motion between neighboring frames using optical flow and
converts the motion saliency map into a binary ROI mask. It is designed for
pre-selecting regions of interest before more expensive downstream processing
such as cell segmentation.

Example
-------
Process all images matching ``*_data.bmp`` in a folder::

    python flow_roi_extractor.py \
        --input-dir ./dataset \
        --output-dir ./flow_output \
        --pattern "*_data.bmp" \
        --step 1 \
        --keep-top 0.20

Input images are sorted by the last integer found in the filename. For each
frame, the script compares it with a neighboring frame and writes a binary
``*_roi.png`` mask to the output directory.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np

Backend = Literal["dis", "farneback", "cuda_farneback", "cuda_tvl1"]


@dataclass(frozen=True)
class FlowROIConfig:
    """Configuration for motion-based ROI extraction."""

    backend: Backend = "dis"
    keep_top: float = 0.20
    min_area_bin: int = 40
    min_area_clean: int = 200
    open_kernel: tuple[int, int] = (3, 3)
    close_kernel: tuple[int, int] = (5, 5)
    mix_gradient: bool = True
    alpha: float = 0.70
    stabilization_skip_eps: float = 0.25
    histogram_bins: int = 256


FARNEBACK_CONFIG = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=11,
    iterations=5,
    poly_n=5,
    poly_sigma=1.2,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
)

DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_FAST

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 1) - 1))
except Exception:
    pass


def read_gray(path: Path) -> np.ndarray:
    """Read an image as grayscale and return a C-contiguous uint8/float32 array."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    if image.dtype != np.float32:
        image = image.astype(np.float32)
    return np.ascontiguousarray(image)


def normalize_01(array: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    min_value = float(array.min())
    max_value = float(array.max())
    return (array - min_value) / (max_value - min_value + 1e-8)


def percentile_threshold_01(
    saliency: np.ndarray,
    keep_top: float,
    bins: int = 256,
) -> float:
    """Fast percentile threshold for saliency maps in [0, 1].

    ``keep_top=0.20`` means pixels in the top 20% saliency range are selected.
    """
    if not 0.0 < keep_top < 1.0:
        raise ValueError("keep_top must be in (0, 1).")

    hist = cv2.calcHist(
        [saliency.astype(np.float32)],
        [0],
        None,
        [bins],
        [0, 1],
    ).ravel()
    cdf = np.cumsum(hist)
    cutoff = (1.0 - keep_top) * cdf[-1]
    index = int(np.searchsorted(cdf, cutoff, side="left"))
    index = int(np.clip(index, 0, bins - 1))
    return float((index + 0.5) / bins)


class Stabilizer:
    """Translation-only image stabilization using phase correlation."""

    def __init__(self) -> None:
        self._window: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    def _ensure_window(self, shape: tuple[int, int]) -> None:
        if self._shape != shape:
            height, width = shape
            self._window = cv2.createHanningWindow((width, height), cv2.CV_32F)
            self._shape = shape

    def stabilize(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        skip_eps: float = 0.25,
    ) -> tuple[np.ndarray, float, float]:
        """Register ``current`` to ``previous`` and return registered image plus shift."""
        height, width = current.shape
        self._ensure_window((height, width))

        assert self._window is not None
        previous_f = previous.astype(np.float32, copy=False)
        current_f = current.astype(np.float32, copy=False)
        (dx, dy), _ = cv2.phaseCorrelate(previous_f, current_f, self._window)

        if abs(dx) + abs(dy) <= skip_eps:
            return current, 0.0, 0.0

        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        registered = cv2.warpAffine(
            current,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return registered, float(dx), float(dy)


class OpticalFlowEstimator:
    """Factory and wrapper for CPU/GPU optical-flow backends."""

    def __init__(self, backend: Backend = "dis") -> None:
        self.backend = backend
        self._dis = None
        self._cuda_farneback = None
        self._cuda_tvl1 = None

    @staticmethod
    def has_cuda() -> bool:
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    def _get_dis(self):
        if self._dis is None:
            self._dis = cv2.DISOpticalFlow_create(DIS_PRESET)
        return self._dis

    def _get_cuda_farneback(self):
        if self._cuda_farneback is None:
            self._cuda_farneback = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=4,
                pyrScale=0.5,
                fastPyramids=False,
                winSize=11,
                numIters=5,
                polyN=7,
                polySigma=1.5,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            )
        return self._cuda_farneback

    def _get_cuda_tvl1(self):
        if self._cuda_tvl1 is None:
            self._cuda_tvl1 = cv2.cuda_OpticalFlowDual_TVL1.create()
        return self._cuda_tvl1

    def estimate(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Estimate dense optical flow with shape ``(H, W, 2)``."""
        if self.backend == "dis":
            return self._get_dis().calc(previous, current, None)

        if self.backend == "farneback":
            return cv2.calcOpticalFlowFarneback(previous, current, None, **FARNEBACK_CONFIG)

        if self.backend.startswith("cuda"):
            if not self.has_cuda():
                raise RuntimeError("CUDA backend requested, but no CUDA device is available.")

            gpu_previous = cv2.cuda_GpuMat()
            gpu_current = cv2.cuda_GpuMat()
            gpu_previous.upload(previous)
            gpu_current.upload(current)

            if self.backend == "cuda_farneback":
                gpu_flow = self._get_cuda_farneback().calc(gpu_previous, gpu_current, None)
            elif self.backend == "cuda_tvl1":
                gpu_flow = self._get_cuda_tvl1().calc(gpu_previous, gpu_current, None)
            else:
                raise ValueError(f"Unknown CUDA backend: {self.backend}")
            return gpu_flow.download()

        raise ValueError(f"Unknown optical-flow backend: {self.backend}")


def motion_saliency(flow: np.ndarray, mix_gradient: bool = True, alpha: float = 0.70) -> np.ndarray:
    """Convert optical flow into a normalized motion saliency map."""
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = cv2.magnitude(u, v)

    if not mix_gradient:
        return normalize_01(magnitude)

    grad_x = cv2.Scharr(magnitude, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(magnitude, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(grad_x, grad_y)

    saliency = alpha * normalize_01(magnitude) + (1.0 - alpha) * normalize_01(gradient)
    return normalize_01(saliency)


class MaskCleaner:
    """Binarization and morphology-based mask cleanup."""

    def __init__(self, open_kernel: tuple[int, int], close_kernel: tuple[int, int]) -> None:
        self.open_kernel = np.ones(open_kernel, np.uint8) if open_kernel else None
        self.close_kernel = np.ones(close_kernel, np.uint8) if close_kernel else None

    @staticmethod
    def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        """Remove connected components smaller than ``min_area``."""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        output = np.zeros_like(mask, dtype=np.uint8)
        for index in range(1, num):
            if stats[index, cv2.CC_STAT_AREA] >= min_area:
                output[labels == index] = 1
        return output

    def binarize_and_clean(
        self,
        saliency: np.ndarray,
        keep_top: float,
        min_area: int,
        histogram_bins: int,
    ) -> np.ndarray:
        """Threshold saliency and apply morphology/area filtering."""
        threshold = percentile_threshold_01(saliency, keep_top, histogram_bins)
        mask = (saliency >= threshold).astype(np.uint8)

        if self.open_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        if self.close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        return self.remove_small_components(mask, min_area)


class FlowROIExtractor:
    """High-level motion ROI extractor for image pairs."""

    def __init__(self, config: FlowROIConfig) -> None:
        self.config = config
        self.stabilizer = Stabilizer()
        self.flow_estimator = OpticalFlowEstimator(config.backend)
        self.cleaner = MaskCleaner(config.open_kernel, config.close_kernel)

    def process_pair(
        self,
        previous_path: Path,
        current_path: Path,
        output_path: Path | None = None,
        save_debug: bool = False,
        debug_dir: Path | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Process one image pair and optionally save the ROI mask.

        Returns a dictionary with ``mask``, ``saliency``, ``flow``, shifts ``dx``/``dy``,
        and processing time in seconds.
        """
        start = time.perf_counter()

        previous = read_gray(previous_path)
        current = read_gray(current_path)
        registered, dx, dy = self.stabilizer.stabilize(
            previous,
            current,
            self.config.stabilization_skip_eps,
        )
        flow = self.flow_estimator.estimate(previous, registered)
        saliency = motion_saliency(flow, self.config.mix_gradient, self.config.alpha)
        raw_mask = self.cleaner.binarize_and_clean(
            saliency,
            self.config.keep_top,
            self.config.min_area_bin,
            self.config.histogram_bins,
        )
        mask = self.cleaner.remove_small_components(raw_mask, self.config.min_area_clean)

        elapsed = time.perf_counter() - start

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), mask * 255)

        if save_debug:
            if debug_dir is None:
                debug_dir = output_path.parent if output_path is not None else Path("debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            stem = current_path.stem
            cv2.imwrite(str(debug_dir / f"{stem}_registered.png"), registered)
            cv2.imwrite(str(debug_dir / f"{stem}_saliency.png"), (saliency * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / f"{stem}_raw_mask.png"), raw_mask * 255)
            cv2.imwrite(str(debug_dir / f"{stem}_roi.png"), mask * 255)

        return {
            "mask": mask,
            "raw_mask": raw_mask,
            "saliency": saliency,
            "flow": flow,
            "dx": dx,
            "dy": dy,
            "time_seconds": elapsed,
        }


def extract_last_number(path: Path) -> int:
    """Return the last integer in a filename, or -1 if no integer is found."""
    numbers = re.findall(r"\d+", path.name)
    return int(numbers[-1]) if numbers else -1


def list_images(input_dir: Path, pattern: str) -> list[Path]:
    """List and naturally sort image paths using the last integer in the filename."""
    paths = sorted(input_dir.glob(pattern), key=lambda p: (p.parent.as_posix(), extract_last_number(p), p.name))
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir} with pattern {pattern!r}.")
    return paths


def iter_frame_pairs(paths: list[Path], step: int) -> Iterable[tuple[Path, Path]]:
    """Yield ``(previous, current)`` frame pairs.

    For most frames, ``current`` is compared with the frame ``step`` positions
    ahead. For the final frames, it is compared with a previous frame so every
    image can still receive an ROI mask.
    """
    if step < 1:
        raise ValueError("step must be >= 1.")
    if len(paths) <= step:
        raise ValueError("The number of images must be larger than step.")

    for index in range(len(paths) - step):
        yield paths[index + step], paths[index]

    for index in range(len(paths) - step, len(paths)):
        yield paths[index - step], paths[index]


def process_sequence(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    step: int,
    config: FlowROIConfig,
    save_debug: bool = False,
) -> None:
    """Process a sequence of images and save one ROI mask per frame."""
    paths = list_images(input_dir, pattern)
    extractor = FlowROIExtractor(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug" if save_debug else None

    times: list[float] = []
    for previous_path, current_path in iter_frame_pairs(paths, step):
        output_path = output_dir / f"{current_path.stem}_roi.png"
        result = extractor.process_pair(
            previous_path,
            current_path,
            output_path=output_path,
            save_debug=save_debug,
            debug_dir=debug_dir,
        )
        times.append(float(result["time_seconds"]))
        print(f"Saved {output_path}")

    mean_time = float(np.mean(times)) if times else 0.0
    print(f"Processed {len(times)} frames. Mean time: {mean_time:.4f} s/frame")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast optical-flow-based ROI extraction.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output ROI masks.")
    parser.add_argument("--pattern", default="*data.bmp", help="Glob pattern for input images.")
    parser.add_argument("--step", type=int, default=1, help="Temporal gap between paired frames.")
    parser.add_argument(
        "--backend",
        choices=["dis", "farneback", "cuda_farneback", "cuda_tvl1"],
        default="dis",
        help="Optical-flow backend.",
    )
    parser.add_argument("--keep-top", type=float, default=0.20, help="Top saliency fraction kept as foreground.")
    parser.add_argument("--min-area-bin", type=int, default=40, help="Minimum component area after binarization.")
    parser.add_argument("--min-area-clean", type=int, default=200, help="Minimum component area in final mask.")
    parser.add_argument("--no-gradient", action="store_true", help="Use flow magnitude only, without gradient fusion.")
    parser.add_argument("--alpha", type=float, default=0.70, help="Weight of flow magnitude when gradient fusion is enabled.")
    parser.add_argument("--save-debug", action="store_true", help="Save intermediate registered/saliency/mask images.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = FlowROIConfig(
        backend=args.backend,
        keep_top=args.keep_top,
        min_area_bin=args.min_area_bin,
        min_area_clean=args.min_area_clean,
        mix_gradient=not args.no_gradient,
        alpha=args.alpha,
    )
    process_sequence(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        step=args.step,
        config=config,
        save_debug=args.save_debug,
    )


if __name__ == "__main__":
    main()
