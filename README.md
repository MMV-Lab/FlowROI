# FlowROI
ROI based compression algorithm for ComplexEye array microscope

Code will be released upon paper publication.

# FlowROI: Optical-Flow-Based ROI Extraction

FlowROI extracts motion-based regions of interest (ROIs) from image sequences. It estimates optical flow between neighboring frames, converts the flow into a motion saliency map, thresholds the most salient pixels, and cleans the result with morphology and connected-component filtering.

## Main features

- CPU backends: DIS optical flow and Farnebäck optical flow
- Optional CUDA Farnebäck / TV-L1 backends if OpenCV was built with CUDA
- Translation-only stabilization using phase correlation
- Fast histogram-based percentile thresholding
- Morphological cleanup and small-component removal
- Optional debug outputs for registered frames, saliency maps, and raw masks

## Installation

```bash
pip install opencv-python numpy
```

For CUDA backends, install an OpenCV build with CUDA support.

## Usage

```bash
python flow_roi_extractor.py \
  --input-dir ./dataset \
  --output-dir ./flow_output \
  --pattern "*_data.bmp" \
  --step 1 \
  --backend dis \
  --keep-top 0.20
```

The output directory will contain one `*_roi.png` binary mask for each input frame.

## Useful arguments

- `--backend`: `dis`, `farneback`, `cuda_farneback`, or `cuda_tvl1`
- `--keep-top`: fraction of highest-saliency pixels kept as foreground, e.g. `0.20`
- `--step`: temporal gap between paired frames
- `--min-area-bin`: minimum component size after initial thresholding
- `--min-area-clean`: minimum component size in final mask
- `--save-debug`: save intermediate images
