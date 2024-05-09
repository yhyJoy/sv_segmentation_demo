# Street View Image Semantic Segmentation

## Overview
This repository demonstrates the use of the MMSegmentation library provided by OpenMMlab for semantic segmentation on street view images. We employ the Cityscapes dataset and utilize the Segformer network for effective segmentation. For detailed configuration procedures of the MMSegmentation, refer to the [MMSegmentation Setup Guide](https://mmsegmentation.readthedocs.io/en/main/get_started.html).

## Project Structure

- **`checkpoints/`**: Contains model weight files.
- **`configs/`**: Contains model configuration files.
- **`sv_img/`**: Stores the original street view images.
- **`sv_img_seg/`**: Stores images after semantic segmentation.
- **`seg_result_statistics.csv`**: Records the proportion of different land cover categories.
- **`segformer_segmentation_code.py`**: Python script for semantic segmentation demo.

## Usage

To run the segmentation model, use:
```bash
python segformer_segmentation_code.py
