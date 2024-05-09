#Semantic Segmentation of Street View Images
This project utilizes the MMSegmentation library from OpenMMlab for semantic segmentation on street view images. We use the Cityscapes dataset and the Segformer network.

#Project Structure
checkpoints/: Contains model weight files.
configs/: Contains model configuration files.
sv_img/: Stores original street view images.
sv_img_seg/: Stores images post-semantic segmentation.
seg_result_statistics.csv: Records the proportion of different land cover categories.
segformer_segmentation_code.py: Demo code for semantic segmentation
