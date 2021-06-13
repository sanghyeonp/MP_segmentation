# Microplastics Segmentation from Fluorescence Microscopy Images for Marine Pollution Monitoring: An Investigation of Deep Convolutional Neural Networks and Test-time Augmentation


## Sanghyeon Park


*This is bachelor dissertation conducted at Ghent University Global Campus (GUGC).*


Counsellors: Ho-min Park, Maria Krishna de Guzman\
Supervisors: Prof. Dr. Wesley De Neve, Prof. Dr. Arnout Van Messem, Prof. Dr. Tanja Cirkovic Velickovic

## Summary
Lack of standardized microplastic (MP) quantification method makes it difficult to compare between related
studies and establish MP monitoring and risk assessment policies applicable worldwide. MP-VAT and MPACT
have been proposed for automatic quantification of MP from a microscopy image. However, these
tools possess drawbacks in either limited accuracy or not being fully automatic. In this bachelor project,
deep learning approach has been utilized to tackle the limitation of the existing tools and make further
improvement through implementation of test-time augmentation (TTA). A convolutional neural network
(CNN) model, specifically U-Net, has been investigated. The influence in the model’s performance with
different image augmentation such as brightness, contrast, and hue, saturation, and value (HSV), has
been investigated as well. In addition, the effect on model by individual and combination of different image
augmentation have also been explored through ablation study. U-Net with SoftDice loss and SGD optimizer
can generate a more precise masks compared to MP-VAT1.0 with increased performance in terms of mean
precision, mean F1-score, and mean IoU by 0.352, 0.162, and 0.188, respectively. Also, the percentage
of over-estimation in MP count has decreased by 68.4%. Implementation of TTA causes a general decline
in the performances of the U-Net. However, the ablation study of image augmentation shows possibility
of performance improvement if a combination of brightness and contrast is utilized. U-Net has potential to
contribute in the establishment of standard method for MP quantification by generating a more precise and
accurate binary mask.


## Keywords
Augmentation, Deep Learning, Environmental Monitoring, Microplastic Quantification, Image Segmentation


## Reference

* src.functional.loss.py is obtained from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch.

* U-Net architectures and the evaluation metrics are obtained from https://github.com/qubvel/segmentation_models.pytorch. (A modification was done for Accuracy object in metrics.py to allow computing balanced accuracy.)

* Nested U-Net architecture is obtained from https://github.com/4uiiurz1/pytorch-nested-unet.

## License
[MIT](https://choosealicense.com/licenses/mit/)
