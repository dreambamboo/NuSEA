# NuSEA: Nuclei Segmentation with Ellipse Annotations
## IEEE Journal of Biomedical and Health Informatics (J-BHI) 2024 Accepted

> ### Authors' Preface
>>In this project, we provide the source codes, models, and model parameters referenced in this article. We have made all the details of the codes publicly available to ensure that reproducing the results of this article is straightforward. Of course, these codes represent just one way to implement the content of this article and have not been optimized or polished in any way. If you encounter any issues, please feel free to modify and debug as needed.
>>For convenience, we have made all data, including the training set, test set, and newly released datasets, available on [https://mcprl.com/html/dataset/NuSEA.html]. Thank you for your attention and use of this work. 

## Environments and Requirements
python 3.7.6  
pytorch 1.13.1+cu117  
numpy 1.21.6  
PIL 9.5.0  
torchvision 0.14.1+cu117  
visdom 0.1.8.9  
opencv-python (cv2) 0.4.2  
sklearn 1.0.2  
scipy 1.4.1  
logging 0.5.1.2  
## Implementation Steps
(1) Environments preparation.

(2) Download data from [https://mcprl.com/html/dataset/NuSEA.html] You may get the processed training sets, test sets and the newly released dataset NuSEA-dataset v1.0.

(3) Choose a dataset (MoNuSeg, CPM-17, or CoNSeP) you want to reproduce, and run the 'train.py' file in the corresponding dir. Before running, the modifications of the training set path in 'utils/dataset.py' and test set path in 'train.py' are essential.

(4) Optional. If you want to draw an ellipse by yourself, an annotation tool recommended here [https://github.com/Angelo-scut/Labelme-improved]

## Reference
Zhu Meng, et al. NuSEA: Nuclei Segmentation with Ellipse Annotations. IEEE Journal of Biomedical and Health Informatics (J-BHI). 2024.  

