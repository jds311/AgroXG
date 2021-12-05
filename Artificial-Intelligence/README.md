# AI Part of the Project

This directory consists the .ipynb files of different AI models along with their modified variants, their results and regarding information files.

## Table of Content

- [Introduction](#introduction)
- [Information Regarding Dataset](#dataset-information)
- [Approach](#approach)
  - [Models Used](#models)
    - [EfficientNet-B0](#efficientnet-b0)
    - [InceptionResNet-V2](#inceptionresnet-v2)
  - [GLCM Feature Extractor](#glcm-feature-extractor)
- [Results](#results)
- [Discussion Over Results](#discussion-over-results)
  - [A.) Effect of model architecture](#a-effect-of-model-architecture)
- [Key-Takeaways](#key-takeaways)
- [References](#references)
- [Contribution](#contribution)

## Introduction

At the first, we tried VGG-19 with fully connected bottom layers where we recieved ~91.14% accuracy. After literature review, we planned to implement 2 models for the same task and select the best one performing. 

## Dataset Information

<table>
  <tr>
    <td align="center">Healthy Wheat</td>
    <td align="center">Wheat Loose Smut</td>
    <td align="center">Crown and Root Rot</td>
    <td align="center">Leaf Rust</td>
  </tr>
  <tr>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/01541.jpeg" width=250></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/0021.jpg" width=250></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/03621.jpg" width=250></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/00071.jpg" width=250></td>
  </tr>
</table> 

## Approach

### Models

#### EfficientNet B0

<div align="center">
  <img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/block-diagram_EffNetB0.png" width=900 />
  <p> Block Diagram - EfficientNet-B0 </p>
</div>

[Implemented Model Structure Diagram](https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/Model_EffNetB0.png)


#### InceptionResNet V2

<div align="center">
  <img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/block-diagram_inceptionResnetV2.png" width=900 />
  <p> Block Diagram - InceptionResNet-V2 </p>
</div>

[Implemented Model Structure Diagram](https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/Model_EffNetB0.png)

## GLCM Feature Extractor

<div align="center">
  <img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/GLCM-features.jpg" width=500 />
  <p> GLCM - Features Formulas </p>
</div>


## Implementation 

<div align="center">
  <img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/Proposed-Model.jpeg" width=900 />
  <p> Proposed Model </p>
</div>

## Results

<table>
  <tr>
    <td></td>
    <td align="center">EfficientNet-B0</td>
    <td align="center">VGG19</td>
    <td align="center">InceptionResNet-V2</td>
  </tr>
  <tr>
    <td align="center" width=125> Accuracy Plots </td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/LossPlot_EffNetB0-Part1.png" width=350></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/LossPlot_VGG19.png" width=350></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/LossPlot_InceptionResNetV2.png" width=350></td>
  </tr>
 
  <tr>
    <td align="center" width=125 > Loss Plots </td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/AccPlot_EffNetB0-Part1.png" width=350></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/AccPlot_VGG19.png" width=350></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/AccPlot_InceptionResNetV2.png" width=350></td>
  </tr>
  
  
  <tr>
    <td align="center"  width=125> Categorical Accuracy Table </td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/CategoricalAccScoreTable.jpg" width=350></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/CategoricalAccScoreTable-VGG.jpg" width=350></td>
    <td><img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/CategoricalAccScoreTable-IncResv2.jpg" width=350></td>
  </tr>
</table>


<div align="center">
  <img src="https://github.com/caped-crusader16/AgroXG/blob/main/Artificial-Intelligence/ExtractedFeatures.jpg" width=400 />
  <p> Features Extracted using GLCM</p>
</div>


## Discussion over Results

### A.) Effect of model architecture

## Key Takeaways

## References
-  S. Hassan, A. Maji, M. Jasi ́nski, Z. Leonowicz and E. Jasi ́nska,”Identification of Plant-Leaf Diseases Using CNN and Transfer-Learning Approach”,  Electronics,  vol.  10,  no.  12, p. 1388, 2021. Available: [Link](10.3390/electronics10121388).
- S. Desai, “Neural Artistic Style Transfer: A Comprehensive Look,” Medium, 14-Sep-2017. [Online]. Available: [Link](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199).
- [Deep Learning-Based Classification of Fruit Diseases: An Application for Precision Agriculture](https://www.researchgate.net/publication/344478535_Deep_Learning-based_Classification_of_Fruit_Diseases_An_Application_for_Precision_Agriculture)
- [Statistical Texture Measures Computed from Gray Level Coocurrence Matrices](https://www.uio.no/studier/emner/matnat/ifi/INF4300/h08/undervisningsmateriale/glcm.pdf)
- [Learning Feature Fusion in Deep Learning-Based Object Detector](https://www.hindawi.com/journals/je/2020/7286187/)

- [Cattle Race Classification Using Gray Level Co-occurrence Matrix Convolutional Neural Networks](https://www.researchgate.net/publication/282856577_Cattle_Race_Classification_Using_Gray_Level_Co-occurrence_Matrix_Convolutional_Neural_Networks)

- [The Real-World-Weight Cross-Entropy Loss Function: Modeling the Costs of Mislabeling](https://ieeexplore.ieee.org/document/8943952)

* * *

## Contribution

> [Manav Vagrecha](https://github.com/caped-crusader16), [Bhumiti Gohel](https://github.com/bhumiti28), [Vatsal Patel](https://github.com/pvatshal), [Jeet Shah](https://github.com/jds311)

## Support

Drop a ⭐ if you like our work. Contact any of us if you wish to have any suggestions or edits.


