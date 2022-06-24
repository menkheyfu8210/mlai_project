# Shallow-learning methods for pedestrian detection

This repository contains the code for the final project of the Machine Learning & Artificial Intelligence course of the Master's Degree in Computer Engineering for Robotics and Smart Industry @UniVR.

## Description

The project explores various shallow-learning techniques for pedestrian classification and detection in images.

## Getting Started

### Dependencies & requirements

The project has the following dependencies:

* [SciPy](https://www.scipy.org/)
* [scikit-image](https://scikit-image.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/)

The project and dataset require at least 6GB of free disk space.

### Dataset preparation

The dataset used in the project is the Daimler Pedestrian Detection Benchmark Dataset, which may be downloaded [here](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Mono_Ped__Detection_Be/daimler_mono_ped__detection_be.html). For use in this project, the dataset must be prepared:

* The pedestrian/non-pedestrian images must be collected respectively under Dataset/Pedestrians/ and Dataset/NonPedestrians/, where Dataset/ is a directory in the root directory of this project.
* A directory Dataset/NonPedestrianPatches/ must be created.
* Appropriately sized patches of the non-pedestrian images are to be extracted by executing:
```
python ./Project/methods/patch_extract.py
```

Note that the last script executed as-is produces around 40 thousand patches. Depending on memory availability this number can be changed by changing the max\_patches parameter in the extract\_patches\_2d function.

### Classifier benchmarks

TODO

### Detector

TODO

## Authors

Lorenzo Busellato - lorenzo.busellato@gmail.com
