# cloudseg

Cloudseg is a repository for training and evaluating various cloud segmentation algorithms on standard cloud datasets.

----------------------------------------------------------------
## Datasets

Currently cloudseg supports the 95-Cloud and 38-Cloud datasets [1,2].

----------------------------------------------------------------
## Models

WIP

----------------------------------------------------------------
## Installation
1. Install python dependencies:

        conda install --file requirements.txt

2. Download and link the data sets:
    - Follow the [38-Cloud instructions](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) to download the dataset and set up the directory tree.
    - Follow the [95-Cloud instructions](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset) to add the 95-Cloud data to the 38-Cloud dataset.
    - Create a symbolic link from `data/95cloud` to the directory with the combined 95-Cloud dataset:

            cd <path to this repository>
            ln -s <path to 95-Cloud> data/95cloud

    - Preprocess the 95-Cloud dataset (NOT optional) to speed up data loading:

            python src/scripts/preprocess_95cloud.py
3. WIP

----------------------------------------------------------------
[1] S. Mohajerani, T. A. Krammer and P. Saeedi, "A Cloud Detection Algorithm for Remote Sensing Images Using Fully Convolutional Neural Networks," 2018 IEEE 20th International Workshop on Multimedia Signal Processing (MMSP), Vancouver, BC, 2018, pp. 1-5. doi: 10.1109/MMSP.2018.8547095 URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8547095&isnumber=8547039

[2] S. Mohajerani and P. Saeedi, "Cloud-Net: An End-To-End Cloud Detection Algorithm for Landsat 8 Imagery," IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium, Yokohama, Japan, 2019, pp. 1029-1032. doi: 10.1109/IGARSS.2019.8898776. Arxive URL: https://arxiv.org/pdf/1901.10077.pdf, IEEE URL: URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8898776&isnumber=8897702
