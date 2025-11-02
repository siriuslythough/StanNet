# Yoga Pose Classification using Fully Complex Convolution Neural Networks

### A EE604 Course Project

## Installation  

To install all the dependencies required for this project, use the `requirements.txt` file. Follow the steps below:  

1. Ensure you have Python 3.8 or above installed.  
2. Install `pip` if it is not already available.  
3. Run the following command in your terminal:  

```bash
pip install -r requirements.txt
```

* Ensure dataset are correctly loaded in the required structure before running the python script.
* We *strongly recommend* the following directory structure
  ```
  - main
  | -- utils.py
  | -- YogaPoses
  | -- venv (if using one)
  | -- main.py
  | -- ...
  ```
  where `...` of course represents the other `.py` files in this repo. Note the dataset folder name `YogaPoses`.

`Note:` Dataset is available [here [YogaPoses]](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification)

`main.py` is our main file. The file is sufficiently self-contained so far as the project is concerned. Ensure you have `YogaPoses [dataset] ` in the same directory as the `.py`.

You can run the code by using the following command.
```
  python main.py \
  --root YogaPoses \
  --arch resnet50 \
  --batch_size 8 \
  --epochs 50 \
  --lr 3e-4 \
  --modelName resnet50
```
`Note` - For detailed explanation of each argument you can refer to python script `main.py`

## Overview
This project builds a robust Yoga pose classification pipeline in PyTorch. It provides a clean training script, configurable architectures, and strong data augmentation to recognize multiple yoga postures from images. While you can use standard backbones like ResNet-50 as a baseline, the core contribution is experimenting with fully complex-valued convolutional neural networks to better capture phase-and-magnitude cues present in human pose textures and edges (as highlighted in the project title). Training, evaluation, and logging are all handled from a single entry script with simple CLI flags.

Copyright Notice
© 2024 Mohd Amir, Nishant Pandey & Tanmay Siddharth . All Rights Reserved.

This repository and its contents are created solely for academic purposes as part of EE604 coursework at IIT Kanpur. Unauthorized reproduction or use of this code for commercial or unethical purposes is strictly prohibited.

Authors:
* Mohd Amir | [mmamir22@iitk.ac.in](mmamir22@iitk.ac.in) (220660)
* Nishant Pandey | [nishantp22@iitk.ac.in](nishantp22@iitk.ac.in) (220724)
* Tanmay Siddharth | [tanmays22@iitk.ac.in](tanmays22@iitk.ac.in) (221129)

## Reference
[1] S. Yadav and K. R. Jerripothula, “FCCNs: Fully Complex-valued Convolutional Networks using Complex-valued Color Model and Loss Function,” *ICCV*, 2023, pp. 10689–10698. [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yadav_FCCNs_Fully_Complex-valued_Convolutional_Networks_using_Complex-valued_Color_Model_and_ICCV_2023_paper.pdf) · [OpenAccess page](https://openaccess.thecvf.com/content/ICCV2023/html/Yadav_FCCNs_Fully_Complex-valued_Convolutional_Networks_using_Complex-valued_Color_Model_and_ICCV_2023_paper.html)

