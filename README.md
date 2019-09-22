
# Great Ape Detection

## Introduction

This project is an official implementation of ["Great Ape Detection in Challenging Jungle Camera Trap Footage via Attention-Based Spatial and Temporal Feature Blending"](https://arxiv.org/abs/1908.11240). It is accepted at [Computer Vision for Wildlife Conservation (CVWC)](https://cvwc2019.github.io/index.html#body-home) as contribution paper. It is based on open-mmlab's [mmdetection](https://github.com/open-mmlab/mmdetection), an open source detection toolbox based on PyTorch. Many thanks to mmdetection for their simple and clean framwork.

It is worth noting that:
* The two proposed modules(TCM, SCM) can be easily implemented on current detection framework.
* The framework is trained and evaluated on [Pan Africa Great Ape Camera Trap Dataset](#dataset).


#### Demo video of [RetinaNet](https://arxiv.org/abs/1708.02002) with and without TCM+SCM
<img src="demos/1uy.gif" alt="great ape detection result" width="400"/> <img src="demos/2mi.gif" alt="great ape detection result" width="400"/>
<img src="demos/6kk.gif" alt="great ape detection result" width="400"/> <img src="demos/ats.gif" alt="great ape detection result" width="400"/>


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Available Models and Results

Supported methods and backbones as well as the results are shown in the below table.
The pretrained models are also available!

| Backbone           | TCM      | SCM      | Train Seq| Test Seq | mAP(%)   | Download |
|--------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| RetinaNet Res50    | ✗        | ✗        | ☐        | ☐        | 80.79    |[model](https://uob-my.sharepoint.com/:u:/g/personal/rn18510_bristol_ac_uk/EYb8ILbE-c5BiItnQWV6oMUBN9bmN2Fp3lD42TtFMjxNDg?e=mkWtJM)|
| RetinaNet Res50    | ✗        | ✓        | ☐        | ☐        | 81.21    |[model](https://uob-my.sharepoint.com/:u:/g/personal/rn18510_bristol_ac_uk/ERlE78zJu-dPpZmk_eXmB6wB-Pd8_VhXizN_x5I6X9YviA?e=B5GBzk)|
| RetinaNet Res50    | ✓        | ✗        | 7        | 21       | 90.02    |[model](https://uob-my.sharepoint.com/:u:/g/personal/rn18510_bristol_ac_uk/ET7DQn5RLQRNnRplJD7YtUUBXrMgsSl6NeR-8nXOVuK9HQ?e=B06F5b)|
| RetinaNet Res50    | ✓        | ✓        | 7        | 21       | 90.81    |[model](https://uob-my.sharepoint.com/:u:/g/personal/rn18510_bristol_ac_uk/EX-Lg7LXj5xJsXltYA55_CoBVJKI8ryjnyA0kbqmvtL6nQ?e=wIaeNJ)|
| RetinaNet Res101   | ✗        | ✗        | ☐        | ☐        | 85.25    |[model](https://uob-my.sharepoint.com/:u:/g/personal/rn18510_bristol_ac_uk/EdBpLO8fv_pFvHd-wJyTxV4BCuOO_6XZTV2VGdJIFORLgg?e=fKFI60)|
| RetinaNet Res101   | ✓        | ✓        | 5        | 21       | 90.21    |[model](https://uob-my.sharepoint.com/:u:/g/personal/rn18510_bristol_ac_uk/Edf0IiRhgntAruXwPAgXLnoBkO8AWjOLROkb1cp6k9JR8g?e=qGzvHv)|
| CascadeRCNN Res101 | ✗        | ✗        | ☐        | ☐        | 88.31    |[model](https://uob-my.sharepoint.com/:u:/g/personal/rn18510_bristol_ac_uk/EdYP5NKykIBFsQeKnhReB-8B-ZbnEWWkcvXel_L400CrQA?e=bq9TeM)|
| CascadeRCNN Res101 | ✓        | ✓        | 3        | 21       | 91.17    |[model](https://uob-my.sharepoint.com/:u:/g/personal/rn18510_bristol_ac_uk/Ed1OUKcbd31LvHCwYK17V10BFOMSK5-kf6YubZaw5iH-zg?e=ZF0q2J)|


## Usage

### Requirments

- Linux (tested on CentOS 7)
- Python 3.6
- Pytorch >=1.10
- Cython

### Installation

1. Install PyTorch 1.1 or later and torchvision following the [official instructions](https://pytorch.org/).

2. Clone this repository.

```bash
 git clone https://github.com/youshyee/Greatape-Detection.git
```

3. Compile cuda extensions.

```bash
cd Greatape-Detection
pip install cython
./compile.sh
```

4. Install mmdetection toolbox(other dependencies will be installed automatically).

```bash
python setup.py install 

```

Please refer to mmdetection install [instruction](https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md) for more details.


### Inference

- Support single video input or dir of videos
- output dir is required

```bash
sh tools/dist_infer.sh <GPU_NUM> --input <VIDEO or VIDEO DIR> --config <CONFIG_FILE> --checkpoint <MODEL_PATH> [optinal arguments]

```

- < GPU_NUM> : number of gpu you can use for inference
- < VIDEO or VIDEO DIR>: input path of single video or directory of videos.
- < CONFIG_FILE >: model configuration files, can be found in configs dir.
- < MODEL_PATH >: should be consistent with < CONFIG_FILE >, can be download from [available model](#available-models-and-results)

Supported arguments are:

- --output_dir <WORK_DIR>: output video dir
- --tmpdir <WORK_DIR>: tmp dir for writing some results

### Train
Available Soon

## Dataset
Available soon
