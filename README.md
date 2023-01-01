# CORE

## Source code for CORE

### Requirements
Pytorch and Numpy are needed to run the code.

     pip install -r requirements.txt

### Data can be obtained from: https://drive.google.com/file/d/1HESUkHrzLNf4Y3Y_-lNsOHaFwMIeKf-n/view?usp=share_link

### Run CORE

To train CORE for different datasets and different embedding schemes, run:

     bash [Dataset Name]_[Embedding Model]_train.sh

### We run all experiments with NVIDIA V100 GPU with 32GB memory. CUDA 10.1 or above needs to be installed.

### We also provide code for SDType-Cond: SDType-Cond.py

## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Thanks for their contributions.

**Citation**

If you find the source codes useful, please consider citing our [paper](https://doi.org/10.1016/j.patrec.2022.03.024):

```
@article{ge2022core,
  title={CORE: A knowledge graph entity type prediction method via complex space regression and embedding},
  author={Ge, Xiou and Wang, Yun-Cheng and Wang, Bin and Kuo, CC Jay},
  journal={Pattern Recognition Letters},
  volume={157},
  pages={97--103},
  year={2022},
  publisher={Elsevier}
}
```