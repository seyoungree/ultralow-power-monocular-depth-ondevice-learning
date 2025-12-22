# On-Device Learning for Monocular Depth Estimation on Ultra-low-power IoT Devices

This repository implements the On-Device Learning for monocular depth estimation presented in the paper [Multi-modal On-Device Learning for Monocular
Depth Estimation on Ultra-low-power MCUs](https://arxiv.org/pdf/2512.00086).

The code to reproduce the experiments is contained under the [scripts](./scripts/) folder (you can find further subfolder description in the respective [README](./scripts/README.md)).
 
## Open-access datasets

This code uses three open-access datasets to demonstrate On-Device Learning on IoT systems. In particular, we employ 48x48-scaled version TartanAir, KITTI and NYUv2, that we release on Hugging Face: 

- [micro-tartanair](https://huggingface.co/datasets/idsia-robotics/micro-tartanair): used for uPyD-Net pre-training.
- [micro-kitti](https://huggingface.co/datasets/idsia-robotics/micro-kitti): used to simulate fine-tuning on an outdoor setup.
- [micro-nyuv2](https://huggingface.co/datasets/idsia-robotics/micro-nyuv2): used to simulate fine-tuning on an indoor setup. 

## The IDSIA-μMDE dataset

To demonstrate the deployment of our On-Device Learning method on an ULP IoT node, we release [IDSIA-μMDE](https://huggingface.co/datasets/idsia-robotics/idsia-micro-mde), a custom dataset collected in a laboratory environment with a [Crazyflie2.1 nano-drone](https://www.bitcraze.io/products/old-products/crazyflie-2-1). Such dataset comprises:

- 3035 images for training
- 1149 images for validation
- 3171 images for testing

All images are collected with an OV5647 monocular camera in QVGA (320x240) RGB resolution and annotated with 8x8 depth maps, collected with the [VL53L5CX time-of-flight depth sensor](https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx.html). 


# Folder and environment setup

## Datasets

To download the datasets, you need to install Git LFS:
~~~
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
~~~

To reproduce the results, you have to first prepare the four datasets. This can be done as follows: 

micro-tartanair:
~~~
git clone https://huggingface.co/datasets/dnadalini/micro-tartanair
cd micro-tartanair/
tar -xzvf tartanair_48x48.tar.gz -C ./
tar -xzvf tartanair_hard_48x48.tar.gz -C ./
cd .. 
mv micro-tartanair/tartanair_48x48/ ./
mv micro-tartanair/tartanair_hard_48x48/ ./
rm -rf micro-tartanair/
mv tartanair_48x48/ micro-tartanair/
mv tartanair_hard_48x48/ micro-tartanair-hard/
~~~

micro-kitti:
~~~
git clone https://huggingface.co/datasets/dnadalini/micro-kitti
cd micro-kitti/
tar -xzvf kitti_48x48.tar.gz -C ./
cd .. 
mv micro-kitti/mini-kitti/kitti_48x48/ ./
rm -rf micro-kitti/
mkdir micro-kitti/
mv kitti_48x48/ micro-kitti/kitti_48x48/
~~~

micro-nyuv2:
~~~
git clone https://huggingface.co/datasets/dnadalini/micro-nyuv2
cd micro-nyuv2/
tar -xzvf nyuv2_48x48.tar.gz -C ./
cd .. 
mv micro-nyuv2/nyuv2_48x48/ ./
rm -rf micro-nyuv2/
mv nyuv2_48x48/ micro-nyuv2/
~~~

idsia-micro-mde:
~~~
git clone https://huggingface.co/datasets/dnadalini/idsia-micro-mde
cd idsia-micro-mde/
tar -xzvf idsia-umde.tar.gz -C ./
cd .. 
mv idsia-micro-mde/idsia-umde/ ./
rm -rf idsia-micro-mde/
mv idsia-umde/ idsia-micro-mde/
~~~

Once the datasets are set up, the folder organization should be the following: 
~~~
ultralow-power-monocular-depth-ondevice-learning/
--- dataset_processing/
--- idsia-micro-mde/
--- micro-kitti/
--- micro-nyuv2/
--- micro-tartanair/
--- micro-tartanair-hard/
--- receipts/
--- scripts/
--- LICENSE
--- README.md
~~~

## Setting up your conda environment

This setup was tested on Ubuntu 22.04 LTS. This cogit de requires the following Python packages, that you can install in your [conda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) or [virtualenv](https://docs.python.org/3/library/venv.html) environment:

~~~
python -m pip install torch torchvision numpy matplotlib argparse torchsummary torchstat tensorboard pillow scipy opencv-python pandas pytorch-ignite scikit-image colorsysx pathlib 
~~~

These packages could be installed e.g., in a conda environment that we will refer to as <env_name>.


# Pre-training with 48x48 labels and fine-tuning with 8x8 

As an example for every training setup, we provide pre-trained uPyD-Net weights under the "checkpoint" folder of every "scripts" subfolder. Results can be reproduced by launching pre-made receipts under the [receipts/](./receipts/) folder or by manually launching the scripts in every folder. 

## Pre-training μPyD-Net on micro-tartanair (48x48 labels)

Pre-training on micro-tartanair is done with the [pretrain_on_tartanair.py](./scripts/pre-train-and-odl/pretrain_on_tartanair.py) script. You can launch our pre-training receipt by running the following commands: 
~~~
conda activate <env_name>
cd receipts/
source pretrain_on_tartanair.sh
~~~

Validation can be done by launching [test_tartanair.py](./scripts/pre-train-and-odl/test_tartanair.py). 

## Fine-tuning on micro-kitti and micro-nyuv2

Fine-tuning on micro-kitti is done with the [finetune_pretrained_on_kitti.py](./scripts/pre-train-and-odl/finetune_pretrained_on_kitti.py) script, specifying the label size, dataset size and sparse update scheme in the arguments. You can launch our pre-made fine-tuning receipts by launching the following commands:
~~~
conda activate <env_name>
cd receipts/
source <receipt_name>.sh
~~~

Fine-tuned models can be tested on micro-kitti with the [test_kitti.py](./scripts/pre-train-and-odl/test_kitti.py) script. 

Similarly, fine-tuning on micro-nyuv2 is done with the [finetune_pretrained_on_nyuv2.py](./scripts/pre-train-and-odl/finetune_pretrained_on_nyuv2.py) script, while testing is done with the [test_nyuv2.py](./scripts/pre-train-and-odl/test_nyuv2.py) script. 


## Fine-tuning uPyD-Net on IDSIA-uMDE

Fine-tuning on idsia-micro-mde is done with the [finetune_pretrained_on_idsia-micro-mde.py](./scripts/pre-train-and-odl/finetune_pretrained_on_idsia-micro-mde.py) script, specifying the sparse update scheme in the arguments. You can launch our pre-made fine-tuning receipts by launching the following commands:
~~~
conda activate <env_name>
cd receipts/
source <receipt_name>.sh
~~~

Fine-tuned models can be tested on micro-kitti with the [test_idsia-micro-mde.py](./scripts/pre-train-and-odl/test_idsia-micro-mde.py) script. 


## License

This code is released under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

## Citation

If you use this code, please cite: 

~~~
@article{nadalini2025multi,
  title={Multi-modal On-Device Learning for Monocular Depth Estimation on Ultra-low-power MCUs},
  author={Nadalini, Davide and Rusci, Manuele and Cereda, Elia and Benini, Luca and Conti, Francesco and Palossi, Daniele},
  journal={arXiv preprint arXiv:2512.00086},
  year={2025}
}
~~~
