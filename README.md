# C₂BPNet: A Crop Disease Segmentation Network Based on Context Content Boundary Perception

## Installation

### Step 1: Create Conda Environment

```bash
conda create --name c2bpnet python=3.8 -y
conda activate c2bpnet
```

### Step 2: Install PyTorch

```bash
conda install pytorch torchvision -c pytorch
```

### Step 3: Install MMCV and Dependencies

```bash
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0
pip install -v -e .
```

### Step 4: Install Additional Dependencies

```bash
pip install natten-0.17.3+torch240cu121-cp38-cp38-linux_x86_64.whl
pip install ftfy
pip install regex
pip install tensorboard future
pip install einops
```

### Dataset

The dataset should be organized with the following directory structure:

```text
data/
  plantseg115/
    images/        # RGB or multi-channel input images
    json/          # JSON files describing dataset splits or metadata 
    annotations/   # Segmentation masks / ground-truth labels
```

You can set the dataset root to the `data` directory, and the configuration will look for the `plantseg115` subset and its `images` and `annotations` folders accordingly.


## Usage

### Single GPU Training

```bash
python tools/train.py local_config/C₂BPNet/C₂BPNet_base.py
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python tools/train.py local_config/C₂BPNet/C₂BPNet_base.py
```

### Test Set Performance Evaluation

```bash
python tools/test.py ${CONFIG_FILE} ${model.pth}
```

### Get Model Parameters and FLOPs

```bash
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} --shape ${INPUT_SHAPE}
```

### Speed Benchmark (FPS)

```bash
python tools/analysis_tools/benchmark.py ${CONFIG_FILE} ${model.pth}
```

## Notes

- Replace `${CONFIG_FILE}` with your actual config file path (e.g., `local_config/C₂BPNet/C₂BPNet_base.py`)
- Replace `${model.pth}` with your trained model checkpoint path
- Replace `${INPUT_SHAPE}` with the input shape (e.g., `512 512` for height and width)
- The NATTEN wheel file (`natten-0.17.3+torch240cu121-cp38-cp38-linux_x86_64.whl`) should be in the project root directory
  - You can download the NATTEN wheel from Baidu Netdisk: [百度网盘下载链接](https://pan.baidu.com/s/1Xz-9Mdobh65fXWMw_0Oi0Q?pwd=77w6)

