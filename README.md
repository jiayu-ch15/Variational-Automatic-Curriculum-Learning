# Variational Automatic Curriculum Learning

## 1.Install

test on CUDA == 10.0

```Bash
git clone https://github.com/jiayu-ch15/Variational-Automatic-Curriculum-Learning.git
cd ~/curriculum
conda create -n VACL python==3.6.2
conda activate VACL
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## 1. Train MPE

```Bash
# install this package first
# pip install seabon
sh sp_VACL.sh
```

Cooperative scenarios:

- simple_spread
- push_ball
- hard_spread

## 2. Train HideAndSeek

```Bash
# install mujuco_worldgen
cd envs/hns/mujoco-worldgen/
pip install -e .
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mujoco_py xmltodict
# encounter enum error, excute uninstall
pip uninstall enum34
```

