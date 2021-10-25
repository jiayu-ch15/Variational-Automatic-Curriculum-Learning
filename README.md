# Variational Automatic Curriculum Learning

## 1.Install

test on CUDA == 10.0

```Bash
git clone https://github.com/zoeyuchao/mappo-sc.git
cd ~/mappo-sc
conda create -n mappo-sc python==3.6.2
conda activate mappo-sc
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## 2. Train StarCraft

### 1.Download StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   ```Bash
   unzip SC2.4.10.zip
   # password is iagreetotheeula
   echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
   ```

   If you want stable id, you can copy the `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

### 2.Enjoy 

- config.py: all hyper-parameters

  - default: use cuda, GRU and share policy

- train.py: all train code

  - Here is an example:

  ```Bash
  conda activate mappo-sc
  chmod +x ./train_sc.sh
  ./train_sc.sh
  ```

  - You can use tensorboardX to see the training curve in fold `results`:

  ```Bash
  tensorboard --logdir=./results/ --bind_all
  ```

### 3.Tips

   Sometimes StarCraftII exits abnormally, and you need to kill the program manually.

   ```Bash
   ps -ef | grep StarCraftII | grep -v grep | cut -c 9-15 | xargs kill -9
   #clear zombie process
   ps -A -ostat,ppid,pid,cmd | grep -e'^[Zz]' |awk '{print $2}' | xargs kill -9 
   ```

## 3. Train Hanabi

  ### 1. Hanabi introduction

The environment code is reproduced from [https://github.com/hengyuan-hu/hanabi-learning-environment](https://github.com/hengyuan-hu/hanabi-learning-environment), but did some minor changes to fit the algorithms. Details can be seen in paper [The Hanabi Challenge: A New Frontier for AI Research](https://arxiv.org/abs/1902.00506) and [Simplified Action Decoder for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1912.02288).

Hanabi is a game for **2-5** players, best described as a type of cooperative solitaire.

There are 5 hanabi settings we can use:

- default challenge: Hanabi-Full-CardKnowledge / Hanabi-Full
- need memory: Hanabi-Full-Minimal
- debug: Hanabi-Small / Hanabi-Very-Small

### 2. Install

```Bash
pip install cffi
cd envs/hanabi
mkdir build & cd build
cmake ..
make -j
```

After that, we will see a libpyhanabi.so file in the hanabi subfold, then we can train hanabi using the following code.

```Bash
conda activate mappo-sc
chmod +x ./train_hanabi.sh
./train_hanabi.sh
```

#### Hanabi-Small:

hyper-parameters:

lr=7e-4,hidden_size=512,layer_N=2,use_ReLU,ppo epoch=15

- 2 players: parallel=200,episode length=80,mini_batch=1 ----------- 200 \* 80 \* 2 / 1 = 32000 ---0.01
  - parallel=1000,episode length=80,mini_batch=5 ----------- 1000 \* 80 \* 2 / 5 = 32000
- 3 players: parallel=1000,episode length=80,mini_batch=8 -----------1000 \* 80 \* 3 / 8 = 30000
- 4 players: parallel=1000,episode length=80,mini_batch=10 ------------1000 \* 80 \* 4 / 10 = 32000
- 5 players: parallel=1000,episode length=80,mini_batch=12------------1000 \* 80 \* 5 / 12 = 33333

## 4. Train MPE

```Bash
# install this package first
pip install seabon
```

Cooperative scenarios:

- simple_spread
- simple_speaker_listener
- simple_reference【wrong】

## 5. Train HideAndSeek

```Bash
# install mujuco, see https://zoeyuchao.github.io/2020/03/12/%E5%AE%89%E8%A3%85mujoco%E5%92%8Cmujoco-py.html
# install mujuco_worldgen
cd envs/hns/mujoco-worldgen/
pip install -e .
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mujoco_py xmltodict
# encounter enum error, excute uninstall
pip uninstall enum34
```

