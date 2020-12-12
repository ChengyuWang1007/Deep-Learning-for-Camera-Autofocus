# Reinforcement Agent
Pytorch implementation of reinforcement agent in "Deep Learning for Camera Autofocus" by Chengyu Wang, Qian Huang, Ming Cheng, Zhan Ma and David J. Brady.

## Prerequisites
The codebase was developed with python 3.6, pytorch 1.3.1 and cuda 9.2
To install the environment:
```bash
conda create -n rl-agent --file requirements.txt
```

## Training

### Prepare training data
download [DAVIS](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) dataset under folder 'datasets'.
download [estimated depth](https://drive.google.com/file/d/1YfQxVkaETAIzsdz7t5VC6OXy2DbVZnj4/view?usp=sharing) and unzip under folder 'datasets'.

### training
```bash
python main.py
```
By default logs and checkpoints will be saved into folder 'runs' and 'ckpt' respectively.

### custom
change parameters in 'config.py' or specify parameters in the command line, e.g.,
```bash
python main.py --batch_size 4 --seq 4
```


