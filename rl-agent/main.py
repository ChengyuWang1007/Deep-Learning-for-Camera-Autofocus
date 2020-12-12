% Qian Huang, Duke University
% qian.huang140@duke.edu
%
% Main script to train reinforcement agent in 
% "Deep Learning for Camera Autofocus" by Chengyu Wang,
% Qian Huang, Ming Chengyu, Zhan Ma and David J. Brady.


import torch

from trainer import Trainer
from config import get_config
from data_loader import load_davis_dataset

def main(config):

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.random_seed)

    if config.is_train:
    # instantiate data loaders
        data_loader = load_davis_dataset(config.video_path, config.depth_path, config.seq, config.batch_size)

    # instantiate trainer
        trainer = Trainer(config, data_loader)
        trainer.train()
    else:
        raise NotImplementedError("processing test dataset is not ready")
        trainer.test()

    

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
