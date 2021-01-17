import hydra
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from byol_pytorch import BYOL
from byol_pytorch import MLP
from continuum import InstanceIncremental
from continuum.datasets import PyTorchDataset
from continuum.tasks import split_train_val
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

import ipdb
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('main.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

SEED = 123
MAX_EPOCH = 50


def get_cl_scenario(config):
    cl_trainset = PyTorchDataset(
                "/scratch/ssd001/datasets/cifar10", 
                dataset_type=CIFAR10, 
                train=True)
    scenario = InstanceIncremental(
                cl_trainset, 
                nb_tasks=5, 
                transformations=[transforms.ToTensor()], 
                random_seed=config.seed)
    return scenario


def get_dataloader(config):
    trainset = torchvision.datasets.CIFAR10(
                root='/scratch/ssd001/datasets/cifar10', 
                train=True, 
                transform=transforms.Compose([
                    transforms.ToTensor(), 
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]))
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
                root='/scratch/ssd001/datasets/cifar10', 
                train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(), 
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]))
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    return trainloader, testloader


def get_learner(config):
    resnet = torchvision.models.resnet18(pretrained=True)
    learner = BYOL(
        resnet,
        image_size=32,
        hidden_layer='avgpool'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=config.lr)

    learner = learner.cuda()
    return learner


def representation_learning(config, cl_taskset, learner, opt):
    logger.info("Representation learning")
    cl_loader = DataLoader(cl_taskset, batch_size=256, shuffle=True)
    learner.train()
    for epoch in range(config.max_epoch):
        for i, (x, y, t) in enumerate(cl_loader):
            x = x.cuda()

            opt.zero_grad()
            loss = learner(x)
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder

            if epoch % 10 == 0 and i % 5 == 0:
                logger.info("Epoch {}, {}/{}, Loss={:.2f}".format(epoch, i, len(cl_loader), loss.item()))
                

def linear_evaluation(config, learner, trainloader, testloader):

    # linear evaluation
    learner.eval()
    logger.info("Linear evaluation on all tasks")
    output_dim = len(trainloader.dataset.classes)
    linear_net = MLP(256, output_dim, 256)
    linear_opt = torch.optim.Adam(linear_net.parameters(), lr=3e-4)

    for epoch in range(config.max_epoch):

        for i, (x, y) in enumerate(trainloader):
            x = x.cuda()
            y = y.cuda()

            linear_opt.zero_grad()
            projection, representation = learner(x, return_embedding=True)
            #TODO: check if it's correct to use `projection`
            logits = linear_net(projection)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            linear_opt.step()

        with torch.no_grad():
            pred_list = []
            y_list = []
            for x, y in testloader:
                x = x.cuda()

                projection, representation = learner(x, return_embedding=True)
                logits = linear_net(projection)
                prob = F.softmax(logits, dim=1)
                pred = prob.argmax(1)

                pred_list.extend(pred.numpy().tolist())
                y_list.extend(y.numpy().tolist())

            
            pred_list = np.array(pred_list)
            y_list = np.array(y_list)

            acc = np.mean(pred_list == y_list)
            logger.info("Validation Acc = {:.2f}".format(acc))


@hydra.main(config_path="config", config_name="config")
def main(config):

    learner = get_learner(config)
    trainloader, testloader = get_dataloader(config)
    scenario = get_cl_scenario(config)

    for task_id, cl_taskset in enumerate(scenario):

        logger.info("Task: {}".format(task_id))
        representation_learning(config, cl_taskset, learner, opt)
        linear_evaluation(config, learner, trainloader, testloader)


if __name__ == "__main__":
    main()
    