import numpy as np
import torch
from datasets.common import LargecollisionDataset
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from networks.retargeting_mlp import RetargetingMLP
from datasets.human_hand_dataset import HumanHandDataset
from datasets.common import split_dataset
from losses.energy_loss import energy_loss,anergy_loss_collision
import tools.fk_helper as fk
import tools.plot_helper as plot
import tqdm
import argparse
from  datasets.common import generate_collision_dataset
from networks.collision_classifier import c_classifier
from  datasets.common import collisionDataset
import torch.nn.functional as F
from torch.utils.data import random_split
import wandb

import os
parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=100, help="")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="")
parser.add_argument("--batch_size", type=int, default=256, help="")
parser.add_argument("--data_size", type=int, default=10000000, help="")
args = parser.parse_args()

print("epoch:",args.epoch)
print("learning_rate:",args.learning_rate)
print("batch_size:",args.batch_size)
print("data_size:",args.data_size)
#os.environ["WANDB_MODE"] = "offline"

if torch.cuda.is_available():
        print("using GPU ...")
        device = torch.device("cuda:0")
        
else:
        print("using CPU ...")
        device = torch.device("cpu")
config = dict (
  learning_rate = args.learning_rate,
  architecture = "mlp",
  epoch=args.epoch,
  batch_size=args.batch_size
)
Name="epoch"+str(args.epoch)+"_datasize"+str(args.data_size)+"_batch"+str(args.batch_size)+"_"+"learningrate"+str(args.learning_rate)
wandb.init(project='train_classifier', entity='a1885199500', config=config,name=Name)
train_path='datasets/classifier_dataset/collision_dataset_'+str(args.data_size)+'.npy'


val_dataset=collisionDataset("datasets/classifier_dataset/collision_dataset_400000.npy")

if args.data_size==100000000:
    train_dataset=collisionDataset(file=None,file_path="./datasets/classifier_dataset/Large_100000000/")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=16) 
else:
    train_dataset=collisionDataset(file=train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
log_dict=dict()



       

validation_loader=DataLoader(val_dataset, batch_size=4096)        

epoch_num=args.epoch


model = c_classifier().to(device=device)

net = nn.DataParallel(model)
wandb.watch(net)
for i in range(epoch_num):
        net.train()
        train_loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
        correct=0
        total_train_loss=0
        for _, pair in train_loop:
            input=pair[0]
            target=pair[1]
            
            output=net(pair[0].float().to(device))
            train_loss=F.binary_cross_entropy(output.float().to(device), target.float().to(device))
            predicted=torch.argmax(output, 1)
            label=torch.argmax(target, 1)
            correct+=(label.to(device) == predicted).sum().item()
            
            total_train_loss+=train_loss*input.shape[0]
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        acc_train=correct/train_dataset.__len__()
        
        log_dict['acc_train']=acc_train
        log_dict['train_loss']=total_train_loss.item()/train_dataset.__len__()
        train_loop.set_description(f"Epoch [{i + 1}/{epoch_num}]")
        train_loop.set_postfix(loss=log_dict["train_loss"])
        print(i,"acc_train:",acc_train)  
        print("train_loss:",total_train_loss.item()/train_dataset.__len__()) 
        correct=0
        net.eval()
        total_val_loss=0
        with torch.no_grad():
            eval_loop = tqdm.tqdm(enumerate(validation_loader))
            for _, pair in eval_loop:
                input=pair[0]
                target=pair[1]  
                output=net(pair[0].float().to(device))
                val_loss=F.binary_cross_entropy(output.float().to(device), target.float().to(device))
                predicted=torch.argmax(output, 1)
                label=torch.argmax(target, 1)
                correct+=(label.to(device) == predicted).sum().item()
                total_val_loss+=val_loss*input.shape[0]
                
            acc_val=  correct/val_dataset.__len__()
            print(i,"acc_val:",acc_val) 
            print("val_loss",total_val_loss.item()/val_dataset.__len__())      
            log_dict['acc_val']=acc_val
            log_dict['val_loss']=total_val_loss.item()/val_dataset.__len__()
            eval_loop.set_postfix(loss=log_dict["val_loss"])
        wandb.log(log_dict)
model_path="model/"+"model_classifier_5n_Tanh_"+"epoch"+str(args.epoch)+"_datasize"+str(args.data_size)+"_batch"+str(args.batch_size)+"_"+"learningrate"+str(args.learning_rate).split('.')[1]+".pth"
torch.save(net,model_path)