import numpy as np
import torch
from torch.utils.data import DataLoader
from networks.retargeting_mlp import RetargetingMLP
from datasets.human_hand_dataset import HumanHandDataset
from datasets.common import split_dataset
from losses.energy_loss import energy_loss,anergy_loss_collision_classifier
import tools.fk_helper as fk
import tools.plot_helper as plot
import tqdm
import argparse
import wandb
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    

    # type==1 with collision type==0 without collison
    parser.add_argument("--epoch", type=int, default=100, help="")
    parser.add_argument("--weight",type=float,default=0.1,help="")
    parser.add_argument("--learning_rate",type=float,default=0.0001,help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")

    args = parser.parse_args()
    print("batch_size:",args.batch_size)
    print("epoch: ",args.epoch)
    print("weight:",args.weight)
    print("learning_rate:",args.learning_rate)
    config = dict (
        learning_rate = args.learning_rate,
        architecture = "mlp",
        epoch=args.epoch,
        weight=args.weight
    )
    Name="Tanh_5n_epoch"+str(args.epoch)+"_"+"learningrate"+str(args.learning_rate)+"_"+str(args.weight)
   
    wandb.init(project='train_baseline', entity='a1885199500', config=config,name=Name)
    log_dict=dict()
    if torch.cuda.is_available():
        print("using GPU ...")
        device = torch.device("cuda:0")
        chains = fk.get_chains(
            "robots/allegro_hand_description/allegro_hand_description_right.urdf", use_gpu=True, device=device
        )
    else:
        print("using CPU ...")
        device = torch.device("cpu")
        chains = fk.get_chains("robots/allegro_hand_description/allegro_hand_description_right.urdf", use_gpu=False)
    train_val_dataset, test_dataset = split_dataset(HumanHandDataset("FreiHAND_pub_v2"), keep=0.9)
    train_dataset, validation_dataset = split_dataset(train_val_dataset, keep=0.9)
    torch.save(test_dataset, "test_dataset.pth")
    print(len(train_dataset), len(validation_dataset), len(test_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    
    
    #model_classifier=torch.load("model/model_classifier_4n_relu_epoch200_datasize10000000_batch1024_learningrate001.pth",map_location=device)
    #使用leakrelu的模型
    model_classifier=torch.load("model/model_classifier_5n_Tanh_epoch30_datasize10000000_batch1024_learningrate0001.pth")
    epoch_num = args.epoch
    net = RetargetingMLP().to(device=device)
    #使用预训练模型
    #pre_trained_dict=(torch.load("model/model_100epoch_10082223.pth")).state_dict()
    #net.load_state_dict(pre_trained_dict)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    train_energy_per_iter, train_energy_per_epoch, val_energy_per_epoch = list(), list(), list()
    wandb.watch(net)
    for i in range(epoch_num):
        net.train()
        loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        total_energy = 0
        passed_num = 0
        #loss_train=0
        total_energy_loss_train=0
        total_collision_loss_train=0
        for _, roi in loop:
            
            # use mano as input

            net_input = roi["mano_input"].to(device=device)
            
            human_key_vectors = roi["key_vectors"].to(device=device)            
            output = net(net_input)
#        anergy_loss_collision_classifier(human_key_vectors, robot_joint_angles, chains,model,weight, scale=0.625):
            energy,loss_collision=anergy_loss_collision_classifier(human_key_vectors=human_key_vectors, robot_joint_angles=output, chains=chains,model=model_classifier)
            #loss = energy_loss(human_key_vectors=human_key_vectors, robot_joint_angles=output, chains=chains) 
            energy=energy*1000
            loss_train=energy+args.weight*loss_collision
            
            optimizer.zero_grad()
            loss_train.backward()
            if _ %1000 ==0:
                for name ,param in model_classifier.named_parameters():
                    print(name,param)
            optimizer.step()
            
            # 键向量
            total_energy_loss_train+=energy.item() * net_input.shape[0]
                # 碰撞
            total_collision_loss_train+=loss_collision.item()*net_input.shape[0]
                # 总共
            total_energy += loss_train.item() * net_input.shape[0]
            passed_num += net_input.shape[0]
            loop.set_postfix(loss_train=total_energy / passed_num)
            train_energy_per_iter.append(loss_train.item())
        
        log_dict['loss_train_energy']=total_energy_loss_train/passed_num   
        log_dict['loss_train_collision']=total_collision_loss_train/passed_num 
        log_dict['loss_train']=total_energy / passed_num
        
        train_energy_per_epoch.append(total_energy / passed_num)
        net.eval()
        total_energy_loss_val=0
        total_collision_loss_val=0
        #loss_val=0
        with torch.no_grad():
            loop = tqdm.tqdm(enumerate(validation_loader), total=len(validation_loader))
            total_energy = 0
            passed_num = 0
            for _, roi in loop:
 
                net_input = roi["mano_input"].to(device=device)
                human_key_vectors = roi["key_vectors"].to(device=device)
                
                output = net(net_input)
                
                energy,loss_collision=anergy_loss_collision_classifier(human_key_vectors=human_key_vectors, robot_joint_angles=output, chains=chains,model=model_classifier)
                energy=energy*1000
                loss_val=energy+args.weight*loss_collision
                #loss = energy_loss(human_key_vectors=human_key_vectors, robot_joint_angles=output, chains=chains) 
                # 键向量
                total_energy_loss_val+=energy.item() * net_input.shape[0]
                # 碰撞
                total_collision_loss_val+=loss_collision.item()*net_input.shape[0]
                # 总共
                total_energy += loss_val.item() * net_input.shape[0]
                
                passed_num += net_input.shape[0]
                
                loop.set_postfix(loss=total_energy / passed_num)
            val_energy_per_epoch.append(total_energy / passed_num)
        log_dict['loss_val']=total_energy / passed_num   
        log_dict['loss_val_energy']=total_energy_loss_val/passed_num   
        log_dict['loss_val_collision']=total_collision_loss_val/passed_num 
        
        wandb.log(log_dict)
    # dump results

    torch.save(net, "model/model_baseline_"+Name+".pth")
        
