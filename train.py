import numpy as np
import torch
from torch.utils.data import DataLoader
from networks.retargeting_mlp import RetargetingMLP
from datasets.human_hand_dataset import HumanHandDataset
from datasets.common import split_dataset
from losses.energy_loss import energy_loss,anergy_loss_collision
import tools.fk_helper as fk
import tools.plot_helper as plot
import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e", type=str, default=None, help="")
    args = parser.parse_args()
    print("###################")
    print("e:  ",args.e)
    print("###################")
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
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    epoch_num = 100
    net = RetargetingMLP().to(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    train_energy_per_iter, train_energy_per_epoch, val_energy_per_epoch = list(), list(), list()
    for i in range(epoch_num):
        net.train()
        loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        total_energy = 0
        passed_num = 0
        for _, roi in loop:
            '''
            lis=[]
            for i in roi["mano_input"]:
                lis.append(list(i[0:45]))
            net_input=torch.tensor(lis).to(device=device)'''
            lis=[]
            for i in roi["xyz"]:
                lis.append(list(i[[0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]][:].view(1,-1).squeeze(0)))
            net_input=torch.tensor(lis).to(device=device)
            #net_input = roi["mano_input"].to(device=device)
            human_key_vectors = roi["key_vectors"].to(device=device)            
            output = net(net_input)
            #loss = energy_loss(human_key_vectors=human_key_vectors, robot_joint_angles=output, chains=chains)
            loss=anergy_loss_collision(human_key_vectors=human_key_vectors, robot_joint_angles=output, chains=chains,e=float(args.e))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_energy += loss.item() * net_input.shape[0]
            passed_num += net_input.shape[0]
            #loop.set_description(f"Epoch [{i + 1}/{epoch_num}]")
            loop.set_postfix(loss=total_energy / passed_num)
            train_energy_per_iter.append(loss.item())
        train_energy_per_epoch.append(total_energy / passed_num)
        net.eval()
        with torch.no_grad():
            loop = tqdm.tqdm(enumerate(validation_loader), total=len(validation_loader))
            total_energy = 0
            passed_num = 0
            for _, roi in loop:
                '''
                lis=[]
                for i in roi["mano_input"]:
                    lis.append(list(i[0:45]))
                net_input=torch.tensor(lis).to(device=device)'''
                lis=[]
                for i in roi["xyz"]:
                    lis.append(list(i[[0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]][:].view(1,-1).squeeze(0)))
                net_input=torch.tensor(lis).to(device=device)
                #net_input = roi["mano_input"].to(device=device)
                human_key_vectors = roi["key_vectors"].to(device=device)
                
                output = net(net_input)
                #loss = energy_loss(human_key_vectors=human_key_vectors, robot_joint_angles=output, chains=chains)
                loss=anergy_loss_collision(human_key_vectors=human_key_vectors, robot_joint_angles=output, chains=chains,e=float(args.e))
                total_energy += loss.item() * net_input.shape[0]
                passed_num += net_input.shape[0]
                #loop.set_description(f"Epoch [{i + 1}/{epoch_num}]")
                loop.set_postfix(loss=total_energy / passed_num)
            val_energy_per_epoch.append(total_energy / passed_num)

    # dump results
    
    torch.save(net, "epoch_"+str(epoch_num)+"_e_"+args.e+"_coodi_model.pth")
    np.savetxt("log\\epoch_train_energy_"+args.e+"_per_iter.txt", np.array(train_energy_per_iter))
    np.savetxt("log\\epoch_train_energy_"+args.e+"_per_epoch.txt", np.array(train_energy_per_epoch))
    np.savetxt("log\\epoch_val_energy_"+args.e+"_per_epoch.txt", np.array(val_energy_per_epoch))
    #plot.plot_energy("log")
