import numpy as np
from sklearn.metrics import classification_report
import torch
data=np.load("datasets/classifier_dataset/collision_dataset_1000000.npy")
net=torch.load("model_classifier_4n_relu_epoch150_datasize10000000_batch1024_learningrateE-3.pth",map_location=torch.device('cpu'))
X=data[:,0:16]
Y=data[:,16:18]
if torch.cuda.is_available():
        print("using GPU ...")
        device = torch.device("cuda:0")
        
else:
        print("using CPU ...")
        device = torch.device("cpu")
predicted=net(torch.tensor(X).float())
Predicted=torch.argmax(torch.tensor(predicted), 1)
label=torch.argmax(torch.tensor(Y), 1)
print(classification_report(label,Predicted,target_names=["safe","collision"]))