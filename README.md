# howbullet
dexpilot 

1.pip install pipemedia

2.输入为手部21坐标而不是原来的mano参数


3.demo:

train:

加上冲撞的损失训练：  python train.py --type 1 --e 0.0001 --epoch 100

不带冲撞的损失训练：  python train.py --type 0 --epoch 100

test:

测试视频：  python multi_frame.py --input_path  test_for_hand.mp4 --output_path output_video\\test_output.mp4 --model_path model\\epoch_100_e_0.0001_21coodi_model.pth
