python train_val.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from Simclrv2 \
              --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 \
              --gpu 0 --limit_data False
