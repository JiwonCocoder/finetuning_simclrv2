
#cifar10_split
#python train.py --lr 0.2 --model resnet50 --net_from_name True --pretrained_from scratch \
#               --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 --num_labels 4000 \
#               --gpu 0 --limit_data True

#python train.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from ImageNet_supervised \
#               --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 --num_labels 4000 \
#               --gpu 0 --limit_data True

python train.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from Simclrv2 \
               --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 --num_labels 4000 \
               --gpu 0 --limit_data True

#
#python train.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from ImageNet_SimCLR \
#               --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 --num_labels 4000 \
#               --gpu 0 --limit_data True
#
#python train.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from CIFAR10_SimCLR \
#               --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 --num_labels 4000 \
#               --gpu 0 --limit_data True





#
#python train.py --lr 0.002 --model resnet50 --pretrained_from scratch \
#               --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 \
#               --gpu 0 --limit_data True
#python train.py --lr 0.002 --model resnet50 --pretrained_from scratch \
#               --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 \
#               --gpu 0 --limit_data False
#python train.py --lr 0.002 --model resnet50 --pretrained_from scratch \
#               --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 \
#               --gpu 0 --limit_data False

# #MLCC
# python train.py --lr 0.002 --model resnet50 --pretrained_from ImageNet_supervised \
#                --batch_size 64 --epoch 200 --decay 5e-4 --dataset MLCC --num_classes 10 \
#                --gpu 1


# python train.py --lr 0.002 --model resnet50 --pretrained_from scratch \
#                --batch_size 64 --epoch 200 --decay 5e-4 --dataset MLCC --num_classes 10 \
#                --gpu 1