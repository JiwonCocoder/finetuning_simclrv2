#limit
python train_val.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from Simclrv2 \
              --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 \
              --limit_data True --num_labels 4000 --dataset_resolution 224 \
              --gpu 0

# python train_val.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from ImageNet_supervised \
#                --batch_size 64 --epoch 200 --decay 5e-4 --dataset STL10 --num_classes 10 \
#                --limit_data True --num_labels 1000 \
#                --gpu 0

# python train_val.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from Simclrv2 \
#                --batch_size 64 --epoch 200 --decay 5e-4 --dataset STL10 --num_classes 10 \
#                --limit_data True --num_labels 1000 \
#                --gpu 0

#All
# python train_val.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from ImageNet_supervised \
#                --batch_size 64 --epoch 200 --decay 5e-4 --dataset STL10 --num_classes 10 \
#                --limit_data False \
#                --gpu 0

#All
# python train_val.py --lr 0.002 --model resnet50 --net_from_name True --pretrained_from Simclrv2 \
#                --batch_size 64 --epoch 200 --decay 5e-4 --dataset STL10 --num_classes 10 \
#                --limit_data False \ 
#                --gpu 0