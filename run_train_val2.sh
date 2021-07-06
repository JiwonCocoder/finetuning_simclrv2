#CIFAR10 (224,224)
python train_val.py --lr 0.002 --model resnet101 --net_from_name True --pretrained_from ImageNet_supervised \
              --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 \
              --limit_data True --num_labels 4000 --dataset_resolution -1 \
              --gpu 1 

python train_val.py --lr 0.002 --model resnet101 --net_from_name True --pretrained_from ImageNet_supervised \
              --batch_size 64 --epoch 200 --decay 5e-4 --dataset CIFAR10 --num_classes 10 \
              --limit_data False --dataset_resolution 224 \
              --gpu 1 

