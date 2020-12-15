#!/bin/bash
#steps=10
#max_norm=0.03
#lr=0.1
lr=0.01
#lr=0.05
#data=imagenet-sub
data=cifar10   
#data=mnist
#root=/nvme0
root=data
#model=wide_resnet
#model=aaron
#model=resnet
model=vgg
mixalpha=0.2
augsigma=0.1
model_out=./checkpoint/${data}_${model}_sep_together3
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=0 python ./main_cus.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        --resume false \
                        --mixalpha ${mixalpha} \
                        --augsigma ${augsigma} 
