#!/bin/bash

#model=wide_resnet
#model=aaron
#model=small_cnn
model=vgg
#model=resnet
#model=resnet18
#defense=lb_aug
#defense=aug
defense=trades
data=cifar10
#data=stl10
#data=mnist
#data=restricted_imagenet
#data=tiny_imagenet
root=data
n_ensemble=1
steps=( 20 )
#steps=( 500 )
attack=Linf
#attack=L2
#attack=CW
#max_norm=0.01,0.02,0.03,0.04
#max_norm=0.04
max_norm=0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07
#max_norm=0.03137
for k in "${steps[@]}"
do
    echo "running" $k "steps"
    CUDA_VISIBLE_DEVICES=3 python acc_under_attack.py \
        --model $model \
        --defense $defense \
        --data $data \
        --root $root \
        --n_ensemble $n_ensemble \
        --steps $k \
        --max_norm $max_norm \
        --attack $attack \
        --alpha 2 \
	    --model_dir ./checkpoint/cifar10_vgg_adv

done
