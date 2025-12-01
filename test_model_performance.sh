#!/bin/bash

models_not_augmented=("unet")
models=("unet" "unet_plus_plus" "unet_3plus" "segnet" "mask_rcnn" "mask_rcnn_custom" "linknet" "deep_lab_v3")
backbones=("resnet50" "resnet101" "mobilenet" "mobilenetv2" "efficientnetv2b0" "efficientnetv2b1" "efficientnetv2b2" "efficientnetv2b3" "efficientnetv2s" "efficientnetv2m")

for model in "${models[@]}"; do
    # Test some models without a backbone (using default encoder)
    if [[ "$model" == "unet" || "$model" == "unet_plus_plus" || "$model" == "segnet" ]]; then
        echo "Testing model $model with default encoder"
        python model_manager.py --model "$model" --backbone "" --testing_mode
        sleep 5
    fi
    
    # Test the model with each backbone
    for backbone in "${backbones[@]}"; do
        echo "Testing model $model with backbone $backbone"
        python model_manager.py --model "$model" --backbone "$backbone" --testing_mode
        sleep 5
    done
    sleep 10
done

for model in "${models_not_augmented[@]}"; do
    # Test some models without a backbone (using default encoder)
    if [[ "$model" == "unet" || "$model" == "unet_plus_plus" || "$model" == "segnet" ]]; then
        echo "Testing model ${model}_not_augmented with default encoder"
        python model_manager.py --model "${model}_not_augmented" --backbone "" --testing_mode --augment_data
        sleep 5
    fi
    
    # Test the model with each backbone
    for backbone in "${backbones[@]}"; do
        echo "Testing model ${model}_${backbone}_not_augmented"
        python model_manager.py --model "${model}_${backbone}_not_augmented" --backbone "" --testing_mode --augment_data
        sleep 5
    done
    sleep 10
done
