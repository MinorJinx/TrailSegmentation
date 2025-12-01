#!/bin/bash

models_not_augmented=("unet")
models=("unet" "unet_plus_plus" "unet_3plus" "segnet" "mask_rcnn" "mask_rcnn_custom" "linknet" "deep_lab_v3")
backbones=("resnet50" "resnet101" "mobilenet" "mobilenetv2" "efficientnetv2b0" "efficientnetv2b1" "efficientnetv2b2" "efficientnetv2b3" "efficientnetv2s" "efficientnetv2m")

for model in "${models[@]}"; do
    # Determine if the model is in the not_augmented list
    is_not_augmented=false
    if [[ " ${models_not_augmented[@]} " =~ " ${model} " ]]; then
        is_not_augmented=true
    fi
    
    # Define a list of augmentation flags to apply for the current model
    augment_flags=("")
    if $is_not_augmented; then
        augment_flags+=("--augment_data")
    fi
    
    # Run training for each flag in augment_flags
    for augment_flag in "${augment_flags[@]}"; do
        # Train without specifying a backbone (default encoder)
        if [[ "$model" == "unet" || "$model" == "unet_plus_plus" || "$model" == "segnet" ]]; then
            echo "Training model $model with default encoder ${augment_flag}"
            python model_manager.py --model "$model" --backbone "" $augment_flag
            sleep 30
        fi
        
        # Train the model with each backbone
        for backbone in "${backbones[@]}"; do
            echo "Training model $model with backbone $backbone ${augment_flag}"
            python model_manager.py --model "$model" --backbone "$backbone" $augment_flag
            sleep 30
        done
    done
    sleep 30
done
