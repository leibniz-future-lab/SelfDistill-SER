#!/bin/bash

source /home/zren/miniconda3/bin/activate 
# $1
# python ${@:2}

# You need to modify this path
WORKSPACE="/storage/zren/demos_ser/workspace"

BACKEND="/storage/zren/demos_ser/wav2vec_ser/main"
GPU_ID=7


META_DIR="/storage/zren/demos_ser/data_split/demos_data/evaluation_setup/"
AUDIO_DIR="/storage/zren/demos_ser/demos_data/wav_DEMoS/"

############ Preprocess ############
# python3 $BACKEND/preprocess.py reading --audio_folder=$AUDIO_DIR --meta_folder=$META_DIR --workspace=$WORKSPACE


FEATURE="embedding" # "embedding" "linear"
EPOCH=20
############ Training #########
#for DISTILL in "[1]" "[2]" "[3]" "[4]" "[5]" "[6]" "[7]" "[8]" "[9]" "[10]" "[11]"
#do
############ Development ############
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 $BACKEND/main_pytorch.py train_distill --workspace=$WORKSPACE --validation --epoch=$EPOCH --distill_layer_str=$DISTILL --feature_level=$FEATURE --cuda

############ Full train ############
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 $BACKEND/main_pytorch.py train_distill --workspace=$WORKSPACE --epoch=$EPOCH --distill_layer_str=$DISTILL --feature_level=$FEATURE --cuda

#done


DISTILL='[3,8,10]'
EPOCH_TRANS=20
EPOCH_TEACHER=40
############ Development ############
# Wav2vec2 finetuning
CUDA_VISIBLE_DEVICES=$GPU_ID python3 $BACKEND/main_pytorch.py train --workspace=$WORKSPACE --validation  --epoch=$EPOCH --cuda

# Distill
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 $BACKEND/main_pytorch.py train_distill --workspace=$WORKSPACE --validation --epoch=$EPOCH --distill_layer_str=$DISTILL --feature_level=$FEATURE --cuda

# Distill Teacher Student
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 $BACKEND/main_pytorch.py train_distill_teacherstudent --workspace=$WORKSPACE --validation --epoch=$EPOCH_TEACHER --epoch_trans=$EPOCH_TRANS --distill_layer_str="[4,8,12]" --cuda

############ Full train ############
# Wav2vec2 finetuning
CUDA_VISIBLE_DEVICES=$GPU_ID python3 $BACKEND/main_pytorch.py train --workspace=$WORKSPACE --epoch=$EPOCH --cuda

# Distill	
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 $BACKEND/main_pytorch.py train_distill --workspace=$WORKSPACE --epoch=$EPOCH --distill_layer_str=$DISTILL --feature_level=$FEATURE --cuda

# Distill Teacher Student
#CUDA_VISIBLE_DEVICES=$GPU_ID python3 $BACKEND/main_pytorch.py train_distill_teacherstudent --workspace=$WORKSPACE --epoch=$EPOCH_TEACHER --epoch_trans=$EPOCH_TRANS --distill_layer_str="[4,8,12]" --cuda
