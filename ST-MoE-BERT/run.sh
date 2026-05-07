#!/bin/bash

# Setup for running training models for four different cities

# Ensure CUDA devices are correctly assigned (assuming a multi-GPU setup)
export CUDA_VISIBLE_DEVICES=0

# Define common parameters
EPOCHS=14
BASE_LR=0.001
MODEL_NAME="MobilityBERTMoE"
MODEL_PATH=""
LOCATION_EMBEDDING_LR=0.0005  # Higher learning rate for location embeddings

# Training for City A
CITY="A"
NUM_LOCATION_IDS=40001
HIDDEN_SIZE=128  # Reduced hidden size for memory
HIDDEN_LAYERS=6  # Further reduced layers for memory
ATTENTION_HEADS=4  # Further reduced attention heads
DAY_EMBED_SIZE=32  # Reduced embedding sizes
TIME_EMBED_SIZE=32
DAY_OF_WEEK_EMBED_SIZE=16
WEEKDAY_EMBED_SIZE=4
LOCATION_EMBED_SIZE=128  # Reduced location embedding size
DROPOUT=0.1  # Reduced dropout for better learning
MAX_SEQ_LENGTH=720  # Example length, adjust as needed
BATCH_SIZE=16  # Reduced batch size for memory

python main.py --model_name $MODEL_NAME \
               --hidden_size $HIDDEN_SIZE \
               --hidden_layers $HIDDEN_LAYERS \
               --attention_heads $ATTENTION_HEADS \
               --day_embedding_size $DAY_EMBED_SIZE \
               --time_embedding_size $TIME_EMBED_SIZE \
               --day_of_week_embedding_size $DAY_OF_WEEK_EMBED_SIZE \
               --weekday_embedding_size $WEEKDAY_EMBED_SIZE \
               --location_embedding_size $LOCATION_EMBED_SIZE \
               --dropout $DROPOUT \
               --max_seq_length $MAX_SEQ_LENGTH \
               --lr $BASE_LR \
               --location_embedding_lr $LOCATION_EMBEDDING_LR \
               --num_epochs $EPOCHS \
               --device "cuda" \
               --city $CITY \
               --batch_size $BATCH_SIZE \
               --subsample True \
               --subsample_number 1000


# # Fine-tune for City B,c,d
# CITY="D"
# NUM_LOCATION_IDS=40001  # Same as City A to avoid mismatch
# HIDDEN_SIZE=128  # Same as City A
# HIDDEN_LAYERS=6  # Same as City A
# ATTENTION_HEADS=4  # Same as City A
# DAY_EMBED_SIZE=32  # Same as City A
# TIME_EMBED_SIZE=32  # Same as City A
# DAY_OF_WEEK_EMBED_SIZE=16  # Same as City A
# WEEKDAY_EMBED_SIZE=4  # Same as City A
# LOCATION_EMBED_SIZE=128  # Same as City A
# DROPOUT=0.1  # Same as City A
# MAX_SEQ_LENGTH=3648  # Same as City A
# BATCH_SIZE=16  # Same as City A
# BASE_LR=0.0005  # Lower learning rate for fine-tuning
# LOCATION_EMBEDDING_LR=0.0002  # Lower learning rate for fine-tuning
# EPOCHS=10  # Fewer epochs for fine-tuning
# MODEL_PATH="checkpoints/best_mobility_bert_moe_dtw_94.21.pth"  # Load pretrained model

# python main.py --model_name $MODEL_NAME \
#                --model_path $MODEL_PATH \
#                --hidden_size $HIDDEN_SIZE \
#                --hidden_layers $HIDDEN_LAYERS \
#                --attention_heads $ATTENTION_HEADS \
#                --day_embedding_size $DAY_EMBED_SIZE \
#                --time_embedding_size $TIME_EMBED_SIZE \
#                --day_of_week_embedding_size $DAY_OF_WEEK_EMBED_SIZE \
#                --weekday_embedding_size $WEEKDAY_EMBED_SIZE \
#                --location_embedding_size $LOCATION_EMBED_SIZE \
#                --dropout $DROPOUT \
#                --max_seq_length $MAX_SEQ_LENGTH \
#                --lr $BASE_LR \
#                --location_embedding_lr $LOCATION_EMBEDDING_LR \
#                --num_epochs $EPOCHS \
#                --device "cuda" \
#                --city $CITY \
#                --batch_size $BATCH_SIZE \
#                --subsample True \
#                --subsample_number 1000
