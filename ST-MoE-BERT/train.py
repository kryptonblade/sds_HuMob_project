import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import os
from model import MobilityBERT, MobilityBERTMoE
from predict import mobility_generation_evaluation
from data_loader import train_test_generate_mob_time_series_dataloader

PAD_LOCATION = 61504
save_dir = 'checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def load_model(model, model_path, device):
    if model_path is not None and model_path != "":
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    return model

def train_step(model, optimizer, criterion, input_seq_feature, historical_locations, predict_seq_feature, future_locations, device):
    model.train()
    optimizer.zero_grad()
    
    input_seq_feature, historical_locations, predict_seq_feature, future_locations = [b.to(device) for b in [input_seq_feature, historical_locations, predict_seq_feature, future_locations]]
    
    # Handle both regular BERT and MoE models
    if hasattr(model, 'moe'):  # MoE model
        logits, aux_loss = model(input_seq_feature, historical_locations, predict_seq_feature)
        main_loss = criterion(logits.view(-1, logits.size(-1)), future_locations.view(-1))
        # Combine main loss with auxiliary loss (weighted)
        loss = main_loss + 0.01 * aux_loss  # Small weight for auxiliary loss
    else:  # Regular BERT model
        logits = model(input_seq_feature, historical_locations, predict_seq_feature)
        loss = criterion(logits.view(-1, logits.size(-1)), future_locations.view(-1))
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy for monitoring
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        mask = future_locations != PAD_LOCATION  # Only count non-padded tokens
        correct = (predictions == future_locations) & mask
        accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else 0.0
    
    return loss.item(), accuracy.item()

def configure_optimizer(model, base_lr, location_embedding_lr):
    # Group parameters to apply different learning rates
    base_params = [p for n, p in model.named_parameters() if "location_embedding" not in n]
    location_embedding_params = [p for n, p in model.named_parameters() if "location_embedding" in n]
    if location_embedding_lr is None:
        location_embedding_lr = base_lr  # Use the base learning rate if none is provided for location embedding
    optimizer = torch.optim.AdamW([
        {'params': base_params},
        {'params': location_embedding_params, 'lr': location_embedding_lr}
    ], lr=base_lr, weight_decay=0.01)
    return optimizer

def configure_model(model_name, num_location_ids, hidden_size, hidden_layers, attention_heads, day_embedding_size, time_embedding_size, day_of_week_embedding_size, weekday_embedding_size, location_embedding_size, dropout, max_seq_length, device):
    if model_name == 'MobilityBERT':
        model = MobilityBERT(num_location_ids, hidden_size, hidden_layers, attention_heads, day_embedding_size, time_embedding_size, day_of_week_embedding_size, weekday_embedding_size, location_embedding_size, dropout, max_seq_length)
    elif model_name == 'MobilityBERTMoE':
        model = MobilityBERTMoE(num_location_ids, hidden_size, hidden_layers, attention_heads, day_embedding_size, time_embedding_size, day_of_week_embedding_size, weekday_embedding_size, location_embedding_size, dropout, max_seq_length)
    model.to(device)
    return model

def train(model, optimizer, train_loader, num_epochs, device, test_df, input_seq_length, predict_seq_length):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LOCATION)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps // 10, num_training_steps=num_training_steps)

    best_dtw = float('inf')  # Initialize with infinity (lower DTW is better)
    best_model_path = None
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            loss, accuracy = train_step(model, optimizer, criterion, *batch, device)
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / num_batches
        print(f"Epoch {epoch+1}:")
        print(f"Loss: {avg_loss:.4f}")
        
        print("Evaluating on users:")
        geobleu_loss, dtw_loss, test_accuracy = test(model, test_df, device, input_seq_length, predict_seq_length)
        print(f"GeoBLEU: {geobleu_loss:.4f}")
        print(f"DTW: {dtw_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Save best model based on DTW (lower is better)
        if dtw_loss < best_dtw:
            best_dtw = dtw_loss
            new_model_path = f'{save_dir}/best_mobility_bert_moe_dtw_{best_dtw:.2f}.pth'
            
            # Remove previous best model if it exists
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            torch.save(model.state_dict(), new_model_path)
            best_model_path = new_model_path
            print(f"Storing as pretrained model: {new_model_path}")
        else:
            print(f"Current DTW: {dtw_loss:.4f}, Best DTW: {best_dtw:.4f} - Model not saved")

def test(model, test_df, device, input_seq_length, predict_seq_length):
    model.eval()
    with torch.no_grad():
        geobleu_loss, dtw_loss, accuracy = mobility_generation_evaluation(model, test_df, device, input_seq_length, predict_seq_length)
    
    return geobleu_loss, dtw_loss, accuracy

def prepare_data_loader(city, input_seq_length, predict_seq_length, batch_size, subsample, random_seed, subsample_number, test_size, look_back_len):
    return train_test_generate_mob_time_series_dataloader(city, input_seq_length, predict_seq_length, subsample, subsample_number, test_size, batch_size, random_seed, look_back_len)
