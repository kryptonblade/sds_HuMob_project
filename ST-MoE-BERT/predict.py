import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geobleu import geobleu

from utils import *

def iterative_mobility_prediction(model, input_seq_feature, historical_locations, predict_seq_feature, input_seq_length, output_seq_length, device):
    batch_size, total_predict_length, _ = predict_seq_feature.shape
    
    # Initialize the output tensor to store all predictions
    all_predictions = torch.zeros(batch_size, total_predict_length, dtype=torch.long, device=device)
    
    # Initial input
    current_input_feature = input_seq_feature # (batch_size, input_seq_length, num_features)
    current_historical_locations = historical_locations
    assert current_input_feature.shape[1] == current_historical_locations.shape[1]
    
    for start_idx in range(0, total_predict_length, output_seq_length):
        end_idx = min(start_idx + output_seq_length, total_predict_length)
        current_predict_feature = predict_seq_feature[:, start_idx:end_idx]
        
        # Generate predictions for the current segment
        with torch.no_grad():
            # Handle both regular BERT and MoE models
            if hasattr(model, 'moe'):  # MoE model
                logits, _ = model(current_input_feature, current_historical_locations, current_predict_feature)
            else:  # Regular BERT model
                logits = model(current_input_feature, current_historical_locations, current_predict_feature)
            
            predictions = torch.argmax(logits, dim=-1)
            
                    
        # Store the predictions
        all_predictions[:, start_idx:end_idx] = predictions
        
        if end_idx < total_predict_length:
            # Calculate how much new data we're adding
            new_data_length = end_idx - start_idx
            
            if current_input_feature.shape[1] + new_data_length <= input_seq_length:
                current_input_feature = torch.cat([current_input_feature, 
                                                   predict_seq_feature[:, start_idx:end_idx]], dim=1)
                current_historical_locations = torch.cat([current_historical_locations, 
                                                          predictions], dim=1)
            else:
                # If it would exceed, slide the window
                retain_length = input_seq_length - new_data_length
                current_input_feature = torch.cat([current_input_feature[:, -retain_length:], 
                                                   predict_seq_feature[:, start_idx:end_idx]], dim=1)
                current_historical_locations = torch.cat([current_historical_locations[:, -retain_length:], 
                                                          predictions], dim=1)
            
            assert current_input_feature.shape[1] == current_historical_locations.shape[1], \
                f"Mismatch in shapes: input_feature {current_input_feature.shape}, historical_locations {current_historical_locations.shape}"
    
    return all_predictions

def mobility_generation_evaluation(model, predict_df, device, input_seq_length, output_seq_length, path=None):
    '''
    Evaluate the model on the prediction dataset
    '''
    from tqdm import tqdm
    
    geo_bleu_list = []
    dtw_list = []
    accuracy_list = []
    
    generates = []
    
    uid_list = predict_df['uid'].unique()
    print(f"Evaluating {len(uid_list)} users")
    
    for uid in tqdm(uid_list, desc="Evaluating users"):
        uid_df = predict_df[predict_df['uid'] == uid]
        # split first 60 days and last 15 days
        input_seq_df = uid_df[uid_df['d'] < 60].copy()
        predict_seq_df = uid_df[uid_df['d'] >= 60].copy()
        
        if len(input_seq_df) == 0 or len(predict_seq_df) == 0:
            continue
            
        input_seq_df['label'] = morton_encode(input_seq_df['x'] - 1, input_seq_df['y'] - 1)
        predict_seq_df['label'] = morton_encode(predict_seq_df['x'] - 1, predict_seq_df['y'] - 1)
        
        input_seq, location_seq = generate_sequence(input_seq_df)
        predict_seq, _ = generate_sequence(predict_seq_df)
        
                
        if len(input_seq) < input_seq_length or len(predict_seq) == 0:
            continue
            
        input_seq_feature = torch.tensor(input_seq[-input_seq_length:, :], dtype=torch.long, device=device).unsqueeze(0)
        historical_locations = torch.tensor(location_seq[-input_seq_length:], dtype=torch.long, device=device).unsqueeze(0)
        predict_seq_feature = torch.tensor(predict_seq, dtype=torch.long, device=device).unsqueeze(0)
        
        # generate predictions
        all_predictions = iterative_mobility_prediction(model, input_seq_feature, historical_locations, predict_seq_feature, input_seq_length, output_seq_length, device)[0]
        
        # Ensure predictions match the length of predict_seq_df
        if len(all_predictions) != len(predict_seq_df):
            # Trim or pad predictions to match
            min_len = min(len(all_predictions), len(predict_seq_df))
            all_predictions = all_predictions[:min_len]
            predict_seq_df = predict_seq_df.iloc[:min_len]
        
        predict_seq_df = predict_seq_df.copy()
        predict_seq_df['label'] = all_predictions.cpu().numpy()
        predict_seq_df = convert_label_back(predict_seq_df)
        
        # calculate BLEU and DTW loss
        generated = [(row['uid'], row['d'], row['t'], row['predict_x'], row['predict_y']) for _, row in predict_seq_df.iterrows()]
        reference = [(row['uid'], row['d'], row['t'], row['x'], row['y']) for _, row in predict_seq_df.iterrows()]
        
        if len(generated) > 0 and len(reference) > 0:
            geo_bleu, dtw, accuracy = calc_bleu_dtw_loss(generated, reference)
            geo_bleu_list.append(geo_bleu)
            dtw_list.append(dtw)
            accuracy_list.append(accuracy)
            generates.extend(generated)
        
    # save the generated data as csv
    if len(generates) > 0:
        generate_df = pd.DataFrame(generates, columns=['uid', 'd', 't', 'x', 'y'])
        if path is not None:
            generate_df.to_csv(path + f'/generate_{get_time_str()}.csv', index=False)
        
        geo_bleu_loss = sum(geo_bleu_list) / len(geo_bleu_list)
        dtw_loss = sum(dtw_list) / len(dtw_list)
        accuracy = sum(accuracy_list) / len(accuracy_list)
    else:
        geo_bleu_loss, dtw_loss, accuracy = 0.0, 0.0, 0.0
    
    return geo_bleu_loss, dtw_loss, accuracy

def generate_sequence(data):
    uid = data['uid'].values[0]
    # for each uid, generate a sequence of 75 days
    # Vectorized approach
    seq_x = []
    seq_y = []

    previous_d = data['d'].values[0]
    previous_t = data['t'].values[0]
    
    for _, row in data.iterrows():
        d = row['d']
        t = row['t']
        label = row['label']
        
        delta_t = (t - previous_t) + 48 * (d - previous_d)
        
        seq_x.append([d, t, uid, d % 7, t % 24, delta_t])
        seq_y.append(label)
        
        previous_d = d
        previous_t = t

    return np.array(seq_x), np.array(seq_y)