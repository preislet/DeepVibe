import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
from models.deepvibeV2 import HybridDeepVibeVAE

# --- CONFIGURATION ---
# Paths for the interaction matrix and the audio feature database
INT_MATRIX_PATH = "./data/full_training_matrix.csv"
FEATURES_PATH = "./data/top_10k_features.csv"
LOG_PATH = "logs/hybrid_training.log"

# Ensure necessary directories exist for saving models and logs
os.makedirs("models/snapshots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

class HybridVibeDataset(Dataset):
    def __init__(self, interaction_csv, features_csv):
        # 1. Load the user-track interaction matrix
        df_int = pd.read_csv(interaction_csv)
        # Identify track columns (all columns except user labels and time metadata)
        self.track_cols = [c for c in df_int.columns if c not in ['user_label', 'week']]
        self.interactions = df_int[self.track_cols].values.astype('float32')
        self.weeks = df_int['week'].values
        
        # 2. Load audio features and align them with the track columns from the matrix
        df_feat = pd.read_csv(features_csv)
        df_feat = df_feat.set_index('spotify_track_uri').reindex(self.track_cols)
        
        # Define the specific audio attributes used for training
        audio_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness', 
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 
            'artist_popularity', 'track_popularity', 'explicit'
        ]
        # Store features as a tensor for efficient matrix multiplication
        self.track_features = df_feat[audio_cols].values.astype('float32')
        self.track_features_tensor = torch.from_numpy(self.track_features)
        print(f"Dataset loaded: {len(self.interactions)} samples, {len(self.track_cols)} features.")

    def __len__(self): 
        return len(self.interactions)

    def __getitem__(self, idx):
        # Get raw interactions for a single user/week
        user_int = torch.from_numpy(self.interactions[idx])
        
        # Calculate the "Current Audio Profile" (Weighted average of features of songs listened to)
        # This serves as the content-based input for the Encoder
        user_audio = (user_int.unsqueeze(0) @ self.track_features_tensor)
        user_audio = user_audio / (user_int.sum() + 1e-8)
        
        return user_int, user_audio.squeeze(0)

def calculate_recall_at_k(model, data_int, data_audio, k_list=[20, 100]):
    """Evaluates how many relevant items the model ranks in the top K positions."""
    model.eval()
    results = {}
    with torch.no_grad():
        # Generate model predictions (reconstructions)
        recon, _, _ = model(data_int, data_audio)
        actual_binary = (data_int > 0).float()
        actual_counts = torch.clamp(actual_binary.sum(dim=1), min=1.0)
        
        for k in k_list:
            # Identify the indices of the top K predicted tracks
            _, topk_indices = torch.topk(recon, k, dim=1)
            # Create a binary mask for top K predictions
            pred_binary = torch.zeros_like(recon).scatter_(1, topk_indices, 1.0)
            # Count hits: where the prediction mask matches the actual binary ground truth
            hits = (pred_binary * actual_binary).sum(dim=1)
            # Calculate average recall for the batch
            recall = (hits / actual_counts).mean().item()
            results[k] = recall
    return results

def loss_function(recon_x, x, mu, logvar, beta, pos_weight=40.0):
    """Hybrid VAE Loss: Weighted Reconstruction Error + KL Divergence."""
    # Assign higher weight to positive interactions (songs actually played) 
    # to combat the sparsity of the 10,000 track matrix
    weights = torch.ones_like(x)
    weights[x > 0] = pos_weight
    
    # Reconstruction Loss (MSE weighted by song presence)
    recon_loss = (weights * (recon_x - x)**2).sum()
    
    # KL Divergence: Regularizes the latent space to follow a normal distribution
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + (beta * kld_loss), recon_loss, kld_loss

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HybridVibeDataset(INT_MATRIX_PATH, FEATURES_PATH)
    # High worker count for fast data loading during batching
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=100)

    # 1. Prepare Validation Data
    # We take a subset of real interactions (where week > 0) to monitor generalization
    real_indices = np.where(dataset.weeks > 0)[0]
    val_indices = np.random.choice(real_indices, min(2000, len(real_indices)), replace=False)
    val_int = torch.from_numpy(dataset.interactions[val_indices]).to(device)
    # Pre-calculate validation audio profiles
    val_audio = (val_int.cpu().unsqueeze(1) @ dataset.track_features_tensor).squeeze(1)
    val_audio = (val_audio / (val_int.cpu().sum(dim=1, keepdim=True) + 1e-8)).to(device)

    # 2. Model, Optimizer, and Learning Rate Scheduler initialization
    input_dim = len(dataset.track_cols)
    model = HybridDeepVibeVAE(interaction_dim=input_dim, audio_dim=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Reduces learning rate when the loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_r20 = 0.0
    print(f"Starting training on {device}...")

    # 3. Training Loop
    for epoch in range(300):
        model.train()
        total_loss, total_recon, total_kld = 0, 0, 0
        
        # Cyclical KL Annealing: Gradually increase Beta to balance reconstruction and regularization
        cycle_len = 40
        beta_max = 0.001
        beta = beta_max * ( (epoch % cycle_len) / (cycle_len * 0.5) )
        beta = min(beta_max, beta)
        
        # Visual progress bar for batches inside the epoch
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}", leave=False)
        
        for x_int, x_audio in pbar:
            x_int, x_audio = x_int.to(device), x_audio.to(device)
            
            # --- Denoising Step ---
            # Randomly hide 10% of the input songs. 
            # This forces the model to use the "Audio Profile" to infer the missing history.
            noise_mask = (torch.rand_like(x_int) > 0.1).float()
            x_int_noisy = x_int * noise_mask
            
            optimizer.zero_grad()
            
            # Forward pass: noisy interactions + clean audio profile
            recon, mu, logvar = model(x_int_noisy, x_audio) 
            
            # Calculate Loss against the FULL (un-noised) original interactions
            loss, r_loss, k_loss = loss_function(recon, x_int, mu, logvar, beta)
            
            loss.backward()
            # Gradient clipping to ensure training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += r_loss.item()
            total_kld += k_loss.item()
            pbar.set_postfix({"loss": f"{loss.item()/len(x_int):.4f}"})

        # Calculate averages for logging
        avg_loss = total_loss / len(dataset)
        avg_recon = total_recon / len(dataset)
        avg_kld = total_kld / len(dataset)
        
        # Update LR scheduler based on current epoch loss
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Format epoch output string
        output_str = f"Epoch {epoch+1:3d}/300 | Loss: {avg_loss:.4f} (Rec: {avg_recon:.4f}, KLD: {avg_kld:.4f}) | Beta: {beta:.5f} | LR: {current_lr:.5f}"

        # 4. Periodic Validation and Checkpointing
        if (epoch + 1) % 10 == 0:
            recalls = calculate_recall_at_k(model, val_int, val_audio, k_list=[20, 100])
            r20, r100 = recalls[20], recalls[100]
            
            output_str += f" | R@20: {r20:.4f} | R@100: {r100:.4f}"
            
            # Save a snapshot for historical reference
            snapshot_path = f"models/snapshots/hybrid_epoch_{epoch+1}.pth"
            save_dict = {
                'model_state': model.state_dict(),
                'track_cols': dataset.track_cols,
                'audio_features': dataset.track_features,
                'recall_20': r20,
                'epoch': epoch+1
            }
            torch.save(save_dict, snapshot_path)
            
            # Save the globally best performing model
            if r20 > best_r20:
                best_r20 = r20
                torch.save(save_dict, "models/hybrid_deepvibe_best.pth")
                output_str += " 🌟 BEST"

        # Print the final result of the epoch to the console
        print(output_str)

    print("Training finished.")

if __name__ == "__main__":
    run_training()