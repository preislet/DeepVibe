import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from models.deepvibeV2 import HybridDeepVibeVAE
import sys
import argparse
from tqdm import tqdm  # Import tqdm pro progress bar
import os

def discovery_search(user_id, 
                     hidden_gems=False,
                     model_path="models/hybrid_deepvibe_best.pth", 
                     full_db_path="/work/preislet/Spotify/processed/track_features_29.csv",
                     training_matrix_path="./data/full_training_matrix.csv"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_name = model_path.split('_')[-1].split('.')[0] if 'epoch' in model_path else 'best'
    
    # --- 1. LOAD MODEL ---
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    track_uris = np.array(checkpoint['track_cols'])
    track_features_matrix = torch.from_numpy(checkpoint['audio_features']).to(device)
    
    model = HybridDeepVibeVAE(interaction_dim=10000, audio_dim=12).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # --- 2. GENERATE UNIQUE DNA ---
    df_int = pd.read_csv(training_matrix_path)
    user_data = df_int[df_int['user_label'] == user_id]
    if user_data.empty:
        print(f"❌ User {user_id} not found.")
        return

    latest_row = user_data.sort_values(by='week', ascending=False).iloc[0]
    user_vec = latest_row[checkpoint['track_cols']].values.astype('float32')
    user_tensor = torch.from_numpy(user_vec).unsqueeze(0).to(device)
    
    with torch.no_grad():
        user_audio = (user_tensor @ track_features_matrix) / (user_tensor.sum() + 1e-8)
        recon, _, _ = model(user_tensor, user_audio)
        
        # Sharpening with Temperature
        temp_recon = recon / 0.05
        top_v, top_i = torch.topk(temp_recon, 50)
        mask = torch.full_like(temp_recon, float('-inf'))
        mask.scatter_(1, top_i, top_v)
        
        weights = torch.softmax(mask, dim=1)
        ideal_dna = (weights @ track_features_matrix).cpu().numpy().flatten()

    # --- 3. DNA VISUALIZATION ---
    feature_names = [
        'acousticness', 'danceability', 'energy', 'instrumentalness', 
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 
        'artist_popularity', 'track_popularity', 'explicit'
    ]

    print("\n" + "="*50)
    mode_title = "HIDDEN GEMS MODE" if hidden_gems else "STANDARD DISCOVERY"
    print(f"{mode_title} FOR: {user_id}")
    print("="*50)
    for name, val in zip(feature_names, ideal_dna):
        bar_length = int(val * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{name:18s} | {val:.4f} | {bar}")
    print("="*50 + "\n")

    # --- 4. SCAN FULL DATABASE ---
    print(f"Loading full database...")
    full_db = pd.read_csv(full_db_path).dropna(subset=feature_names).drop_duplicates('spotify_track_uri')
    
    # --- HARD POPULARITY FILTER FOR HIDDEN GEMS ---
    if hidden_gems:
        # Ponecháme pouze tracky, které nejsou mainstream (pop < 50)
        print("Filtering for underground tracks (popularity < 50)...")
        full_db = full_db[full_db['track_popularity'] < 50].copy()

    # Příprava matice pro výpočet (využijeme tqdm pro vizuální zpětnou vazbu)
    print(f"Preparing feature matrix for {len(full_db)} tracks...")
    db_matrix = full_db[feature_names].values
    
    # Normalizace
    scaler = MinMaxScaler()
    db_matrix_scaled = scaler.fit_transform(db_matrix)
    
    # Výpočet podobnosti s tqdm
    print("Calculating vibe similarity scores...")
    # Protože cosine_similarity na velkých maticích je rychlá, 
    # rozdělíme ji na bloky, aby tqdm dávalo smysl u obřích DB
    batch_size = 100000
    similarities = []
    
    for i in tqdm(range(0, len(db_matrix_scaled), batch_size), desc="Scanning Vibe"):
        batch = db_matrix_scaled[i:i+batch_size]
        sim_batch = cosine_similarity([ideal_dna], batch).flatten()
        similarities.extend(sim_batch)
    
    full_db['similarity'] = similarities
    
    # --- SCORING LOGIC ---
    if hidden_gems:
        # V režimu Gems penalizujeme i zbytek popularity, abychom podpořili úplné "nuly"
        pop_penalty = (full_db['track_popularity'] / 100.0) * 0.2
        full_db['vibe_score'] = full_db['similarity'] - pop_penalty
    else:
        full_db['vibe_score'] = full_db['similarity']
    
    # Filter out history
    history_uris = set(track_uris[user_vec > 0.05])
    recommendations = full_db[~full_db['spotify_track_uri'].isin(history_uris)]
    top_export = recommendations.sort_values('vibe_score', ascending=False).head(1000)
    top_discoveries = top_export.head(10)

    output_dir = f"recommendations/{user_id}"
    os.makedirs(output_dir, exist_ok=True)

    mode_tag = "gems" if hidden_gems else "standard"
    filename = f"{user_id}_{epoch_name}_{mode_tag}.csv"
    save_path = os.path.join(output_dir, filename)

    top_export['spotify_url'] = top_export['spotify_track_uri'].apply(lambda x: f"https://open.spotify.com/track/{x.split(':')[-1]}")
    top_export.to_csv(save_path, index=False)

    
    
    # --- 5. PRINT RECOMMENDATIONS ---
    print("\n" + "="*90)
    print(f"TOP DISCOVERIES FOR: {user_id}")
    print("="*90)
    
    for i, (_, row) in enumerate(top_discoveries.iterrows()):
        track_id = row['spotify_track_uri'].split(':')[-1]
        spotify_url = f"https://open.spotify.com/track/{track_id}"
        
        print(f"{i+1:2d}. {row['track_name']} — {row['artist_name']}")
        print(f"    [Score: {row['vibe_score']:.4f} | Sim: {row['similarity']:.4f} | Pop: {row['track_popularity']}]")
        print(f"    🔗 {spotify_url}\n")
    
    print("="*90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepVibe Discovery Search')
    parser.add_argument('user_id', type=str, help='ID of the user')
    parser.add_argument('--gems', action='store_true', help='Enable Hidden Gems mode (penalize popularity)')
    parser.add_argument('--model', type=str, default="models/snapshots/hybrid_epoch_60.pth", help='Path to model')
    
    args = parser.parse_args()
    
    discovery_search(args.user_id, hidden_gems=args.gems, model_path=args.model)