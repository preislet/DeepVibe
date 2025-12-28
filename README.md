# DeepVibe
DeepVibe: Hybrid Variational Autoencoder (VAE) for personalized music discovery. Combines user listening history with Spotify audio features to generate a unique "Ideal DNA" profile. Features temperature-scaled inference, hidden gems mode, and cyclical KL annealing for high-precision, content-aware recommendations.

Here is a comprehensive and professionally structured `README.md` file for your **DeepVibe** project. It includes the architecture details, path configurations, and usage instructions we've developed.

---

# 🧬 DeepVibe: Hybrid VAE Music Discovery

**DeepVibe** is an advanced music recommendation engine that utilizes a **Hybrid Multi-VAE (Variational Autoencoder)** architecture. It goes beyond simple collaborative filtering by mapping user history and acoustic metadata into a unified latent "Vibe Space."

## 🏗 Model Architecture

The core of DeepVibe is a deep generative model designed to understand the "why" behind musical taste:

* **Dual Input:** The model processes a sparse **Interaction Vector** (10,000 track dimensions) alongside a dense **Audio Profile** (12 acoustic features like Energy, Valence, and Acousticness).
* **Bottleneck Latent Space:** An Encoder compresses these inputs into a 256-dimensional "Taste Extrakt," regularized via KL-Divergence to ensure a continuous and searchable musical space.
* **Linear/SELU Stack:** High-capacity layers with Scaled Exponential Linear Units (SELU) for self-normalizing training stability.

---

## 📂 Project Structure & Paths

To ensure the system functions correctly, maintain the following directory structure:

```text
DeepVibe/
├── data/
│   ├── full_training_matrix.csv  # User-track interactions (10k track columns)
│   └── top_10k_features.csv      # Audio features for the tracks in the matrix
├── models/
│   ├── deepvibeV2.py            # PyTorch model class definition
│   ├── hybrid_deepvibe_best.pth # Best performing model checkpoint
│   └── snapshots/               # Epoch-based model backups (Auto-saved)
├── logs/
│   ├── hybrid_training.log      # Structured CSV logs for plotting
│   └── training_detailed.log    # Detailed training events
├── recommendations/             # Generated discovery reports (Auto-created)
├── train.py                     # Main training script
└── inference.py           # Discovery search & DNA generation script

```

### Path Configuration

If your environment differs, update the constants at the top of `train.py` and `discovery_final.py`:

* **`INT_MATRIX_PATH`**: Path to your interaction CSV.
* **`FEATURES_PATH`**: Path to the 12-feature CSV for the core 10k tracks.
* **`full_db_path`**: Location of the full Spotify database (for the global Discovery scan).

---

## Usage

### 1. Training with Cyclical Annealing

The training script utilizes **Cyclical KL Annealing** to prevent latent space collapse and **Denoising** to force the model to learn acoustic relationships.

```bash
python train.py

```

### 2. High-Fidelity Discovery

Generate a unique **Vibe DNA** and search the global database for matches.

* **Standard Mode:** Finds the highest correlation matches for a user's current vibe.
* **Hidden Gems Mode:** Penalizes popularity to surface underground tracks with high acoustic similarity.

```bash
# Standard search
python inference.py user_1

# Find underground "Hidden Gems"
python inference.py user_1 --gems

```

---

## Optimization Strategies

* **Temperature-Scaled Inference:** We use  during DNA generation to "sharpen" the model's focus, ensuring the DNA represents the absolute peak of the user's preference.
* **Weighted MSE:** Positive interactions are weighted **40x** more than zeros to solve the "needle in a haystack" problem of sparse music data.
* **Automatic Reporting:** Results are saved to `recommendations/{user_id}/{user_id}_{epoch}_{mode}.csv`.

---
