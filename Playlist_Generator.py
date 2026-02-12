import os
import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mutagen import File
#from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================
MUSIC_DIR = r"\\TOWER\Music-Mark"          # Path to your music folder
OUTPUT_DIR = r"C:\Users\mark\Desktop"    # Where playlists will be saved
NUM_PLAYLISTS = 6           # Number of playlists to generate
SUPPORTED_EXTS = (".mp3", ".flac", ".wav", ".m4a")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_features(filepath):
    y, sr = librosa.load(filepath, mono=True, duration=60)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.squeeze(tempo))  # <-- FIX

    rms = float(np.mean(librosa.feature.rms(y=y)))
    brightness = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zero_crossing = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    # Ensure everything is 1D floats
    feature_vector = np.array(
        [tempo, rms, brightness, zero_crossing, *mfcc_means],
        dtype=np.float32
    )

    return feature_vector, tempo, rms, brightness

def get_metadata(filepath):
    audio = File(filepath, easy=True)
    if not audio:
        return {"title": "Unknown", "artist": "Unknown", "genre": "Unknown"}

    return {
        "title": audio.get("title", ["Unknown"])[0],
        "artist": audio.get("artist", ["Unknown"])[0],
        "genre": audio.get("genre", ["Unknown"])[0],
    }

# ==========================
# PLAYLIST NAMING AI
# ==========================
def name_playlist(cluster_df):
    avg_tempo = cluster_df["tempo"].mean()
    avg_energy = cluster_df["rms"].mean()
    avg_brightness = cluster_df["brightness"].mean()

    if avg_tempo < 90:
        tempo_label = "Slow"
    elif avg_tempo < 120:
        tempo_label = "Midtempo"
    else:
        tempo_label = "Fast"

    if avg_energy < 0.02:
        energy_label = "Chill"
    elif avg_energy < 0.05:
        energy_label = "Balanced"
    else:
        energy_label = "Energetic"

    tone_label = "Warm" if avg_brightness < 2000 else "Bright"

    return f"{energy_label} {tempo_label} ({tone_label})"

# ==========================
# SCAN MUSIC LIBRARY
# ==========================
tracks = []
features = []

print("Scanning music library...")

for root, _, files in os.walk(MUSIC_DIR):
    for file in files:
        if file.lower().endswith(SUPPORTED_EXTS):
            path = os.path.join(root, file)
            #relative_path=path.replace('\\TOWER\Music-Mark\\',"")
            relative_path = os.path.relpath(path,MUSIC_DIR)
            try:
                feat, tempo, rms, brightness = extract_features(path)
                meta = get_metadata(path)

                tracks.append({
                    "relative_path":relative_path,
                    "path": path,
                    "tempo": tempo,
                    "rms": rms,
                    "brightness": brightness,
                    **meta
                })
                features.append(feat)
            except Exception as e:
                print(f"Skipped {file}: {e}")

if not tracks:
    raise RuntimeError("No supported music files found.")

# ==========================
# CLUSTERING (AI)
# ==========================
X = np.array(features, dtype=np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=NUM_PLAYLISTS,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

df = pd.DataFrame(tracks)
df["playlist"] = labels

# ==========================
# CREATE PLAYLIST FILES
# ==========================
print("\nCreating playlists...\n")

for playlist_id in sorted(df["playlist"].unique()):
    subset = df[df["playlist"] == playlist_id]
    playlist_name = name_playlist(subset)

    safe_name = playlist_name.replace(" ", "_")
    playlist_path = os.path.join(OUTPUT_DIR, f"{safe_name} — {len(subset)} tracks.m3u")

    with open(playlist_path, "w", encoding="utf-8") as f:
        for track_path in subset["relative_path"]:
            f.write(track_path + "\n")

    print(f"Created: {playlist_name} — {len(subset)} tracks")

print("\nAll playlists generated successfully 🎵")
