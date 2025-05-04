# 1_data_exploration.py
import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define paths with the correct directory structure
TRAIN_DATA_DIR = "/Users/priyanshuwalia7/Downloads/Dataset/audios/train"
TEST_DATA_DIR = "/Users/priyanshuwalia7/Downloads/Dataset/audios/test"
TRAIN_CSV = "/Users/priyanshuwalia7/Downloads/Dataset/train.csv"
TEST_CSV = "/Users/priyanshuwalia7/Downloads/Dataset/test.csv"
OUTPUT_DIR = "exploration_output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Total training audio files: {len(train_df)}")
print(f"Total test audio files: {len(test_df)}")

# Display training data information
print("\nTraining Data Preview:")
print(train_df.head())
print("\nTraining Data Columns:")
print(train_df.columns.tolist())
print("\nTraining Data Statistics:")
print(train_df.describe())

# Function to explore audio file
def explore_audio_file(file_path, output_prefix):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        print(f"Shape: {y.shape}")
        
        # Plot waveform
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, duration, len(y)), y)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_waveform_{os.path.basename(file_path).split('.')[0]}.png"))
        plt.close()
        
        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_spectrogram_{os.path.basename(file_path).split('.')[0]}.png"))
        plt.close()
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'mean_amplitude': np.mean(np.abs(y)),
            'std_amplitude': np.std(y),
            'max_amplitude': np.max(np.abs(y)),
            'zero_crossings': np.sum(librosa.zero_crossings(y))
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Check a few training files
print("\nExploring training audio files:")
train_stats = []
for idx, row in train_df.head(5).iterrows():
    file_path = os.path.join(TRAIN_DATA_DIR, row['filename'])
    if os.path.exists(file_path):
        stats = explore_audio_file(file_path, "train")
        if stats:
            stats['filename'] = row['filename']
            if 'label' in row:
                stats['label'] = row['label']
            train_stats.append(stats)
    else:
        print(f"File not found: {file_path}")

# Check a few test files
print("\nExploring test audio files:")
test_stats = []
for idx, row in test_df.head(5).iterrows():
    file_path = os.path.join(TEST_DATA_DIR, row['filename'])
    if os.path.exists(file_path):
        stats = explore_audio_file(file_path, "test")
        if stats:
            stats['filename'] = row['filename']
            test_stats.append(stats)
    else:
        print(f"File not found: {file_path}")

# Check for missing files in training set
missing_train_files = []
for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Checking training files"):
    file_path = os.path.join(TRAIN_DATA_DIR, row['filename'])
    if not os.path.exists(file_path):
        missing_train_files.append(row['filename'])

if missing_train_files:
    print(f"\nMissing {len(missing_train_files)} training files:")
    for f in missing_train_files[:10]:  # Show first 10
        print(f"  - {f}")
    if len(missing_train_files) > 10:
        print(f"  ... and {len(missing_train_files) - 10} more")
else:
    print("\nAll training files found!")

# Check for missing files in test set
missing_test_files = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Checking test files"):
    file_path = os.path.join(TEST_DATA_DIR, row['filename'])
    if not os.path.exists(file_path):
        missing_test_files.append(row['filename'])

if missing_test_files:
    print(f"\nMissing {len(missing_test_files)} test files:")
    for f in missing_test_files[:10]:  # Show first 10
        print(f"  - {f}")
    if len(missing_test_files) > 10:
        print(f"  ... and {len(missing_test_files) - 10} more")
else:
    print("\nAll test files found!")

# Create stats dataframes
if train_stats:
    train_stats_df = pd.DataFrame(train_stats)
    print("\nTraining Audio Statistics:")
    print(train_stats_df.describe())
    train_stats_df.to_csv(os.path.join(OUTPUT_DIR, "train_audio_stats.csv"), index=False)

if test_stats:
    test_stats_df = pd.DataFrame(test_stats)
    print("\nTest Audio Statistics:")
    print(test_stats_df.describe())
    test_stats_df.to_csv(os.path.join(OUTPUT_DIR, "test_audio_stats.csv"), index=False)

# If there's a label column in training data, analyze its distribution
if 'label' in train_df.columns:
    plt.figure(figsize=(10, 6))
    train_df['label'].hist(bins=20)
    plt.title('Distribution of Grammar Scores')
    plt.xlabel('Grammar Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_DIR, "grammar_score_distribution.png"))
    plt.close()
    
    print("\nLabel Distribution Statistics:")
    print(train_df['label'].describe())