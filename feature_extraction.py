# 2_feature_extraction.py
import os
import pandas as pd
import numpy as np
import librosa
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Define paths with the correct directory structure
TRAIN_DATA_DIR = "/Users/priyanshuwalia7/Downloads/Dataset/audios/train"
TEST_DATA_DIR = "/Users/priyanshuwalia7/Downloads/Dataset/audios/test"
TRAIN_CSV = "/Users/priyanshuwalia7/Downloads/Dataset/train.csv"
TEST_CSV = "/Users/priyanshuwalia7/Downloads/Dataset/test.csv"
OUTPUT_DIR = "model_artifacts"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output files
TRAIN_FEATURES_FILE = os.path.join(OUTPUT_DIR, "train_audio_features.pkl")
TEST_FEATURES_FILE = os.path.join(OUTPUT_DIR, "test_audio_features.pkl")

# Function to extract features from an audio file
def extract_audio_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zero_crossings = librosa.zero_crossings(y).sum()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        
        # Speech features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1)
        mfcc_vars = mfccs.var(axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        
        # Rhythm features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = chroma.mean(axis=1)
        
        # Energy features
        rms = librosa.feature.rms(y=y).mean()
        
        # Speech rate approximation (syllable-like energy peaks)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        estimated_speech_rate = len(peaks) / duration if duration > 0 else 0
        
        # Pauses and silences
        non_silent_regions = librosa.effects.split(y, top_db=30)
        num_silent_regions = len(non_silent_regions) - 1 if len(non_silent_regions) > 0 else 0
        total_silence_duration = duration - sum([(end - start) / sr for start, end in non_silent_regions]) if len(non_silent_regions) > 0 else 0
        silence_ratio = total_silence_duration / duration if duration > 0 else 0
        
        # Fluency metrics
        if len(non_silent_regions) > 1:
            speech_chunks = [y[start:end] for start, end in non_silent_regions]
            chunk_durations = [(end - start) / sr for start, end in non_silent_regions]
            mean_chunk_duration = np.mean(chunk_durations)
            std_chunk_duration = np.std(chunk_durations)
            pause_durations = []
            for i in range(len(non_silent_regions) - 1):
                pause_start = non_silent_regions[i][1]
                pause_end = non_silent_regions[i + 1][0]
                pause_durations.append((pause_end - pause_start) / sr)
            mean_pause_duration = np.mean(pause_durations) if pause_durations else 0
            std_pause_duration = np.std(pause_durations) if pause_durations else 0
        else:
            mean_chunk_duration = 0
            std_chunk_duration = 0
            mean_pause_duration = 0
            std_pause_duration = 0
        
        # Prosodic features (pitch contour statistics)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_max_indices = np.argmax(magnitudes, axis=0)
        pitches_max = np.array([pitches[i, t] for t, i in enumerate(pitch_max_indices)])
        pitches_max = pitches_max[pitches_max > 0]  # Remove zero pitches
        if len(pitches_max) > 0:
            pitch_mean = np.mean(pitches_max)
            pitch_std = np.std(pitches_max)
            pitch_min = np.min(pitches_max)
            pitch_max = np.max(pitches_max)
            pitch_range = pitch_max - pitch_min
        else:
            pitch_mean = 0
            pitch_std = 0
            pitch_min = 0
            pitch_max = 0
            pitch_range = 0
            
        # Features dictionary
        features = {
            'duration': duration,
            'tempo': tempo,
            'zero_crossings': zero_crossings,
            'zero_crossing_rate': zero_crossing_rate,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'rms_energy': rms,
            'estimated_speech_rate': estimated_speech_rate,
            'num_silent_regions': num_silent_regions,
            'silence_ratio': silence_ratio,
            'mean_chunk_duration': mean_chunk_duration,
            'std_chunk_duration': std_chunk_duration,
            'mean_pause_duration': mean_pause_duration,
            'std_pause_duration': std_pause_duration,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
        }
        
        # Add MFCCs
        for i, mfcc_mean in enumerate(mfcc_means):
            features[f'mfcc{i+1}_mean'] = mfcc_mean
        for i, mfcc_var in enumerate(mfcc_vars):
            features[f'mfcc{i+1}_var'] = mfcc_var
            
        # Add chroma features
        for i, chroma_mean in enumerate(chroma_means):
            features[f'chroma{i+1}_mean'] = chroma_mean
            
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # Load training and test data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"Processing {len(train_df)} training files and {len(test_df)} test files")
    
    # Extract features for training files
    train_features_dict = {}
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting training features"):
        file_path = os.path.join(TRAIN_DATA_DIR, row['filename'])
        if os.path.exists(file_path):
            features = extract_audio_features(file_path)
            if features:
                train_features_dict[row['filename']] = features
    
    print(f"Successfully extracted features for {len(train_features_dict)} training files")
    
    # Extract features for test files
    test_features_dict = {}
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting test features"):
        file_path = os.path.join(TEST_DATA_DIR, row['filename'])
        if os.path.exists(file_path):
            features = extract_audio_features(file_path)
            if features:
                test_features_dict[row['filename']] = features
    
    print(f"Successfully extracted features for {len(test_features_dict)} test files")
    
    # Save features to pickle files
    with open(TRAIN_FEATURES_FILE, 'wb') as f:
        pickle.dump(train_features_dict, f)
    
    with open(TEST_FEATURES_FILE, 'wb') as f:
        pickle.dump(test_features_dict, f)
    
    print(f"Training features saved to {TRAIN_FEATURES_FILE}")
    print(f"Test features saved to {TEST_FEATURES_FILE}")
    
    # Create dataframes with features for inspection
    train_features_list = []
    for filename, feats in train_features_dict.items():
        feat_dict = {'filename': filename}
        feat_dict.update(feats)
        # Add label if available
        label_row = train_df[train_df['filename'] == filename]
        if 'label' in label_row.columns and not label_row.empty:
            feat_dict['label'] = label_row['label'].values[0]
        train_features_list.append(feat_dict)
    
    test_features_list = []
    for filename, feats in test_features_dict.items():
        feat_dict = {'filename': filename}
        feat_dict.update(feats)
        test_features_list.append(feat_dict)
    
    train_features_df = pd.DataFrame(train_features_list)
    test_features_df = pd.DataFrame(test_features_list)
    
    # Save to CSV for inspection
    train_features_df.to_csv(os.path.join(OUTPUT_DIR, "train_audio_features_table.csv"), index=False)
    test_features_df.to_csv(os.path.join(OUTPUT_DIR, "test_audio_features_table.csv"), index=False)
    
    print("Feature tables saved for inspection")
    
    # Print feature statistics
    print("\nTraining Features Statistics:")
    print(train_features_df.describe())
    
if 'label' in train_features_df.columns:
    print("\nTop 10 features correlated with grammar score:")
    # Select only numeric columns for correlation
    numeric_cols = train_features_df.select_dtypes(include=['number']).columns
    # Ensure 'label' is included but 'filename' is excluded
    if 'label' not in numeric_cols:
        print("Warning: 'label' column is not numeric. Check your data.")
    else:
        correlations = train_features_df[numeric_cols].corr()['label'].sort_values(ascending=False)
        print(correlations[1:11])