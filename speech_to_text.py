import os
import pandas as pd
import pickle
from tqdm import tqdm
import speech_recognition as sr
import numpy as np
import librosa

# Define paths with the correct directory structure
TRAIN_DATA_DIR = "/Users/priyanshuwalia7/Downloads/Dataset/audios/train"
TEST_DATA_DIR = "/Users/priyanshuwalia7/Downloads/Dataset/audios/test"
TRAIN_CSV = "/Users/priyanshuwalia7/Downloads/Dataset/train.csv"
TEST_CSV = "/Users/priyanshuwalia7/Downloads/Dataset/test.csv"
OUTPUT_DIR = "model_artifacts"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output files
TRAIN_TRANSCRIPTS_FILE = os.path.join(OUTPUT_DIR, "train_audio_transcripts.pkl")
TEST_TRANSCRIPTS_FILE = os.path.join(OUTPUT_DIR, "test_audio_transcripts.pkl")

# Function to transcribe audio - optimized version
def transcribe_audio(file_path):
    try:
        # Load audio with librosa (using a lower sample rate for faster processing)
        y, sr_rate = librosa.load(file_path, sr=16000)  # Using a lower sample rate
        
        # Create a more reasonable chunk size for faster processing
        chunk_size = 5 * sr_rate  # 5-second chunks instead of 10 seconds
        chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
        
        recognizer = sr.Recognizer()
        full_transcript = ""
        
        # Only process the first 3 chunks to speed things up (or fewer if the file is shorter)
        process_chunks = chunks[:min(3, len(chunks))]
        
        for chunk in process_chunks:
            audio_segment = sr.AudioData(chunk.tobytes(), sr_rate, 2)
            try:
                # Only try Google's speech recognition
                chunk_transcript = recognizer.recognize_google(audio_segment)
                full_transcript += " " + chunk_transcript
            except:
                # Skip if recognition fails
                pass
                
        full_transcript = full_transcript.strip()
        
        # Extract minimal text features
        text_features = {}
        if full_transcript:
            # Just basic stats
            text_features['word_count'] = len(full_transcript.split())
            text_features['char_count'] = len(full_transcript)
            text_features['avg_word_length'] = text_features['char_count'] / max(1, text_features['word_count'])
            
            # Sentence count approximation
            sentence_count = full_transcript.count('.') + full_transcript.count('!') + full_transcript.count('?')
            text_features['sentence_count'] = max(1, sentence_count)  # Ensure at least 1 sentence
            text_features['avg_words_per_sentence'] = text_features['word_count'] / text_features['sentence_count']
        
        return {
            'transcript': full_transcript,
            'text_features': text_features
        }
    
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return {
            'transcript': "",
            'text_features': {}
        }

# Function to process a subset of files
def process_subset(file_list, data_dir, max_files=None):
    results = {}
    # Limit the number of files to process if specified
    if max_files and max_files > 0:
        file_list = file_list[:max_files]
        
    for filename in tqdm(file_list, desc=f"Processing {data_dir} files"):
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            result = transcribe_audio(file_path)
            results[filename] = result
    
    return results

# Main execution
if __name__ == "__main__":
    # Load training and test data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Set the maximum number of files to process in each set
    # Adjust these numbers based on your dataset size and available time
    MAX_TRAIN_FILES = 100  # Limit to 100 training files
    MAX_TEST_FILES = 50    # Limit to 50 test files
    
    print(f"Processing up to {MAX_TRAIN_FILES} training files and {MAX_TEST_FILES} test files")
    
    # Process a subset of training files
    train_files = train_df['filename'].tolist()
    train_transcripts_dict = process_subset(train_files, TRAIN_DATA_DIR, MAX_TRAIN_FILES)
    
    print(f"Successfully transcribed {len(train_transcripts_dict)} training files")
    
    # Process a subset of test files
    test_files = test_df['filename'].tolist()
    test_transcripts_dict = process_subset(test_files, TEST_DATA_DIR, MAX_TEST_FILES)
    
    print(f"Successfully transcribed {len(test_transcripts_dict)} test files")
    
    # Save transcripts to pickle files
    with open(TRAIN_TRANSCRIPTS_FILE, 'wb') as f:
        pickle.dump(train_transcripts_dict, f)
    
    with open(TEST_TRANSCRIPTS_FILE, 'wb') as f:
        pickle.dump(test_transcripts_dict, f)
    
    print(f"Training transcripts saved to {TRAIN_TRANSCRIPTS_FILE}")
    print(f"Test transcripts saved to {TEST_TRANSCRIPTS_FILE}")
    
    # Create dataframes with transcripts and text features for inspection
    train_transcripts_list = []
    for filename, result in train_transcripts_dict.items():
        trans_dict = {'filename': filename, 'transcript': result['transcript']}
        trans_dict.update(result['text_features'])
        # Add label if available
        label_row = train_df[train_df['filename'] == filename]
        if 'label' in label_row.columns and not label_row.empty:
            trans_dict['label'] = label_row['label'].values[0]
        train_transcripts_list.append(trans_dict)
    
    test_transcripts_list = []
    for filename, result in test_transcripts_dict.items():
        trans_dict = {'filename': filename, 'transcript': result['transcript']}
        trans_dict.update(result['text_features'])
        test_transcripts_list.append(trans_dict)
    
    train_transcripts_df = pd.DataFrame(train_transcripts_list)
    test_transcripts_df = pd.DataFrame(test_transcripts_list)
    
    # Save to CSV for inspection
    train_transcripts_df.to_csv(os.path.join(OUTPUT_DIR, "train_audio_transcripts_table.csv"), index=False)
    test_transcripts_df.to_csv(os.path.join(OUTPUT_DIR, "test_audio_transcripts_table.csv"), index=False)
    
    print("Transcript tables saved for inspection")
    
    # Print transcript statistics
    print("\nTraining Transcripts Statistics:")
    # Filter out columns that are object type (like transcript text)
    numeric_cols = train_transcripts_df.select_dtypes(include=[np.number]).columns
    print(train_transcripts_df[numeric_cols].describe())
    
    # If label column exists, show correlation with text features
    if 'label' in train_transcripts_df.columns:
        numeric_cols = train_transcripts_df.select_dtypes(include=[np.number]).columns
        print("\nText features correlated with grammar score:")
        correlations = train_transcripts_df[numeric_cols].corr()['label'].sort_values(ascending=False)
        print(correlations[1:])  # Skip the self-correlation (1.0)