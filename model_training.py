import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Define paths
OUTPUT_DIR = "model_artifacts"
TRAIN_CSV = "/Users/priyanshuwalia7/Downloads/Dataset/train.csv"
TEST_CSV = "/Users/priyanshuwalia7/Downloads/Dataset/test.csv"
TRAIN_AUDIO_FEATURES_FILE = os.path.join(OUTPUT_DIR, "train_audio_features.pkl")
TEST_AUDIO_FEATURES_FILE = os.path.join(OUTPUT_DIR, "test_audio_features.pkl")
TRAIN_TRANSCRIPTS_FILE = os.path.join(OUTPUT_DIR, "train_audio_transcripts.pkl")
TEST_TRANSCRIPTS_FILE = os.path.join(OUTPUT_DIR, "test_audio_transcripts.pkl")

# Function to load a pickle file
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Warning: {file_path} not found, returning empty dict")
        return {}

# Function to create a combined features dataframe
def create_feature_dataframe(audio_features, transcripts, labels_df=None):
    # Start with filenames
    files = list(set(list(audio_features.keys()) + list(transcripts.keys())))
    features_data = []
    
    for filename in files:
        feature_dict = {'filename': filename}
        
        # Add audio features if available
        if filename in audio_features:
            for key, value in audio_features[filename].items():
                if isinstance(value, (int, float, bool, np.number)):
                    feature_dict[f'audio_{key}'] = value
        
        # Add transcript features if available
        if filename in transcripts and 'text_features' in transcripts[filename]:
            for key, value in transcripts[filename]['text_features'].items():
                if isinstance(value, (int, float, bool, np.number)):
                    feature_dict[f'text_{key}'] = value
        
        features_data.append(feature_dict)
    
    # Create dataframe
    features_df = pd.DataFrame(features_data)
    
    # Merge with labels if provided
    if labels_df is not None and 'label' in labels_df.columns:
        merged_df = features_df.merge(labels_df[['filename', 'label']], on='filename', how='inner')
        return merged_df
    else:
        return features_df

# Main execution
if __name__ == "__main__":
    print("Loading data files...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_audio_features = load_pickle(TRAIN_AUDIO_FEATURES_FILE)
    test_audio_features = load_pickle(TEST_AUDIO_FEATURES_FILE)
    train_transcripts = load_pickle(TRAIN_TRANSCRIPTS_FILE)
    test_transcripts = load_pickle(TEST_TRANSCRIPTS_FILE)
    
    print(f"Loaded {len(train_df)} labeled training examples and {len(test_df)} test examples")
    
    print("Creating combined feature dataframes...")
    train_df = create_feature_dataframe(train_audio_features, train_transcripts, train_df)
    
    # For test dataframe, don't try to merge with 'label' column
    test_features_df = create_feature_dataframe(test_audio_features, test_transcripts)
    
    # Save the combined feature dataframes
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_combined_features.csv"), index=False)
    test_features_df.to_csv(os.path.join(OUTPUT_DIR, "test_combined_features.csv"), index=False)
    
    print(f"Features shape: Train={train_df.shape}, Test={test_features_df.shape}")
    
    # Prepare features and target for modeling
    X = train_df.drop(['filename', 'label'], axis=1, errors='ignore')
    y = train_df['label'] if 'label' in train_df.columns else None
    
    if y is None:
        print("Error: No 'label' column found in training data")
        exit(1)
    
    # Handle missing values
    X = X.fillna(0)
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} examples")
    print(f"Validation set: {X_val.shape[0]} examples")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "feature_scaler.joblib"))
    
    # Train a Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_preds = rf_model.predict(X_val_scaled)
    val_mse = mean_squared_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save the feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    # Train final model on all data
    print("Training final model on all data...")
    X_all_scaled = scaler.transform(X)
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X_all_scaled, y)
    
    # Save the model
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, "grammar_scoring_model.joblib"))
    
    # Prepare test data for predictions (if needed)
    test_features = test_features_df.drop('filename', axis=1, errors='ignore')
    test_features = test_features.fillna(0)
    
    # Ensure test features have the same columns as training features
    for col in X.columns:
        if col not in test_features.columns:
            test_features[col] = 0
    
    # Reorder columns to match training data
    test_features = test_features.reindex(columns=X.columns)
    
    # Scale the test features
    test_features_scaled = scaler.transform(test_features)
    
    # Make predictions
    test_predictions = final_model.predict(test_features_scaled)
    
    # Add predictions to test dataframe
    test_df['predicted_label'] = test_predictions
    
    # Save predictions
    test_predictions_df = pd.DataFrame({
        'filename': test_features_df['filename'],
        'predicted_score': test_predictions
    })
    test_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
    
    print(f"Model and predictions saved to {OUTPUT_DIR}")