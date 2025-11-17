import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import gc
from datetime import datetime

# --- GPU Configuration ---
print("="*60)
print("GPU CONFIGURATION")
print("="*60)

# List all GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth (prevents TensorFlow from allocating all GPU memory at once)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✓ Physical GPUs: {len(gpus)}")
        print(f"✓ Logical GPUs: {len(logical_gpus)}")
        
        for i, gpu in enumerate(gpus):
            print(f"\n  GPU {i}:")
            print(f"    Name: {gpu.name}")
            print(f"    Type: {gpu.device_type}")
            
    except RuntimeError as e:
        print(f"✗ GPU configuration error: {e}")
        exit(1)
else:
    print("✗ No GPU detected!")
    print("Please install GPU drivers and CUDA toolkit")
    exit(1)

# Enable mixed precision training (uses float16 for faster computation)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"\n✓ Mixed precision enabled: {policy.name}")
print("  This will speed up training by ~2x on modern GPUs")

# Enable XLA compilation for additional speedup
tf.config.optimizer.set_jit(True)
print("✓ XLA JIT compilation enabled")

print("="*60 + "\n")

# --- Configuration ---
BASE_DIR = r"D:\github_projects\stressdetect"
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Model hyperparameters - OPTIMIZED FOR GPU
BATCH_SIZE = 16  # Good for GPU
EPOCHS = 50
LEARNING_RATE = 0.0001
MAX_SAMPLES_PER_CLASS = 1000  # Increase for full dataset

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def load_data(max_samples_per_class=MAX_SAMPLES_PER_CLASS):
    """
    Load preprocessed data efficiently.
    """
    print("="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    print(f"Max samples per class: {max_samples_per_class}")
    
    video_data = []
    audio_data = []
    labels = []
    
    # Load stress samples (label = 1)
    stress_dir = os.path.join(PREPROCESSED_DIR, "stress")
    if os.path.exists(stress_dir):
        files = [f for f in os.listdir(stress_dir) if f.endswith('_video.npy')]
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        
        print(f"\nLoading {len(files)} stress samples...")
        
        for video_file in tqdm(files, desc="Loading stress"):
            audio_file = video_file.replace('_video.npy', '_audio.npz')
            video_path = os.path.join(stress_dir, video_file)
            audio_path = os.path.join(stress_dir, audio_file)
            
            if os.path.exists(audio_path):
                try:
                    video = np.load(video_path)
                    audio = np.load(audio_path)
                    
                    if video.shape[0] == 60:  # Ensure 60 frames
                        video_data.append(video)
                        audio_data.append(audio['log_mel'])
                        labels.append(1)
                except Exception as e:
                    continue
    
    # Load non-stress samples (label = 0)
    non_stress_dir = os.path.join(PREPROCESSED_DIR, "non-stress")
    if os.path.exists(non_stress_dir):
        files = [f for f in os.listdir(non_stress_dir) if f.endswith('_video.npy')]
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        
        print(f"Loading {len(files)} non-stress samples...")
        
        for video_file in tqdm(files, desc="Loading non-stress"):
            audio_file = video_file.replace('_video.npy', '_audio.npz')
            video_path = os.path.join(non_stress_dir, video_file)
            audio_path = os.path.join(non_stress_dir, audio_file)
            
            if os.path.exists(audio_path):
                try:
                    video = np.load(video_path)
                    audio = np.load(audio_path)
                    
                    if video.shape[0] == 60:  # Ensure 60 frames
                        video_data.append(video)
                        audio_data.append(audio['log_mel'])
                        labels.append(0)
                except Exception as e:
                    continue
    
    print(f"\n✓ Total samples loaded: {len(video_data)}")
    print(f"  - Stress: {sum(labels)}")
    print(f"  - Non-stress: {len(labels) - sum(labels)}")
    
    if len(video_data) == 0:
        return None, None, None
    
    return (
        np.array(video_data, dtype=np.float32), 
        np.array(audio_data, dtype=np.float32), 
        np.array(labels, dtype=np.int32)
    )

def build_video_model(input_shape):
    """
    Video model: ResNet50 + Bi-LSTM
    Optimized for GPU with CuDNN-compatible LSTM
    """
    video_input = layers.Input(shape=input_shape, name='video_input')
    
    # ResNet50 feature extractor
    resnet_base = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Fine-tune last few layers
    for layer in resnet_base.layers[:-20]:
        layer.trainable = False
    
    # Apply ResNet to each frame
    x = layers.TimeDistributed(resnet_base, name='resnet_features')(video_input)
    
    # Bi-LSTM layers (CuDNN optimized for GPU)
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.3),
        name='bi_lstm_1'
    )(x)
    
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False, dropout=0.3),
        name='bi_lstm_2'
    )(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu', name='video_dense_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', name='video_dense_2')(x)
    video_output = layers.Dropout(0.4)(x)
    
    model = models.Model(inputs=video_input, outputs=video_output, name='video_model')
    return model

def build_audio_model(input_shape):
    """
    Audio model: 1D-CNN
    Optimized for GPU with batch normalization
    """
    audio_input = layers.Input(shape=input_shape, name='audio_input')
    
    # Reshape for 1D convolution
    x = layers.Reshape((input_shape[0] * input_shape[1], 1))(audio_input)
    
    # 1D Convolutional blocks
    x = layers.Conv1D(64, kernel_size=8, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, kernel_size=8, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(256, kernel_size=8, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(0.3)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu', name='audio_dense_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', name='audio_dense_2')(x)
    audio_output = layers.Dropout(0.4)(x)
    
    model = models.Model(inputs=audio_input, outputs=audio_output, name='audio_model')
    return model

def build_fusion_model(video_shape, audio_shape):
    """
    Fusion model combining video and audio
    """
    print("\n" + "="*60)
    print("BUILDING BASELINE MODEL (GPU OPTIMIZED)")
    print("="*60)
    
    print("\nBuilding video model (ResNet50 + Bi-LSTM)...")
    video_model = build_video_model(video_shape)
    
    print("Building audio model (1D-CNN)...")
    audio_model = build_audio_model(audio_shape)
    
    # Create inputs
    video_input = layers.Input(shape=video_shape, name='video_input')
    audio_input = layers.Input(shape=audio_shape, name='audio_input')
    
    # Get features
    video_features = video_model(video_input)
    audio_features = audio_model(audio_input)
    
    # Fusion
    print("\nFusing video and audio features...")
    concatenated = layers.Concatenate(name='fusion')([video_features, audio_features])
    
    # Final classification layers
    x = layers.Dense(256, activation='relu', name='fusion_dense_1')(concatenated)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', name='fusion_dense_2')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', name='fusion_dense_3')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer (float32 for numerical stability with mixed precision)
    output = layers.Dense(1, activation='sigmoid', name='output', dtype='float32')(x)
    
    model = models.Model(
        inputs=[video_input, audio_input],
        outputs=output,
        name='baseline_fusion_model'
    )
    
    print("\n✓ Model architecture created")
    return model

def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Stress', 'Stress'],
                yticklabels=['Non-Stress', 'Stress'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 16, 'weight': 'bold'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()

def main():
    """Main training function."""
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING - STRESS DETECTION (GPU)")
    print("="*60)
    
    # Load data
    video_data, audio_data, labels = load_data()
    
    if video_data is None:
        print("\n✗ No data found! Check preprocessing output.")
        return
    
    # Split data
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    indices = np.arange(len(video_data))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )
    
    X_video_train = video_data[train_idx]
    X_audio_train = audio_data[train_idx]
    y_train = labels[train_idx]
    
    X_video_val = video_data[val_idx]
    X_audio_val = audio_data[val_idx]
    y_val = labels[val_idx]
    
    X_video_test = video_data[test_idx]
    X_audio_test = audio_data[test_idx]
    y_test = labels[test_idx]
    
    print(f"\nTrain samples: {len(X_video_train)}")
    print(f"Validation samples: {len(X_video_val)}")
    print(f"Test samples: {len(X_video_test)}")
    
    # Clear memory
    del video_data, audio_data, labels, indices
    gc.collect()
    
    # Build model
    model = build_fusion_model(
        video_shape=X_video_train.shape[1:],
        audio_shape=X_audio_train.shape[1:]
    )
    
    # Compile model
    print("\nCompiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Print model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\n✓ Total parameters: {total_params:,}")
    
    # Callbacks
    log_dir = os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'baseline_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL ON GPU")
    print("="*60)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"TensorBoard logs: {log_dir}")
    print("="*60 + "\n")
    
    history = model.fit(
        [X_video_train, X_audio_train],
        y_train,
        validation_data=([X_video_val, X_audio_val], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(MODEL_DIR, 'baseline_final.keras'))
    print(f"\n✓ Final model saved to: {MODEL_DIR}")
    
    # Save training history
    with open(os.path.join(RESULTS_DIR, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    # Plot training history
    plot_training_history(
        history, 
        os.path.join(RESULTS_DIR, 'training_history.png')
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(
        [X_video_test, X_audio_test],
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Get predictions
    y_pred_prob = model.predict([X_video_test, X_audio_test], batch_size=BATCH_SIZE)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    test_f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"AUC:       {test_auc:.4f}")
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_test, y_pred, 
        target_names=['Non-Stress', 'Stress'],
        digits=4
    ))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    )
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'test_loss': float(test_loss),
        'total_params': int(total_params),
        'batch_size': BATCH_SIZE,
        'epochs': len(history.history['loss']),
        'learning_rate': LEARNING_RATE
    }
    
    with open(os.path.join(RESULTS_DIR, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save results as text
    with open(os.path.join(RESULTS_DIR, 'results_summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("STRESS DETECTION MODEL - FINAL RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: Baseline (ResNet50 + Bi-LSTM + 1D-CNN)\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Training Device: GPU\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs Trained: {len(history.history['loss'])}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n\n")
        f.write("="*60 + "\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:  {test_acc:.4f}\n")
        f.write(f"Precision: {test_prec:.4f}\n")
        f.write(f"Recall:    {test_rec:.4f}\n")
        f.write(f"F1-Score:  {test_f1:.4f}\n")
        f.write(f"AUC:       {test_auc:.4f}\n")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✓ Models saved: {MODEL_DIR}")
    print(f"✓ Results saved: {RESULTS_DIR}")
    print(f"✓ TensorBoard logs: {log_dir}")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}")
    print("="*60)

if __name__ == "__main__":
    main()