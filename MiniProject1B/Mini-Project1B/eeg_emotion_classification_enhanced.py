# -*- coding: utf-8 -*-
"""
Enhanced EEG Emotion Classification System
Extended from project 1B to handle full dataset with multiple model architectures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os
import json
from typing import Dict, List, Tuple, Any
import seaborn as sns

# Constants
SAMPLE_RATE = 32  # Hz
GAMES = ["boring", "calm", "horror", "funny"]
ELECTRODES = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "O1", "O2", "P7", "P8", "T7", "T8"]

class EEGDataLoader:
    """Enhanced data loader for EEG emotion classification"""
    
    def __init__(self, data_dir: str = "data", sample_rate: int = 32, clip_length: int = 2):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.electrode = "T7"  # Default electrode
        
    def load_single_subject_data(self, subject_id: str = "S01") -> pd.DataFrame:
        """Load data for a single subject"""
        data = []
        for game_id, game in enumerate(GAMES):
            file_path = os.path.join(self.data_dir, f"{subject_id}G{game_id + 1}AllChannels.csv")
            if os.path.exists(file_path):
                game_data = pd.read_csv(file_path)
                game_data["game"] = game
                game_data["subject"] = subject_id
                data.append(game_data)
            else:
                print(f"Warning: File {file_path} not found")
        
        if data:
            return pd.concat(data, axis=0, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def load_multiple_subjects_data(self, subject_ids: List[str]) -> pd.DataFrame:
        """Load data for multiple subjects"""
        all_data = []
        for subject_id in subject_ids:
            subject_data = self.load_single_subject_data(subject_id)
            if not subject_data.empty:
                all_data.append(subject_data)
        
        if all_data:
            return pd.concat(all_data, axis=0, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def preprocess_data(self, data: pd.DataFrame, electrode: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess EEG data into clips"""
        if electrode is None:
            electrode = self.electrode
        
        # Select electrode and game columns
        data = data[[electrode, "game"]].copy()
        
        # Create clips
        clipped_data = []
        y = []
        
        for game_id, game in enumerate(GAMES):
            game_data = data[data['game'] == game][electrode].to_numpy()
            if len(game_data) > 0:
                # Split into clips
                clips = np.array_split(
                    game_data,
                    len(game_data) // (self.clip_length * self.sample_rate)
                )
                clipped_data.extend(clips)
                y.extend([game_id] * len(clips))
        
        if not clipped_data:
            return np.array([]), np.array([])
        
        # Remove edge effects by ensuring all clips have the same length
        min_length = min(len(arr) for arr in clipped_data)
        X = np.array([arr[:min_length] for arr in clipped_data], dtype=float)
        y = np.array(y, dtype=int)
        
        return X, y

class EEGDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.Tensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleCNN(nn.Module):
    """Simple CNN architecture (baseline from project 1B)"""
    
    def __init__(self, input_size: int = 64, num_classes: int = 4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(1, 1, kernel_size=4, padding="same")
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return self.log_softmax(x)

class MediumCNN(nn.Module):
    """Medium complexity CNN"""
    
    def __init__(self, input_size: int = 64, num_classes: int = 4):
        super(MediumCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(8, 16, kernel_size=4, padding="same")
        self.conv3 = nn.Conv1d(16, 32, kernel_size=4, padding="same")
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # Calculate the correct input size after pooling operations
        # input_size -> pool -> input_size//2 -> pool -> input_size//4
        self.fc1 = nn.Linear(32 * (input_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LargeCNN(nn.Module):
    """Large complexity CNN"""
    
    def __init__(self, input_size: int = 64, num_classes: int = 4):
        super(LargeCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, padding="same")
        self.conv3 = nn.Conv1d(32, 64, kernel_size=4, padding="same")
        self.conv4 = nn.Conv1d(64, 128, kernel_size=4, padding="same")
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # Calculate the correct input size after pooling operations
        # input_size -> pool -> input_size//2 -> pool -> input_size//4
        self.fc1 = nn.Linear(128 * (input_size // 4), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class VeryLargeCNN(nn.Module):
    """Very large complexity CNN"""
    
    def __init__(self, input_size: int = 64, num_classes: int = 4):
        super(VeryLargeCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, padding="same")
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, padding="same")
        self.conv4 = nn.Conv1d(128, 256, kernel_size=4, padding="same")
        self.conv5 = nn.Conv1d(256, 512, kernel_size=4, padding="same")
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # Calculate the correct input size after pooling operations
        # input_size -> pool -> input_size//2 -> pool -> input_size//4
        self.fc1 = nn.Linear(512 * (input_size // 4), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class MassiveCNN(nn.Module):
    """Massive complexity CNN"""
    
    def __init__(self, input_size: int = 64, num_classes: int = 4):
        super(MassiveCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, padding="same")
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, padding="same")
        self.conv4 = nn.Conv1d(256, 512, kernel_size=4, padding="same")
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=4, padding="same")
        self.conv6 = nn.Conv1d(1024, 2048, kernel_size=4, padding="same")
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # Calculate the correct input size after pooling operations
        # input_size -> pool -> input_size//2 -> pool -> input_size//4
        self.fc1 = nn.Linear(2048 * (input_size // 4), 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

class EEGTrainer:
    """Enhanced trainer with comprehensive metrics tracking"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = F.nll_loss(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                logits = self.model(X_batch)
                loss = F.nll_loss(logits, y_batch)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == y_batch).sum().item()
                total_samples += y_batch.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def train(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, 
              epochs: int = 100, learning_rate: float = 0.0001) -> Dict[str, Any]:
        """Train the model with comprehensive tracking"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            test_loss, test_acc = self.evaluate(test_loader)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Final evaluation
        final_test_loss, final_test_acc = self.evaluate(test_loader)
        
        results = {
            'final_train_loss': self.train_losses[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_test_loss': final_test_loss,
            'final_test_accuracy': final_test_acc,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }
        
        return results

def create_model_architectures() -> Dict[str, nn.Module]:
    """Create different model architectures"""
    architectures = {
        'SimpleCNN': SimpleCNN(),
        'MediumCNN': MediumCNN(),
        'LargeCNN': LargeCNN(),
        'VeryLargeCNN': VeryLargeCNN(),
        'MassiveCNN': MassiveCNN()
    }
    return architectures

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiments(data_loader: EEGDataLoader, X: np.ndarray, y: np.ndarray, 
                   batch_size: int = 32, epochs: int = 50) -> pd.DataFrame:
    """Run experiments with different model architectures"""
    
    # Prepare data
    X_expanded = np.expand_dims(X, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_expanded, y, test_size=0.3, random_state=42)
    
    # Create data loaders
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get model architectures
    architectures = create_model_architectures()
    
    results = []
    
    for model_name, model in architectures.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Count parameters
        num_params = count_parameters(model)
        print(f"Number of parameters: {num_params:,}")
        
        # Train model
        trainer = EEGTrainer(model)
        training_results = trainer.train(train_loader, test_loader, epochs=epochs)
        
        # Store results
        result = {
            'Model': model_name,
            'Parameters': num_params,
            'Train_Accuracy': training_results['final_train_accuracy'],
            'Test_Accuracy': training_results['final_test_accuracy'],
            'Train_Loss': training_results['final_train_loss'],
            'Test_Loss': training_results['final_test_loss'],
            'Training_Time': training_results['training_time']
        }
        results.append(result)
        
        print(f"Final Test Accuracy: {training_results['final_test_accuracy']:.4f}")
        print(f"Training Time: {training_results['training_time']:.2f} seconds")
    
    return pd.DataFrame(results)

def plot_results(results_df: pd.DataFrame, save_path: str = None):
    """Plot the results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Parameters vs Accuracy
    axes[0, 0].plot(results_df['Parameters'], results_df['Test_Accuracy'], 'bo-', label='Test Accuracy')
    axes[0, 0].plot(results_df['Parameters'], results_df['Train_Accuracy'], 'ro-', label='Train Accuracy')
    axes[0, 0].set_xlabel('Number of Parameters')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Size vs Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Parameters vs Training Time
    axes[0, 1].plot(results_df['Parameters'], results_df['Training_Time'], 'go-')
    axes[0, 1].set_xlabel('Number of Parameters')
    axes[0, 1].set_ylabel('Training Time (seconds)')
    axes[0, 1].set_title('Model Size vs Training Time')
    axes[0, 1].grid(True)
    
    # Model comparison bar chart
    x_pos = range(len(results_df))
    axes[1, 0].bar([x - 0.2 for x in x_pos], results_df['Train_Accuracy'], 0.4, label='Train Accuracy', alpha=0.7)
    axes[1, 0].bar([x + 0.2 for x in x_pos], results_df['Test_Accuracy'], 0.4, label='Test Accuracy', alpha=0.7)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Model Comparison - Accuracy')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(results_df['Model'], rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y')
    
    # Training time bar chart
    axes[1, 1].bar(results_df['Model'], results_df['Training_Time'], alpha=0.7)
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Model Comparison - Training Time')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Main function to run the enhanced EEG emotion classification"""
    print("Enhanced EEG Emotion Classification System")
    print("=" * 50)
    
    # Initialize data loader
    data_loader = EEGDataLoader()
    
    # Load data (using available S01 data for now)
    print("Loading data...")
    data = data_loader.load_single_subject_data("S01")
    
    if data.empty:
        print("No data found. Please ensure data files are in the 'data' directory.")
        return
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Available games: {data['game'].unique()}")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = data_loader.preprocess_data(data)
    
    if len(X) == 0:
        print("No valid clips found after preprocessing.")
        return
    
    print(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Run experiments
    print("\nStarting experiments...")
    results_df = run_experiments(data_loader, X, y, epochs=30)  # Reduced epochs for faster testing
    
    # Display results
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Save results
    results_df.to_csv('eeg_experiment_results.csv', index=False)
    print(f"\nResults saved to 'eeg_experiment_results.csv'")
    
    # Plot results
    plot_results(results_df, 'eeg_experiment_plots.png')
    print("Plots saved to 'eeg_experiment_plots.png'")
    
    # Save detailed results as JSON
    detailed_results = {
        'experiment_info': {
            'sample_rate': SAMPLE_RATE,
            'clip_length': data_loader.clip_length,
            'electrode': data_loader.electrode,
            'games': GAMES,
            'total_samples': len(X),
            'train_test_split': 0.7
        },
        'results': results_df.to_dict('records')
    }
    
    with open('eeg_detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("Detailed results saved to 'eeg_detailed_results.json'")

if __name__ == "__main__":
    main()
