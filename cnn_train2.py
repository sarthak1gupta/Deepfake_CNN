import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import cv2
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import logging
import json
from typing import Tuple, List, Dict
import warnings
import gc
import random
from collections import Counter
import time
from google.colab import drive
import shutil

warnings.filterwarnings('ignore')

drive.mount('/content/drive')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class BalancedDeepfakeDataset(Dataset):
    def __init__(self, real_dir: str, fake_dir: str, transform=None, 
                 image_size: int = 224, augment: bool = True, 
                 balance_classes: bool = True, max_samples_per_class: int = None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.image_size = image_size
        self.augment = augment
        self.balance_classes = balance_classes
        
        # Get all image paths with better error handling
        self.real_images = self._get_image_paths(real_dir)
        self.fake_images = self._get_image_paths(fake_dir)
        
        logger.info(f"Found {len(self.real_images)} real images and {len(self.fake_images)} fake images")
        
        # Handle class imbalance
        if balance_classes:
            self.real_images, self.fake_images = self._balance_classes(
                self.real_images, self.fake_images, max_samples_per_class
            )
        
        # Create final dataset
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        
        # Shuffle the dataset
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        if len(combined) == 0:
            logger.warning(f"No images found in {real_dir} and {fake_dir}. Dataset will be empty.")
            self.images, self.labels = [], []
        else:
            self.images, self.labels = zip(*combined)
        
        logger.info(f"Final dataset: {len(self.real_images)} real, {len(self.fake_images)} fake images")
        
        # Enhanced augmentations for training
        if augment:
            self.aug_transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=0.1),
                    A.RandomRotate90(p=0.3),
                ], p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                ], p=0.4),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=7, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
                    A.Downscale(scale_min=0.7, scale_max=0.9, p=1.0),
                ], p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                               min_holes=1, min_height=8, min_width=8, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.aug_transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _get_image_paths(self, directory):
        """Get all valid image paths from directory"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_paths = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))
        
        print(f"DEBUG: {directory} -> {len(image_paths)} images found: {image_paths[:5]}")
        return image_paths
    
    def _balance_classes(self, real_images, fake_images, max_samples_per_class):
        """Balance classes to handle data imbalance"""
        real_count = len(real_images)
        fake_count = len(fake_images)
        
        if max_samples_per_class:
            target_count = min(max_samples_per_class, real_count, fake_count)
        else:
            target_count = min(real_count, fake_count)
        
        # Randomly sample to balance classes
        if real_count > target_count:
            real_images = random.sample(real_images, target_count)
        if fake_count > target_count:
            fake_images = random.sample(fake_images, target_count)
        
        logger.info(f"Balanced classes: {len(real_images)} real, {len(fake_images)} fake")
        return real_images, fake_images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Optimized image loading
            image = self._load_image(img_path)
            
            # Apply preprocessing
            image = self._preprocess_image(image)
            
            # Apply augmentations
            transformed = self.aug_transform(image=image)
            image = transformed['image']
            
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            transformed = self.aug_transform(image=image)
            return transformed['image'], torch.tensor(label, dtype=torch.long)
    
    def _load_image(self, img_path):
        """Optimized image loading with fallback"""
        try:
            # Try PIL first (faster for many formats)
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except:
            # Fallback to OpenCV
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Cannot load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _preprocess_image(self, image):
        """Enhanced image preprocessing"""
        # Handle edge cases
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # Remove alpha channel
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Handle very small images
        h, w = image.shape[:2]
        if min(h, w) < 64:
            scale_factor = 64 / min(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Enhance very blurry images
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            image = cv2.filter2D(image, -1, kernel)
        
        return image

class MultiScaleFeatureExtractor(nn.Module):
    """Enhanced multi-scale feature extractor"""
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        features = []
        
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x1 = self.dropout(x1)
        features.append(self.pool(x1).flatten(1))
        
        x2 = torch.relu(self.bn2(self.conv2(x1)))
        x2 = self.dropout(x2)
        features.append(self.pool(x2).flatten(1))
        
        x3 = torch.relu(self.bn3(self.conv3(x2)))
        x3 = self.dropout(x3)
        features.append(self.pool(x3).flatten(1))
        
        x4 = torch.relu(self.bn4(self.conv4(x3)))
        x4 = self.dropout(x4)
        features.append(self.pool(x4).flatten(1))
        
        return torch.cat(features, dim=1)

class EnhancedRobustDeepfakeDetector(nn.Module):
    """Enhanced robust deepfake detection model"""
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        
        # Main backbone - EfficientNet-B4
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        backbone_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Multi-scale feature extractor
        self.multiscale_extractor = MultiScaleFeatureExtractor()
        multiscale_features = 64 + 128 + 256 + 512
        
        # Enhanced frequency domain analysis
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Edge detection branch
        self.edge_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined classifier
        total_features = backbone_features + multiscale_features + 256 + 128
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(total_features // 2, total_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Main backbone features
        backbone_feat = self.backbone(x)
        
        # Multi-scale features
        multiscale_feat = self.multiscale_extractor(x)
        
        # Frequency domain features
        freq_feat = self.freq_branch(x)
        
        # Edge detection features
        edge_feat = self.edge_branch(x)
        
        # Combine all features
        combined_feat = torch.cat([backbone_feat, multiscale_feat, freq_feat, edge_feat], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined_feat)
        attended_feat = combined_feat * attention_weights
        
        # Final classification
        output = self.classifier(attended_feat)
        return output

class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss with class weighting"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, train_acc, loss, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'train_acc': train_acc,
        'loss': loss,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, scheduler, filepath):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['val_acc'], checkpoint['train_acc']

def train_model_enhanced(model, train_loader, val_loader, num_epochs=50, device='cuda',
                        checkpoint_dir='checkpoints', resume_from=None):

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Calculate class weights for imbalanced dataset
    class_counts = Counter([train_loader.dataset.labels[i] for i in range(len(train_loader.dataset))])
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([total_samples / class_counts[i] for i in range(len(class_counts))], 
                                dtype=torch.float32, device=device)
    
    # Enhanced optimizer with different learning rates for different parts
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Lower LR for pretrained backbone
        {'params': other_params, 'lr': 1e-4}      # Higher LR for new layers
    ], weight_decay=1e-4)
    
    # Enhanced scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Loss function with class weights
    criterion = AdaptiveFocalLoss(alpha=class_weights, gamma=2)
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        start_epoch, best_val_acc, _ = load_checkpoint(model, optimizer, scheduler, resume_from)
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Clear cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
        logger.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc, train_acc, 
                          val_losses[-1], checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save to Drive
            best_model_path = '/content/drive/MyDrive/best_deepfake_detector.pth'
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc, train_acc, 
                          val_losses[-1], best_model_path)
            
            # Also save locally
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc, train_acc, 
                          val_losses[-1], 'best_model_local.pth')
            
            logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {patience} epochs without improvement')
            break
        
        scheduler.step()
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model_enhanced(model, test_loader, device='cuda'):
    """Enhanced model evaluation with more metrics"""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Evaluating'):
            data, targets = data.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    logger.info("=== Enhanced Model Evaluation Results ===")
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Real Images - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    logger.info(f"Fake Images - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
    logger.info(f"Specificity (Real detection): {specificity:.4f}")
    logger.info(f"Sensitivity (Fake detection): {sensitivity:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/content/drive/MyDrive/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1, specificity, sensitivity

def main():
    """Main training pipeline optimized for Colab"""
    
    # Configuration optimized for Colab
    config = {
        'real_train_dir': '/content/drive/MyDrive/Dataset/Train/Real',
        'fake_train_dir': '/content/drive/MyDrive/Dataset/Train/Fake',
        'real_val_dir': '/content/drive/MyDrive/Dataset/Validation/Real',
        'fake_val_dir': '/content/drive/MyDrive/Dataset/Validation/Fake',
        'real_test_dir': '/content/drive/MyDrive/Dataset/Test/Real',
        'fake_test_dir': '/content/drive/MyDrive/Dataset/Test/Fake',
        'batch_size': 16,  # Reduced for large dataset and memory efficiency
        'num_epochs': 30,
        'image_size': 224,
        'num_workers': 2,  # Reduced for Colab
        'max_samples_per_class': 50000,  # Limit to prevent memory issues
        'device': device
    }
    
    logger.info(f"Configuration: {config}")
    
    # Create datasets with balancing
    train_dataset = BalancedDeepfakeDataset(
        config['real_train_dir'], config['fake_train_dir'],
        image_size=config['image_size'], augment=True, 
        balance_classes=True, max_samples_per_class=config['max_samples_per_class']
    )
    
    val_dataset = BalancedDeepfakeDataset(
        config['real_val_dir'], config['fake_val_dir'],
        image_size=config['image_size'], augment=False,
        balance_classes=True, max_samples_per_class=config['max_samples_per_class']//4
    )
    
    test_dataset = BalancedDeepfakeDataset(
        config['real_test_dir'], config['fake_test_dir'],
        image_size=config['image_size'], augment=False,
        balance_classes=False  # Keep original distribution for testing
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], 
        pin_memory=True, persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], 
        pin_memory=True, persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], 
        pin_memory=True, persistent_workers=True
    )
    
    # Initialize enhanced model
    model = EnhancedRobustDeepfakeDetector(num_classes=2, dropout_rate=0.4)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Display model summary
    logger.info("Model Architecture:")
    logger.info(f"- Backbone: EfficientNet-B4")
    logger.info(f"- Multi-scale features: {64 + 128 + 256 + 512}")
    logger.info(f"- Frequency features: 256")
    logger.info(f"- Edge features: 128")
    logger.info(f"- Total feature dimensions: {model.classifier[0].in_features}")
    
    # Check for existing checkpoints
    checkpoint_path = '/content/drive/MyDrive/best_deepfake_detector.pth'
    resume_from = checkpoint_path if os.path.exists(checkpoint_path) else None
    
    # Train model
    logger.info("Starting enhanced training...")
    train_losses, val_losses, train_accs, val_accs = train_model_enhanced(
        model, train_loader, val_loader, 
        num_epochs=config['num_epochs'], device=config['device'],
        checkpoint_dir='/content/drive/MyDrive/checkpoints',
        resume_from=resume_from
    )
    
    # Load best model for testing
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.2f}%")
    else:
        logger.warning("No best model found, using current model state")
    
    model.to(device)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = evaluate_model_enhanced(model, test_loader, device)
    
    # Plot enhanced training curves
    plt.figure(figsize=(20, 12))
    
    # Loss curves
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(2, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(val_accs, label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate schedule (if available)
    plt.subplot(2, 3, 3)
    if 'lr_history' in locals():
        plt.plot(lr_history, linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # Performance metrics
    plt.subplot(2, 3, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'Sensitivity']
    values = [test_results[0], test_results[1][1], test_results[2][1], 
              test_results[3][1], test_results[4], test_results[5]]
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 
                                          'lightyellow', 'lightpink', 'lightgray'])
    plt.title('Test Set Performance Metrics', fontsize=14)
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Class-wise performance
    plt.subplot(2, 3, 5)
    classes = ['Real', 'Fake']
    precision_vals = test_results[1]
    recall_vals = test_results[2]
    f1_vals = test_results[3]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
    plt.bar(x, recall_vals, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8)
    
    plt.title('Class-wise Performance', fontsize=14)
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dataset statistics
    plt.subplot(2, 3, 6)
    train_real = len(train_dataset.real_images)
    train_fake = len(train_dataset.fake_images)
    val_real = len(val_dataset.real_images)
    val_fake = len(val_dataset.fake_images)
    test_real = len(test_dataset.real_images)
    test_fake = len(test_dataset.fake_images)
    
    categories = ['Train', 'Validation', 'Test']
    real_counts = [train_real, val_real, test_real]
    fake_counts = [train_fake, val_fake, test_fake]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, real_counts, width, label='Real', alpha=0.8)
    plt.bar(x + width/2, fake_counts, width, label='Fake', alpha=0.8)
    
    plt.title('Dataset Distribution', fontsize=14)
    plt.ylabel('Number of Images')
    plt.xlabel('Dataset Split')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/enhanced_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comprehensive results
    results = {
        'config': config,
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Enhanced EfficientNet-B4 with Multi-scale Features'
        },
        'dataset_info': {
            'train': {'real': train_real, 'fake': train_fake},
            'validation': {'real': val_real, 'fake': val_fake},
            'test': {'real': test_real, 'fake': test_fake}
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accs,
            'val_accuracies': val_accs
        },
        'test_results': {
            'accuracy': float(test_results[0]),
            'precision': {
                'real': float(test_results[1][0]),
                'fake': float(test_results[1][1])
            },
            'recall': {
                'real': float(test_results[2][0]),
                'fake': float(test_results[2][1])
            },
            'f1_score': {
                'real': float(test_results[3][0]),
                'fake': float(test_results[3][1])
            },
            'specificity': float(test_results[4]),
            'sensitivity': float(test_results[5])
        },
        'best_epoch': checkpoint.get('epoch', 0) if 'checkpoint' in locals() else 0,
        'best_val_accuracy': checkpoint.get('val_acc', 0) if 'checkpoint' in locals() else 0
    }
    
    # Save results to Drive
    with open('/content/drive/MyDrive/enhanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create model summary for deployment
    model_summary = {
        'model_name': 'Enhanced Robust Deepfake Detector',
        'version': '2.0',
        'architecture': 'EfficientNet-B4 + Multi-scale Features + Frequency Analysis + Edge Detection',
        'input_size': (3, config['image_size'], config['image_size']),
        'num_classes': 2,
        'performance': {
            'test_accuracy': float(test_results[0]),
            'fake_detection_rate': float(test_results[5]),
            'real_detection_rate': float(test_results[4])
        },
        'model_path': '/content/drive/MyDrive/best_deepfake_detector.pth',
        'preprocessing': {
            'resize': config['image_size'],
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    }
    
    with open('/content/drive/MyDrive/model_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    return results

def load_trained_model(model_path, device='cuda'):
    """Load trained model for inference"""
    model = EnhancedRobustDeepfakeDetector(num_classes=2, dropout_rate=0.4)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device='cuda', image_size=224):
    """Predict single image"""
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
    
    return {
        'prediction': 'Real' if prediction == 0 else 'Fake',
        'confidence': float(probabilities[0][prediction]),
        'probabilities': {
            'real': float(probabilities[0][0]),
            'fake': float(probabilities[0][1])
        }
    }

if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*60)
    print("EXAMPLE: Loading and using the trained model")
    print("="*60)

    model_path = '/content/drive/MyDrive/best_deepfake_detector.pth'
    if os.path.exists(model_path):
        trained_model = load_trained_model(model_path, device)
        print(f"Model loaded successfully from {model_path}")
        print("You can now use predict_image() function to classify new images!")
    else:
        print("Model file not found. Please run training first.")
    
    print("="*60)