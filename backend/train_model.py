import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
from imutils import paths
import os
import time
import copy
from tqdm import tqdm
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
data_dir = Path("dataset")
train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"

def get_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomErasing(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

def prepare_dataset():
    if not train_dir.exists():
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        print("Splitting dataset...")
        image_paths = list(paths.list_images(data_dir))
        train_paths, rest_paths = train_test_split(
            image_paths, 
            test_size=0.3, 
            stratify=[Path(p).parent.name for p in image_paths], 
            random_state=42
        )
        val_paths, test_paths = train_test_split(
            rest_paths, 
            test_size=0.5, 
            stratify=[Path(p).parent.name for p in rest_paths], 
            random_state=42
        )

        print("Copying files to train/val/test directories...")
        for path_list, folder in zip([train_paths, val_paths, test_paths], [train_dir, val_dir, test_dir]):
            for img_path in tqdm(path_list, desc=f"Copying to {folder.name}"):
                label = Path(img_path).parent.name
                label_dir = folder / label
                label_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, label_dir / Path(img_path).name)

def load_datasets():
    transforms_dict = get_transforms()
    return {
        "train": datasets.ImageFolder(train_dir, transform=transforms_dict["train"]),
        "val": datasets.ImageFolder(val_dir, transform=transforms_dict["val"]),
        "test": datasets.ImageFolder(test_dir, transform=transforms_dict["val"])
    }

def get_dataloaders(datasets_dict):
    return {
        "train": DataLoader(datasets_dict["train"], batch_size=16, shuffle=True, num_workers=4),
        "val": DataLoader(datasets_dict["val"], batch_size=16, shuffle=False, num_workers=4),
        "test": DataLoader(datasets_dict["test"], batch_size=16, shuffle=False, num_workers=4)
    }

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Phase')

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{torch.sum(preds == labels.data).item() / inputs.size(0):.4f}'
                })

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'models/best_model.pth')
                    print(f'New best model saved! Validation accuracy: {best_acc:.4f}')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch} due to no improvement in validation accuracy.')
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloader, class_names):
    """Evaluates the model on the test set and prints a classification report."""
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n--- Test Results ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    cm_path = 'models/confusion_matrix.png'
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.show()

def main():
    os.makedirs('models', exist_ok=True)
    print("Preparing dataset...")
    prepare_dataset()
    print("Loading datasets...")
    datasets_dict = load_datasets()
    dataloaders = get_dataloaders(datasets_dict)
    class_names = datasets_dict['train'].classes
    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")

    print("Initializing model...")
    num_classes = len(class_names)
    # Use EfficientNet from timm
    model = timm.create_model('efficientnet_b3a', pretrained=True, num_classes=num_classes)
    # Unfreeze all layers for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    # Use ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    print("Starting training...")
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=30, patience=5)
    torch.save(model.state_dict(), 'models/final_model.pth')
    print("Training complete! Models saved in 'models' directory.")

    # Test the best model
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load('models/best_model.pth'))
    test_model(model, dataloaders['test'], class_names)

if __name__ == "__main__":
    main() 