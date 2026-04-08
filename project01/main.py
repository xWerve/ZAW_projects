# Imports
import os
import random
from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cable_dir = os.path.join(BASE_DIR, "cable")

if not os.path.exists(cable_dir):
    raise FileNotFoundError(f"'cable'404: {BASE_DIR}")

# Find all test images
test_images = glob(os.path.join(cable_dir, 'test', '*', '*.png'))
defect_images = [img for img in test_images if 'good' not in img]
good_test_images = [img for img in test_images if 'good' in img]

# Find all training images
train_images = glob(os.path.join(cable_dir, 'train', '*', '*.png'))

data_pairs = []
for img_path in defect_images:
    parts = img_path.split(os.sep)
    defect_type = parts[-2]
    file_name = parts[-1]
    mask_name = file_name.replace('.png', '_mask.png')
    gt_path = os.path.join(cable_dir, 'ground_truth', defect_type, mask_name)

    if os.path.exists(gt_path):
        data_pairs.append({
            'image': img_path,
            'mask': gt_path,
            'is_defective': 1
        })

test_ds = []
TARGET_SIZE = (256, 256)

for item in data_pairs:
    img = Image.open(item['image']).convert('RGB').resize(TARGET_SIZE)
    mask = Image.open(item['mask']).convert('L').resize(TARGET_SIZE)

    img_array = np.array(img)
    mask_array = (np.array(mask) > 128).astype(np.uint8) * 255
    test_ds.append([img_array, mask_array, item['is_defective']])

for path in good_test_images:
    img = Image.open(path).convert('RGB').resize(TARGET_SIZE)
    test_ds.append([np.array(img), np.zeros(TARGET_SIZE, dtype=np.uint8), 0])

train_ds = []
for path in train_images:
    img = Image.open(path).convert('RGB').resize(TARGET_SIZE)
    train_ds.append([np.array(img), np.zeros(TARGET_SIZE, dtype=np.uint8), 0])

# Data Split
random.shuffle(test_ds)
split = int(0.5 * len(test_ds))
train_data = train_ds + test_ds[:split]
val_data = test_ds[split:]

print(f'Final Training size: {len(train_data)}')
print(f'Final Validation size: {len(val_data)}')

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=[-15.0, 15.0], p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


class CableDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_array, mask_array, label = self.data[idx]

        if self.transform:
            augmented = self.transform(image=img_array, mask=mask_array)
            img_tensor = augmented['image'].float()  # Zabezpieczenie typu
            mask_tensor = augmented['mask'].float()
        else:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_array).float() / 255.0

        mask_tensor = (mask_tensor > 0.5).float()
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor


train_loader = DataLoader(CableDataset(train_data, transform=train_transform), batch_size=8, shuffle=True)
val_loader = DataLoader(CableDataset(val_data, transform=val_transform), batch_size=1)

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

criterion_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
criterion_bce = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)


def calculate_iou(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.4).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if union == 0:
        return 1.0
    return (intersection / (union + 1e-7)).item()


num_epochs = 50
best_iou = 0.0
model_save_path = os.path.join(BASE_DIR, "cable_model.pth")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion_dice(outputs, masks) + criterion_bce(outputs, masks)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_ious = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_ious.append(calculate_iou(outputs.cpu(), masks))

    avg_val_iou = np.mean(val_ious)

    print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

    if avg_val_iou > best_iou:
        print(f"New Best: {best_iou:.4f} > {avg_val_iou:.4f} Saved")
        best_iou = avg_val_iou
        torch.save(model.state_dict(), model_save_path)

print("=========================================")
print(f"IoU: {best_iou:.4f}")
print(f"Model: {model_save_path}")