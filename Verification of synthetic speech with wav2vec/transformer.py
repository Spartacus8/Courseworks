import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16

class NonMosaicDataset(Dataset):
    def __init__(self, image_folder, transform=None, block_size=64):
        self.image_folder = image_folder
        self.transform = transform
        self.block_size = block_size
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = cv2.imread(img_path)
        mask_path = os.path.join(self.image_folder, "../mosaics", self.image_files[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 找到未打码区域的边界
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        # 裁剪未打码区域
        non_mosaic = image[y:y+h, x:x+w]

        # 将未打码区域分割成多个矩形块
        blocks = []
        for i in range(0, non_mosaic.shape[0], self.block_size):
            for j in range(0, non_mosaic.shape[1], self.block_size):
                block = non_mosaic[i:i+self.block_size, j:j+self.block_size]
                if block.shape[0] == self.block_size and block.shape[1] == self.block_size:
                    # 检查矩形块是否与打码区域有重叠
                    block_mask = mask[y+i:y+i+self.block_size, x+j:x+j+self.block_size]
                    if cv2.countNonZero(block_mask) == 0:
                        blocks.append(block)

        # 对每个块应用转换
        if self.transform:
            blocks = [self.transform(block) for block in blocks]

        return blocks

def train_and_decode_with_transformer(image_folder, block_size=64, epochs=10, batch_size=32, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 创建未打码图像块的数据集
    non_mosaic_dataset = NonMosaicDataset(os.path.join(image_folder, 'non_mosaics'), transform=transform, block_size=block_size)
    non_mosaic_dataloader = DataLoader(non_mosaic_dataset, batch_size=batch_size, shuffle=True)

    # 初始化Vision Transformer模型
    model = vit_b_16(num_classes=3 * block_size * block_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练Vision Transformer模型
    for epoch in range(epochs):
        for blocks in non_mosaic_dataloader:
            blocks = torch.stack(blocks).squeeze(1).view(-1, 3, block_size, block_size)
            optimizer.zero_grad()
            outputs = model(blocks)
            loss = criterion(outputs, blocks.view(-1, 3 * block_size * block_size))
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 使用训练好的模型对马赛克部分进行去码
    mosaic_folder = os.path.join(image_folder, 'mosaics')
    decoded_folder = os.path.join(image_folder, 'decensored_transformer')
    os.makedirs(decoded_folder, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for filename in os.listdir(mosaic_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                mosaic_path = os.path.join(mosaic_folder, filename)
                mosaic_image = cv2.imread(mosaic_path)
                mask = cv2.imread(os.path.join(mosaic_folder, filename), cv2.IMREAD_GRAYSCALE)

                # 找到打码区域的边界
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])

                # 裁剪打码区域
                mosaic_cropped = mosaic_image[y:y+h, x:x+w]

                # 将打码区域分割成多个矩形块
                mosaic_blocks = []
                for i in range(0, mosaic_cropped.shape[0], block_size):
                    for j in range(0, mosaic_cropped.shape[1], block_size):
                        block = mosaic_cropped[i:i+block_size, j:j+block_size]
                        if block.shape[0] == block_size and block.shape[1] == block_size:
                            mosaic_blocks.append(block)

                # 对每个块应用转换并进行去码
                mosaic_blocks = [transform(block).unsqueeze(0) for block in mosaic_blocks]
                decoded_blocks = [model(block).view(3, block_size, block_size).permute(1, 2, 0).numpy() for block in mosaic_blocks]
                decoded_blocks = [(block * 0.5 + 0.5) * 255 for block in decoded_blocks]
                decoded_blocks = [block.astype('uint8') for block in decoded_blocks]

                # 将去码后的块拼接回原图
                decoded_image = mosaic_image.copy()
                idx = 0
                for i in range(0, h, block_size):
                    for j in range(0, w, block_size):
                        if i + block_size <= h and j + block_size <= w:
                            decoded_image[y+i:y+i+block_size, x+j:x+j+block_size] = decoded_blocks[idx]
                            idx += 1

                decoded_path = os.path.join(decoded_folder, filename)
                cv2.imwrite(decoded_path, decoded_image)

    print("Training and decoding with Vision Transformer completed.")