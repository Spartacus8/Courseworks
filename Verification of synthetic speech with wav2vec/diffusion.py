import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler

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

def train_and_decode_with_diffusion(image_folder, block_size=64, epochs=100, batch_size=32, learning_rate=1e-4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建未打码图像块的数据集
    non_mosaic_dataset = NonMosaicDataset(os.path.join(image_folder, 'non_mosaics'), transform=transform, block_size=block_size)
    non_mosaic_dataloader = DataLoader(non_mosaic_dataset, batch_size=batch_size, shuffle=True)

    # 初始化Diffusion模型
    model = UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    # 初始化DDPM调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 初始化loss变量
    loss = torch.tensor(0.0)
    # 训练Diffusion模型
    for epoch in range(epochs):
        for blocks in non_mosaic_dataloader:
            if not blocks:
                continue  # 如果blocks为空,则跳过该批次
            blocks = torch.stack(blocks).squeeze(1)
            noise = torch.randn(blocks.shape)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (blocks.shape[0],)).long()

            noisy_blocks = noise_scheduler.add_noise(blocks, noise, timesteps)
            noise_pred = model(noisy_blocks, timesteps, return_dict=False)[0]
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 使用训练好的模型对马赛克部分进行去码
    mosaic_folder = os.path.join(image_folder, 'mosaics')
    decoded_folder = os.path.join(image_folder, 'decensored_diffusion')
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
                decoded_blocks = [noise_scheduler.add_noise(block, torch.randn_like(block), torch.tensor([noise_scheduler.num_train_timesteps - 1])) for block in mosaic_blocks]
                decoded_blocks = [model(block, torch.tensor([noise_scheduler.num_train_timesteps - 1]), return_dict=False)[0] for block in decoded_blocks]
                decoded_blocks = [(block.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255 for block in decoded_blocks]
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
    print("Training and decoding with Diffusion model completed.")

