import cv2
import numpy as np
import os

def detect_and_segment_mosaic(img_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mosaic'), exist_ok=True) 
    os.makedirs(os.path.join(output_dir, 'non_mosaic'), exist_ok=True)
    
    # 生成不同尺寸的网格模板
    grid_sizes = range(11, 21) 
    grid_templates = []
    for size in grid_sizes:
        template = np.zeros((size, size), dtype=np.uint8)
        template[::size-1] = 255
        template[:,::size-1] = 255
        grid_templates.append(template)
    
    # 遍历所有图片
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        
        # 转为灰度图并边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 10, 20)
        
        # 反转边缘图像并模糊 
        edges = 255 - edges
        edges = cv2.GaussianBlur(edges, (3,3), 0)
        
        # 多尺度模板匹配
        match_results = []
        for template in grid_templates:
            match_result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
            match_results.append(match_result)
        match_result = np.max(match_results, axis=0)
        
        # 阈值化得到mosaic mask
        _, mask = cv2.threshold(match_result, 0.3, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        
        # 用mask分割马赛克和非马赛克部分
        mosaic = cv2.bitwise_and(img, img, mask=mask)
        non_mosaic = cv2.bitwise_and(img, img, mask=255-mask)
        
        # 保存结果
        cv2.imwrite(os.path.join(output_dir, 'mosaic', filename), mosaic)
        cv2.imwrite(os.path.join(output_dir, 'non_mosaic', filename), non_mosaic)