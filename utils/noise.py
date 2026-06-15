"""
噪声生成模块
-----------
碑帖图像常见噪声类型：
1. 高斯噪声 —— 拓印过程中纸张纤维产生的均匀细颗粒
2. 椒盐噪声 —— 碑面风化形成的随机黑白点
3. 拓印纹理噪声 —— 墨水不均匀造成的斑驳、晕染（这是碑帖特有的！）

思路：干净书法图 + 合成噪声 = 训练对
    ground truth = 干净图
    input        = 干净图 + 噪声
"""

import numpy as np
import cv2


def add_gaussian_noise(img, sigma_range=(10, 50)):
    """添加高斯噪声。
    
    参数:
        img: numpy数组, 范围[0,1], shape=(H,W) 或 (H,W,C)
        sigma_range: 噪声强度范围(随机取)，值越大噪声越猛
            - 10-20: 轻微噪声
            - 20-40: 中等噪声
            - 40-50: 重度噪声
    
    原理:
        每个像素加一个服从 N(0, sigma²) 的随机值
        sigma 越大，随机值偏离越远，噪声越明显
    """
    sigma = np.random.uniform(*sigma_range) / 255.0  # 归一化到[0,1]范围
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    noisy = img + noise
    return np.clip(noisy, 0, 1)


def add_salt_pepper_noise(img, amount_range=(0.01, 0.05)):
    """添加椒盐噪声。
    
    参数:
        img: numpy数组, 范围[0,1]
        amount_range: 噪声像素占比范围
            - 0.01: 1%的像素变成黑/白点
            - 0.05: 5%的像素变成黑/白点
    
    原理:
        随机选一些像素，一半变成0(黑=椒)，一半变成1(白=盐)
        碑帖上经常有这种——碑面破损形成的随机白点/黑点
    """
    amount = np.random.uniform(*amount_range)
    noisy = img.copy()
    
    # 生成随机 mask
    h, w = img.shape[:2]
    num_pixels = int(h * w * amount)
    
    # 椒（黑点）
    coords_y = np.random.randint(0, h, num_pixels)
    coords_x = np.random.randint(0, w, num_pixels)
    noisy[coords_y, coords_x] = 0
    
    # 盐（白点）
    coords_y = np.random.randint(0, h, num_pixels)
    coords_x = np.random.randint(0, w, num_pixels)
    noisy[coords_y, coords_x] = 1
    
    return noisy


def add_rubbing_texture(img, intensity_range=(0.1, 0.3)):
    """添加拓印纹理噪声 —— 碑帖特有的！
    
    模拟拓片上"墨不均匀"的效果：
        拓印时纸贴在碑面上，用墨包拍打，手劲不一样、碑面凹凸不同，
        就会出现深浅不一的"斑驳感"。
    
    实现方法:
        1. 生成低频随机噪声（大块斑驳，不是细颗粒）
        2. 高斯模糊让它更"自然"（真实拓印纹理是缓慢过渡的）
        3. 叠加到原图上
    
    这个噪声模型是我们的创新点之一！传统去噪方法假设高斯白噪声，
    但碑帖的噪声是这种低频纹理+高频颗粒的混合。
    """
    intensity = np.random.uniform(*intensity_range)
    h, w = img.shape[:2]
    
    # 生成低频纹理：先做小图的随机噪声，再放大（等于低频成分）
    small_h, small_w = h // 16, w // 16
    texture = np.random.randn(small_h, small_w).astype(np.float32)
    # 放大到原尺寸（双线性插值=平滑过渡）
    texture = cv2.resize(texture, (w, h), interpolation=cv2.INTER_LINEAR)
    # 再模糊一下让过渡更自然
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=h // 8)
    # 归一化到 [-1, 1]
    texture = texture / (texture.std() + 1e-8)
    
    noisy = img + texture * intensity
    return np.clip(noisy, 0, 1).astype(np.float32)


def add_composite_noise(img, noise_level='medium'):
    """组合噪声 —— 模拟真实碑帖的复杂退化。
    
    真实碑帖上不会只有一种噪声，通常是几种叠加：
        轻度: 轻微高斯 + 少量椒盐
        中度: 中等高斯 + 椒盐 + 轻微拓印纹理
        重度: 重度高斯 + 椒盐 + 重度拓印纹理
    
    参数:
        noise_level: 'light' / 'medium' / 'heavy'
    """
    configs = {
        'light': {
            'gaussian_sigma': (5, 15),
            'sp_amount': (0.005, 0.015),
            'rubbing_intensity': (0.03, 0.08),
        },
        'medium': {
            'gaussian_sigma': (15, 30),
            'sp_amount': (0.01, 0.03),
            'rubbing_intensity': (0.1, 0.2),
        },
        'heavy': {
            'gaussian_sigma': (30, 50),
            'sp_amount': (0.03, 0.06),
            'rubbing_intensity': (0.2, 0.4),
        },
    }
    cfg = configs[noise_level]
    
    noisy = add_gaussian_noise(img, cfg['gaussian_sigma'])
    noisy = add_salt_pepper_noise(noisy, cfg['sp_amount'])
    noisy = add_rubbing_texture(noisy, cfg['rubbing_intensity'])
    
    return noisy.astype(np.float32)


if __name__ == "__main__":
    """测试：造一张假图，加各种噪声，保存看看效果"""
    import os
    os.makedirs("results", exist_ok=True)
    
    # 造一张 256x256 灰度"假碑帖"（白底黑字，模拟简单笔画）
    img = np.ones((256, 256), dtype=np.float32)  # 白底
    # 画几条"笔画"
    cv2.line(img, (50, 50), (200, 200), 0, 3)
    cv2.line(img, (50, 200), (200, 50), 0, 3)
    cv2.putText(img, "TEST", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    
    # 加各种噪声并保存
    cv2.imwrite("results/noise_clean.png", (img * 255).astype(np.uint8))
    
    noisy_gauss = add_gaussian_noise(img.copy(), (30, 30))
    cv2.imwrite("results/noise_gaussian.png", (noisy_gauss * 255).astype(np.uint8))
    
    noisy_sp = add_salt_pepper_noise(img.copy(), (0.03, 0.03))
    cv2.imwrite("results/noise_saltpepper.png", (noisy_sp * 255).astype(np.uint8))
    
    noisy_rub = add_rubbing_texture(img.copy(), (0.2, 0.2))
    cv2.imwrite("results/noise_rubbing.png", (noisy_rub * 255).astype(np.uint8))
    
    noisy_all = add_composite_noise(img.copy(), 'medium')
    cv2.imwrite("results/noise_composite.png", (noisy_all * 255).astype(np.uint8))
    
    print("噪声测试图已保存到 results/ 文件夹")
    print("  noise_clean.png      - 干净原图")
    print("  noise_gaussian.png   - 高斯噪声")
    print("  noise_saltpepper.png - 椒盐噪声")
    print("  noise_rubbing.png    - 拓印纹理噪声")
    print("  noise_composite.png  - 组合噪声(真实模拟)")
