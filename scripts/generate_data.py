"""
训练数据生成脚本
--------------
用系统中文字体批量生成"干净书法图"作为训练数据的 ground truth。

为什么这样做？
    1. 碑帖去噪需要 (带噪, 干净) 配对，但真实碑帖没有"干净版"
    2. 学术界标准做法：干净图 + 合成噪声 = 训练对
    3. 用字体生成干净图：量大、可控、无版权问题

生成策略：
    - 每张图随机选一段中文文本
    - 随机选字体（楷体/仿宋，最像碑帖）
    - 随机字号、位置、轻微旋转
    - 输出灰度图 256×256（训练用 patch）
    - 同时输出 512×512 大图（测试用）

用法:
    python scripts/generate_data.py --num 500 --output data/raw
"""

import os
import sys
import platform
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 常用汉字（3500常用字里挑的，涵盖笔画简单到复杂）
COMMON_CHARS = (
    "人大中国时会学出年地方生以行工有天月所为上下产面这同什经和自"
    "发来理家水子用在不他就到说事对问到高然后和小而地道着也三是年"
    "风起无花春山夜月雪云水石松竹梅兰菊鹤鸣唱清明静远深长流光阴"
    "书画诗词歌赋文章墨笔纸砚观临碑帖拓印刻石古今史志贤圣道德"
    "仁义礼智信忠孝悌慈爱和平福寿康宁永乐天人合一心如止水"
    "剑气箫声琴韵棋局太极阴阳五行八卦乾坤震巽坎离艮兑"
    "龙虎凤鸾麒麟玄武朱雀白虎青龙金木水火土日月星辰"
    "春风化雨秋水长天白云苍狗沧海桑田物是人非事如春梦"
)

# 根据操作系统选择字体路径
def get_font_paths():
    """自动检测可用的中文字体"""
    if platform.system() == 'Windows':
        candidates = [
            "C:/Windows/Fonts/simkai.ttf",      # 楷体
            "C:/Windows/Fonts/STKAITI.TTF",      # 华文楷体
            "C:/Windows/Fonts/simfang.ttf",      # 仿宋
            "C:/Windows/Fonts/STFANGSO.TTF",     # 华文仿宋
        ]
    else:
        # Linux: 尝试常见路径
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",       # 文泉驿正黑
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",     # 文泉驿微米黑
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
    
    found = [p for p in candidates if os.path.exists(p)]
    
    if not found:
        # 最后的兜底：用 PIL 默认字体（不支持中文，但至少不崩）
        # 尝试 fc-list 找字体
        if platform.system() != 'Windows':
            import subprocess
            try:
                result = subprocess.run(['fc-list', ':lang=zh', 'file'],
                                       capture_output=True, text=True, timeout=5)
                for line in result.stdout.strip().split('\n'):
                    path = line.split(':')[0].strip()
                    if path and os.path.exists(path):
                        found.append(path)
                        if len(found) >= 3:
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    
    if not found:
        print("[WARNING] 没找到中文字体！请安装：")
        print("  Ubuntu/Debian: sudo apt install fonts-wqy-zenhei")
        print("  CentOS/RHEL:   sudo yum install wqy-zenhei-fonts")
        sys.exit(1)
    
    print(f"[Font] 找到 {len(found)} 个可用字体: {[os.path.basename(f) for f in found]}")
    return found


FONT_PATHS = None  # 延迟初始化，在 main() 里调用 get_font_paths()


def get_random_text(min_chars=1, max_chars=12):
    """随机取一段中文"""
    length = random.randint(min_chars, max_chars)
    return ''.join(random.choices(COMMON_CHARS, k=length))


def generate_single_char_image(size=256):
    """生成单字大图（模拟碑帖拓片里的大字）"""
    char = random.choice(COMMON_CHARS)
    font_path = random.choice(FONT_PATHS)
    
    # 字号占图的 60%-85%
    font_size = int(size * random.uniform(0.6, 0.85))
    font = ImageFont.truetype(font_path, font_size)
    
    # 白底黑字（碑帖拓片通常是黑底白字，但训练时白底黑字更通用）
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    
    # 计算文字位置（居中 + 随机偏移）
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2 + random.randint(-10, 10)
    y = (size - text_h) // 2 + random.randint(-10, 10)
    
    draw.text((x, y), char, fill=0, font=font)
    
    return img, char


def generate_multi_char_image(size=256):
    """生成多字图（模拟碑帖里一行/几列字）"""
    font_path = random.choice(FONT_PATHS)
    
    # 随机布局：竖排（传统）或横排
    vertical = random.random() > 0.5
    
    if vertical:
        # 竖排 1-3 列
        cols = random.randint(1, 3)
        chars_per_col = random.randint(2, 5)
        font_size = int(size / max(chars_per_col + 1, cols + 1) * 0.8)
    else:
        # 横排 1-4 行
        rows = random.randint(1, 4)
        chars_per_row = random.randint(2, 6)
        font_size = int(size / max(rows + 1, chars_per_row) * 0.8)
    
    font_size = max(font_size, 20)
    font = ImageFont.truetype(font_path, font_size)
    
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    
    if vertical:
        # 竖排书写（从右到左）
        col_width = size // (cols + 1)
        for col in range(cols):
            x = size - (col + 1) * col_width
            for row in range(chars_per_col):
                y = int((row + 0.5) * size / (chars_per_col + 1))
                char = random.choice(COMMON_CHARS)
                draw.text((x, y), char, fill=0, font=font)
    else:
        # 横排书写
        row_height = size // (rows + 1)
        for row in range(rows):
            y = int((row + 0.5) * row_height)
            for col in range(chars_per_row):
                x = int((col + 0.5) * size / (chars_per_row + 1))
                char = random.choice(COMMON_CHARS)
                draw.text((x, y), char, fill=0, font=font)
    
    return img


def generate_calligraphy_style(img):
    """给生成的文字图加一些"书法感"（可选的后处理）
    
    - 轻微模糊模拟毛笔边缘
    - 随机轻微变形模拟手写不规则
    """
    # 轻微高斯模糊（模拟毛笔的柔和边缘）
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    
    return img


def invert_image(img):
    """反转图像（白底黑字 → 黑底白字）
    
    碑帖拓片是黑底白字，50%概率生成这种。
    """
    arr = np.array(img)
    arr = 255 - arr
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description='生成训练数据')
    parser.add_argument('--num', type=int, default=500, help='生成图片数量')
    parser.add_argument('--size', type=int, default=256, help='图片尺寸')
    parser.add_argument('--output', type=str, default='data/raw', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子(可复现)')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化字体路径
    global FONT_PATHS
    FONT_PATHS = get_font_paths()
    
    print(f"开始生成 {args.num} 张训练图...")
    print(f"  尺寸: {args.size}×{args.size}")
    print(f"  输出: {args.output}/")
    print(f"  字体: {len(FONT_PATHS)} 种")
    print()
    
    for i in range(args.num):
        # 60% 单字大图，40% 多字图
        if random.random() < 0.6:
            img, _ = generate_single_char_image(args.size)
        else:
            img = generate_multi_char_image(args.size)
        
        # 加书法后处理
        img = generate_calligraphy_style(img)
        
        # 50% 概率反转为黑底白字（碑帖风格）
        if random.random() > 0.5:
            img = invert_image(img)
        
        # 保存
        fname = f"train_{i:04d}.png"
        img.save(os.path.join(args.output, fname))
        
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i+1}/{args.num}")
    
    print(f"\n[Done] {args.num} 张图片已保存到 {args.output}/")
    print(f"  现在可以开始训练: python train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
