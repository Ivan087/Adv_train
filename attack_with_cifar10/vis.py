import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dct import *
import torch
def add_gaussian_noise(image, mean=0.5, std=0.05):
    """给定图像添加高斯噪声"""
    gauss = np.random.normal(mean, std, image.shape)
    noisy = image + gauss
    return np.clip(noisy, 0, 1)  # 像素值保持在0到1之间

# 创建一个纯黑的图像
# image_size = 256
# image = np.zeros((image_size, image_size))
image = Image.open("./Adv_train/attack_with_cifar10/img.png").convert("L")
image = np.array(image) / 255
# 添加高斯噪声
gaussian_noisy_image = add_gaussian_noise(image) + 0.5

def addsalt_pepper(img, SNR):
    img_ = img.copy()
    h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    # mask = np.repeat(mask, c, axis=0)     # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 1    # 盐噪声
    img_[mask == 2] = 0      # 椒噪声
    # img_[mask == 0] = 0.5   # base
    return img_

pepper_noisy_image = addsalt_pepper(image, 0.7)

def add_uniform_noise(image: np.ndarray, prob=0.05):
    """
    随机生成一个0~1的mask，作为噪声
    :param image:图像
    :param prob: 噪声比例
    :param vaule: 噪声值
    :return:
    """
    h, w = image.shape[:2]
    noise = np.random.uniform(low=0.0, high=1.0, size=(h, w)).astype(dtype=np.float32)  # 产生均匀分布的噪声
    # mask = np.zeros(shape=(h, w), dtype=np.uint8) + vaule
    output = image + noise 
    output = np.clip(output, 0, 1)
    # output = np.uint8(output)
    return output

uniform_noisy_image = add_uniform_noise(image)

def fft_2d(image):
    # 应用傅立叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # 移动零频率分量到中心

    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift)+0.01)

    return magnitude_spectrum
# 显示高斯噪声图像
plt.figure(figsize=(10, 5))
plt.subplot(2, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示高斯噪声图像
plt.subplot(2, 4, 2)
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.title('Gaussian Noise')
plt.axis('off')

# 显示椒盐噪声图像
plt.subplot(2, 4, 3)
plt.imshow(pepper_noisy_image, cmap='gray')
plt.title('Salt & Pepper Noise')
plt.axis('off')

# 显示椒盐噪声图像
plt.subplot(2, 4, 4)
plt.imshow(uniform_noisy_image, cmap='gray')
plt.title('Uniform Noise')
plt.axis('off')

##
plt.subplot(2, 4, 5)
plt.imshow(fft_2d(image),  cmap='gray')
# plt.title('Salt & Pepper Noise')
plt.axis('off')


plt.subplot(2, 4, 6)
plt.imshow(fft_2d(gaussian_noisy_image), cmap='gray')
# plt.title('Salt & Pepper Noise')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(fft_2d(pepper_noisy_image), cmap='gray')
# plt.title('Salt & Pepper Noise')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(fft_2d(uniform_noisy_image), cmap='gray')
# plt.title('Salt & Pepper Noise')
plt.axis('off')
plt.show()



##########################
x_dct = dct_2d(torch.tensor(image).unsqueeze(0),'ortho').squeeze(0)
mask = (torch.rand_like(x_dct)* 2 * 0.3 + 1 - 0.3)
print(x_dct.min(),x_dct.max())
noise = np.random.uniform(low=x_dct.min(), high=x_dct.max()/2, size=x_dct.size()).astype(dtype=np.float32)
# noise = np.random.uniform(low=0.0, high=1.0, size=x_dct.size()).astype(dtype=np.float32)
dct_spectrum = np.log(np.abs(x_dct + noise) + 1e-10)  

plt.subplot(141)
plt.imshow(image,cmap='gray')
plt.title('Original')
plt.axis('off')

# DCT频谱图
plt.subplot(142)
plt.imshow(np.log(np.abs(x_dct)),cmap='gray')
plt.title('DCT')
plt.axis('off')

plt.subplot(143)
plt.imshow(np.log(np.abs(x_dct)*mask),cmap='gray')
plt.title('Multiplicative')
plt.axis('off')

# DCT频谱图
plt.subplot(144)
plt.imshow(dct_spectrum,cmap='gray')
plt.title('Addidative')
plt.axis('off')

plt.show()

###############################
uni_noise = np.random.uniform(low=fft_2d(image).min(), high=fft_2d(image).mean(), size=image.shape).astype(dtype=np.float32)
