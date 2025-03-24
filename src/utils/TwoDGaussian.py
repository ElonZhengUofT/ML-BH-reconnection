import numpy as np
from scipy.ndimage import gaussian_filter


def gaussianize_image(image, sigma=15):
    """
    对输入图像应用高斯滤波，将二值标签转化为2D Gaussian热图。

    参数:
      image: 输入的二维 numpy 数组，通常是二值图（目标点为1，其它为0）。
      sigma: 高斯滤波的标准差，控制扩散范围，默认3。

    返回:
      经过高斯平滑处理后的图像（二维 numpy 数组），其峰值区域呈现2D Gaussian形状。
    """
    # 直接使用 gaussian_filter 对图像进行平滑
    return gaussian_filter(image, sigma=sigma)


# 测试示例：
if __name__ == '__main__':
    # 创建一个全0的图像，大小为 100x100
    img = np.zeros((100, 100), dtype=np.float32)
    # 在图像中央设置一个目标点
    img[50, 50] = 1.0

    # 对图像应用高斯平滑
    sigma = 5  # 可根据需要调整sigma以改变扩散范围
    gaussian_img = gaussianize_image(img, sigma=sigma)

    # 显示结果
    import matplotlib.pyplot as plt

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(gaussian_img, cmap='hot')
    plt.title(f"Gaussianized (sigma={sigma})")
    plt.show()


