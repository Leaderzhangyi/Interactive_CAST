'''
@Description:	xxx
@Date:	2023/10/16 16:22:14
@Author:	ZinkCas
@version:	1.0
'''
# Receptive Field?
# 卷积神经网络每一层输出的特征图（feature map）上的像素点映射回输入图像上的区域大小。
import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys 
sys.path.append("/home/ps/mylin/Interactive_CAST")
from models import net

# 加载预训练的VGG模型
vgg = net.vgg
vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))

# 获取最后一层卷积层
last_conv_layer = vgg[-2]


# 加载一张真实照片
image_path = '/home/ps/mylin/Interactive_CAST/datasets/dh/testB/0.jpg'
image = Image.open(image_path)

# 预处理图像以匹配VGG的要求
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_image = preprocess(image)
input_image = input_image.unsqueeze(0)  # 增加批次维度

# 将图像传递到最后一层卷积层
# activation = vgg.features(input_image)
activation = vgg.forward(input_image)


# 计算感受野
def calculate_receptive_field(layer):
    receptive_field = 1
    stride = 1
    for module in reversed(list(vgg.children())):
        if module == layer:
            break
        if isinstance(module, nn.Conv2d):
            receptive_field += (module.kernel_size[0] - 1) * stride
            stride *= module.stride[0]
    return receptive_field


# 计算感受野
receptive_field = calculate_receptive_field(last_conv_layer)
print(f"感受野大小: {receptive_field}")

# 可视化感受野
plt.imshow(np.array(image))
plt.gca().add_patch(plt.Rectangle((0, 0), receptive_field, receptive_field, edgecolor='r', facecolor='none', lw=2))
plt.savefig("./iamge.png",dpi = 80)
