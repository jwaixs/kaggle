from load_data import load_image

image, annotations = load_image('img_00003.jpg')
image = image.astype(float)
x, y, z = image.shape
image.resize(z, x, y)
image[0] = (image[0] - 0.485) / 0.229
image[1] = (image[1] - 0.456) / 0.224
image[2] = (image[2] - 0.406) / 0.225

from torchvision import models
from torch.autograd import Variable
import torch
model = models.vgg16(pretrained = True).cuda(0)

vimage = Variable(torch.Tensor([image.astype(float)]).cuda(0))
features = model.features(vimage)
