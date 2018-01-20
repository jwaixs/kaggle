import torch
torch.manual_seed(37)

from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from carvana import CARVANA
from unet import unet_256_small, unet_256
from criterion import diceLoss, BCELossLogits2d

train_dataset = CARVANA(
    root = '/data/noud/kaggle/carvana',
    subset = 'small_test',
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = 4,
    shuffle = True,
    pin_memory = True,
    num_workers = 4
)

model = unet_256_small().cuda()
criterion = {
    'loss' : diceLoss(),
    'acc' : diceLoss()
}
optimizer = torch.optim.SGD(
    model.parameters(),
    weight_decay = 0.05,
    lr = 0.05,
    momentum = 0.99
)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    lloss = list()
    pbar = tqdm(train_loader)
    for inputs, targets in pbar:
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion['loss'](outputs, targets)
        loss.backward()
        optimizer.step()

        lloss.append(loss.data[0])
        pbar.set_description('Epoch: {} Loss: {:.4f} Learning rate: {:.4f}'.format(
            epoch, sum(lloss) / len(lloss), optimizer.param_groups[0]['lr']
        ))

    return model

for epoch in range(100):
    model = train(train_loader, model, criterion, optimizer, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.9

from PIL import Image
inputs, targets = next(iter(train_loader))
for i in range(4):
    targets /= targets.max()
    ret2 = transforms.ToNumpy()(targets)
    i2 = Image.fromarray(255 * ret2[i][0])
    i2.show()
    outputs = model(Variable(inputs.cuda()))
    ret = transforms.ToNumpy()(outputs)
    i1 = Image.fromarray(255 * ret[i][0] / ret.max())
    i1.show()
