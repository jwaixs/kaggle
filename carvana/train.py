import torch

from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from carvana import CARVANA
from unet import unet_256
from criterion import diceLoss

train_dataset = CARVANA(
    root = '/data/noud/kaggle/carvana',
    subset = 'train',
    transform = transforms.Compose([
        transforms.Scale(1024),
        transforms.RandomCrop((1024, 1024)),
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

model = unet_256().cuda()
criterion = {
    'loss' : diceLoss(),
    'acc' : diceLoss()
}
optimizer = torch.optim.SGD(
    model.parameters(),
    weight_decay = 0.05,
    lr = 0.001,
    momentum = 0.99
)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    pbar = tqdm(train_loader)
    for inputs, targets in pbar:
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion['loss'](outputs, targets)
        loss.backward()
        optimizer.step()

        print(loss)

for epoch in range(5):
    print(epoch)

    train(train_loader, model, criterion, optimizer, epoch)

