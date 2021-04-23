from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import utils as vutils

import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.5, 0, 0, 0, 0.5, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        save_image_tensor(x[0].unsqueeze(0), "src.jpg")
        x = self.stn(x)
        save_image_tensor(x[0].unsqueeze(0), "target.jpg")

        # Perform the usual forward passn
        import pdb; pdb.set_trace()

        return x

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = Net().to(device)
    model = Net()

    print("test begin")

    for ii in range(64):
        src = "256.jpg"
        # img0 = cv2.imread(src)
        import matplotlib.image as mpimg
        img0 = mpimg.imread(src)
        img0 = img0[10:224+10, 100:224+100, :]

        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()
        # img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        if(ii==0):
            img_batch = img
        else:
            img_batch = torch.cat((img_batch,img),dim=0)
        
        # img_batch.to(device)

    print(img_batch.shape)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for epoch in range(10):
            output = model(img_batch)
    
    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            y1 = stn1(x1)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]


    print("test end")