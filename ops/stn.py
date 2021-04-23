import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import utils as vutils
 
resnet18 = getattr(torchvision.models, "resnet18")(pretrained = False)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 6)


class STN(nn.Module):
    # *在任意模块前加上 Spatial Transform

    def __init__(self):
        super(STN, self).__init__()

        """
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 52 * 52, 32), #! c*h*w
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.5, 0, 0, 0, 0.5, 0], dtype=torch.float))

        # *用Res18结构来做特征提取器那不是很稳健（比自己设计的navie模型要重，但是效果可能好一点），而且还有预训练好的参数做
        self.resnet18 = getattr(torchvision.models, "resnet18")(pretrained = False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 6)
        """


    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        
        # import pdb; pdb.set_trace()
        xs = xs.view(-1, 32 * 52 * 52) #! c*h*w
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


    def st_res18(self, x):
        
        theta = resnet18(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


    def forward(self, x):
        # import pdb; pdb.set_trace()
        # save_image_tensor(x[0].unsqueeze(0), "before_st.jpg")
        # x = self.stn(x) # * Naive版STN
        x = self.st_res18(x) # * res18提取仿射阵
        # save_image_tensor(x[0].unsqueeze(0), "after_st.jpg")

        return x

if __name__ == "__main__":
    stn = STN()

    print('=> Testing CPU...')

    # test forward
    print('=> Testing forward...')

    with torch.no_grad():
        for i in range(10):
            print(i)
            x = torch.rand(2 * 8, 3, 224, 224) # (16, 3, 224, 224)
            y = stn(x) # (16, 1000)
            print(y.shape)
            # import pdb; pdb.set_trace()


    # test backward
    print('=> Testing backword...')

    with torch.enable_grad():
        for i in range(10):
            print(i)
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            y1 = stn(x1)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]