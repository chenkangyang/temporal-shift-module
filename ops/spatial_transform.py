"""
仅支持224的图像输入
参数：通道数最宽64，空间通过两个大卷积降维到52
最好调一调，没有res结构，网络不宜过宽过深，空间上应该添加更多卷积，卷到足够小再avgpool到1x1
毕竟将224*224*3 的特征 ===映射===> 2*3 的仿射矩阵；

目前通道数为：3-64-32-(32*52*52)-32-6, 通道设计成bottleneck那样较好，节省参数，现在是哥窄口瓶子
空间采样：224--conv-->218--max-->109--conv-->105--max-->52 
显然卷积核感受野较小(18)，最好保证最后的空间感受野覆盖224区域内一个人的大小

所以该模块网络结构比较僵硬，transfer效果应该一般


========== 1.18 更新 =============
学到了两个设计原则：3个3卷积能达到1个7卷积的感受野，且参数量小
当输出特征图尺寸减半时，输出特征图的通道数应该加倍，这样保证相邻卷积层所包含的信息量不会相差太大
用resnet18做仿射矩阵的特征提取器不是很稳健？

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import utils as vutils
 
resnet18 = getattr(torchvision.models, "resnet18")(pretrained = False)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 6)

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


class SpatialTransform(nn.Module):
    # *在任意模块前加上 Spatial Transform

    def __init__(self, net):
        super(SpatialTransform, self).__init__()

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
        self.net = net
    



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

        return self.net(x)



class STN_API(nn.Module):
    def __init__(self):
        super(STN_API, self).__init__()

    def maxtrix_padding(self, mat, size):
        new_mat = torch.zeros(size).cuda()
        new_mat[:, -1, -1] = 1
        new_mat[:, :mat.size(1), :] = mat
        return new_mat

    def matrix_reduce(self, mat, size):
        return mat[:size[0], :size[1], :size[2]]

    def rotate(self, max_degree=45):
        degree = np.random.randint(0, max_degree)
        angle = -degree * math.pi / 180
        mat = torch.zeros((2, 3)).cuda()
        mat[0, 0] = math.cos(angle)
        mat[1, 1] = math.cos(angle)
        mat[0, 1] = -math.sin(angle)
        mat[1, 0] = math.sin(angle)
        return mat

    def scale(self, max_scale=1.15, min_scale=0.85):
        scale_x = random.random() * (max_scale - min_scale) + min_scale
        scale_y = random.random() * (max_scale - min_scale) + min_scale
        mat = torch.zeros((2, 3)).cuda()
        mat[0, 0] = math.cos(scale_x)
        mat[1, 1] = math.cos(scale_y)
        return mat

    def translation(self, up_bound_x=0.3, up_bound_y=0.3):
        tran_x = random.random() * up_bound_x
        tran_y = random.random() * up_bound_y
        mat = torch.zeros((2, 3)).cuda()
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[0, 2] = tran_x
        mat[1, 2] = tran_y
        return mat

    def shear(self, up_bound_x=0.3, up_bound_y=0.3):
        mat = torch.zeros((2, 3)).cuda()
        shear_x = random.random() * up_bound_x
        shear_y = random.random() * up_bound_y
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[0, 1] = shear_x
        mat[1, 0] = shear_y
        return mat

    def reflection(self):
        mat = torch.zeros((2, 3)).cuda()
        prob = random.random()
        if prob < 0.5:
            mat[0, 0] = -1
            mat[1, 1] = 1
        else:
            mat[0, 0] = 1
            mat[1, 1] = -1
        return mat

    # random generate affnix matrix
    def forward(self, x):
        bsz, c, t, h, w = x.size()
        output = torch.zeros_like(x).cuda()
        theta = torch.zeros((bsz, 2, 3)).cuda()
        for j in range(bsz):
            # theta[j] = self.rotate()
            theta[j] = self.scale()
            # theta[j] = self.translation()
            # theta[j] = self.shear()
            # theta[j] = self.reflection()
        grid = F.affine_grid(theta, (bsz, c, h, w))
        for i in range(t):
            output[:, :, i, :, :] = F.grid_sample(x[:, :, i, :, :], grid)
        # new_theta = self.maxtrix_padding(theta, (bsz, 3, 3))
        # # for each sample in batch size, should get inverse
        # inverse_theta = new_theta.clone()
        # for j in range(bsz):
        #     inverse_theta[j] = torch.inverse(new_theta[j])
        # inverse_theta = self.matrix_reduce(inverse_theta, (bsz, 2, 3))
        # inverse_grid = F.affine_grid(inverse_theta, (bsz, c, h, w))
        # for i in range(t):
        #     output[:, :, i, :, :] = F.grid_sample(x[:, :, i, :, :], inverse_grid)
        return output


def make_spatial_transform(net):
    # 在 ResNet 第一层前加上 Spatial Transform
    # import pdb; pdb.set_trace()
    if isinstance(net, torchvision.models.ResNet):

        def add_stn(layer):
            layer = SpatialTransform(layer)
            return layer

        net.conv1 = add_stn(net.conv1)
    
    else:
        raise NotImplementedError("net.name")

if __name__ == '__main__':

    # stn1 = SpatialTransform(nn.Sequential())

    # print('=> Testing CPU...')
    # # test forward
    # with torch.no_grad():
    #     for i in range(10):
    #         x = torch.rand(2 * 8, 3, 224, 224)
    #         y = stn1(x)

    # # test backward
    # with torch.enable_grad():
    #     for i in range(10):
    #         x1 = torch.rand(2 * 8, 3, 224, 224)
    #         x1.requires_grad_()
    #         y1 = stn1(x1)
    #         grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]

    # print('=> Testing GPU...')
    # stn1.cuda()
    # # test forward
    # with torch.no_grad():
    #     for i in range(10):
    #         x = torch.rand(2 * 8, 3, 224, 224).cuda()
    #         y1 = stn1(x)

    # # test backward
    # with torch.enable_grad():
    #     for i in range(10):
    #         x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
    #         x1.requires_grad_()
    #         y1 = stn1(x1)
    #         grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]

    # print('Test passed.')


    # =============================================


    print('=== Spatial Transform ResNet ===')

    net = getattr(torchvision.models, "resnet50")(True)

    print("Origin Net:", net)
    make_spatial_transform(net)
    stn2 = net
    print("\n\nST ResNet:", net)
    # import pdb; pdb.set_trace()

    print('=> Testing CPU...')


    # test forward
    print('=> Testing forward...')

    with torch.no_grad():
        for i in range(10):
            print(i)
            x = torch.rand(2 * 8, 3, 224, 224) # (16, 3, 224, 224)
            y = stn2(x) # (16, 1000)
            print(y.shape)
            # import pdb; pdb.set_trace()


    # test backward
    print('=> Testing backword...')

    with torch.enable_grad():
        for i in range(10):
            print(i)
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            y1 = stn2(x1)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]

    print('=> Testing GPU...')
    stn2.cuda()
    # test forward
    print('=> Testing forward...')

    with torch.no_grad():
        for i in range(10):
            print(i)
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = stn2(x)

    # test backward
    print('=> Testing backward...')

    with torch.enable_grad():
        for i in range(10):
            print(i)
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            y1 = stn2(x1)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]

    print('Test passed.')

