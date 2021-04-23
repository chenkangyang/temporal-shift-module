# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import sys
sys.path.append("..")
from archs import repvgg

class TemporalShift(nn.Module):
    def __init__(self, net, input_channels, n_segment=3, n_div=8, inplace=False, soft=False, init_mode="shift"):
        super(TemporalShift, self).__init__()
        self.net = net
        # self.input_channels = net.in_channels
        self.input_channels = input_channels

        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div

        self.inplace = inplace
        self.soft = soft
        self.mode = init_mode

        if self.soft:
            self.conv_shift = nn.Conv1d(
                self.input_channels, self.input_channels,
                kernel_size=3, padding=1, groups=self.input_channels,
                bias=False)
            # weight_size: (self.input_channels, 1, 3)
            # 以下是3种初始化方法
            if self.mode == 'shift':
                # import pdb; pdb.set_trace()
                self.conv_shift.weight.requires_grad = True
                self.conv_shift.weight.data.zero_()
                self.conv_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
                self.conv_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
                if 2*self.fold < self.input_channels:
                    self.conv_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
            elif self.mode == 'fixed':
                self.conv_shift.weight.requires_grad = True
                self.conv_shift.weight.data.zero_()
                self.conv_shift.weight.data[:, 0, 1] = 1 # fixed
            elif self.mode == 'norm':
                self.conv_shift.weight.requires_grad = True

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):

        if self.soft: # 可学习的 1D Temporal kernel
            nt, c, h, w = x.size()
            n_batch = nt // self.n_segment
            x = x.view(n_batch, self.n_segment, c, h, w)
            x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
            x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
            x = self.conv_shift(x)  # (n_batch*h*w, c, n_segment)
            x = x.view(n_batch, h, w, c, self.n_segment)
            x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
            x = x.contiguous().view(nt, c, h, w)
        else: 
            x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
            
        return out.view(nt, c, h, w)
   

class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, soft = False, init_mode="shift", place='blockres', temporal_pool=False, deploy=False):
    '''
    1D时序卷积参数初始化："shift" 初始化 [0,0,1] 左移；"fixed" 无偏初始化[1, 1, 1]；"norm" 随机初始化 normal
    '''
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    
    # for 0.5  resnet18 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if isinstance(net, torchvision.models.ResNet):
    # if 1:
    # for 0.5  resnet18 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    # import pdb; pdb.set_trace()

                    blocks[i] = TemporalShift(b, b.conv1.in_channels, n_segment=this_segment, n_div=n_div, soft=soft, init_mode=init_mode)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23: # > res101
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    # import pdb; pdb.set_trace()
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, b.conv1.in_channels, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*blocks)
            
            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    
    elif isinstance(net, repvgg.RepVGG):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    # import pdb; pdb.set_trace()
                    blocks[i] = TemporalShift(b, b.in_channels, n_segment=this_segment, n_div=n_div, soft=soft, init_mode=init_mode)
                return nn.Sequential(*(blocks))

            net.stage1 = make_block_temporal(net.stage1, n_segment_list[0])
            net.stage2 = make_block_temporal(net.stage2, n_segment_list[1])
            net.stage3 = make_block_temporal(net.stage3, n_segment_list[2])
            net.stage4 = make_block_temporal(net.stage4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            # repvgg最深的 stage3 最多才16层，最少14层，没必要隔层添加shift module了
            print('=> Using n_round {} to insert temporal shift'.format(n_round))
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    # import pdb; pdb.set_trace()
                    if i % n_round == 0:
                        if deploy:
                            blocks[i].rbr_reparam = TemporalShift(b.rbr_reparam, b.rbr_reparam.in_channels, n_segment=this_segment, n_div=n_div, soft=soft, init_mode=init_mode)
                        else:
                            if blocks[i].rbr_dense: 
                                blocks[i].rbr_dense.conv = TemporalShift(b.rbr_dense.conv, b.rbr_dense.conv.in_channels, n_segment=this_segment, n_div=n_div, soft=soft, init_mode=init_mode)
                            if blocks[i].rbr_1x1:
                                blocks[i].rbr_1x1.conv = TemporalShift(b.rbr_1x1.conv, b.rbr_1x1.conv.in_channels, n_segment=this_segment, n_div=n_div, soft=soft, init_mode=init_mode)

                return nn.Sequential(*blocks)
            
            # net.stage0 = make_block_temporal(net.stage0, n_segment_list[0]) # 加了就在低层进行时序融合
            net.stage1 = make_block_temporal(net.stage1, n_segment_list[0])
            net.stage2 = make_block_temporal(net.stage2, n_segment_list[1])
            net.stage3 = make_block_temporal(net.stage3, n_segment_list[2])
            net.stage4 = make_block_temporal(net.stage4, n_segment_list[3])

    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
   
    print('=== Temporal Soft Shift RepVGG ===')

    from archs.repvgg import repvgg_A0, repvgg_B1g2
    # imagenet pretrained, deploy MODE
    is_deploy = False
    model1 = repvgg_A0(pretrained=True, deploy=is_deploy)
    # model1 = repvgg_B1g2(pretrained=True, deploy=is_deploy)
    # model1 = getattr(torchvision.models, "resnet18")(True)
    # model4 = getattr(torchvision.models, "resnet50")(True)

    print("Origin Net:", model1)
    # make_temporal_shift(model3, n_segment=8, n_div=8, place="block", temporal_pool=False)
    make_temporal_shift(model1, n_segment=8, n_div=8, place="blockres", temporal_pool=False, deploy=is_deploy, soft=True)

    print("\n\nTSM:", model1)
    import pdb; pdb.set_trace()

    print('=> Testing CPU...')

    # test forward
    print('=> Testing forward...')

    with torch.no_grad():
        for i in range(10):
            print(i)
            x = torch.rand(2 * 8, 3, 224, 224) # (16, 3, 224, 224)
            y = model1(x) # (16, 1000)
            print(y.shape)

    # test backward
    print('=> Testing backword...')

    with torch.enable_grad():
        for i in range(10):
            print(i)
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            y1 = model1(x1)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]

    print('=> Testing GPU...')
    model1.cuda()
    # test forward
    print('=> Testing forward...')

    with torch.no_grad():
        for i in range(10):
            print(i)
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = model1(x)

    # test backward
    print('=> Testing backward...')

    with torch.enable_grad():
        for i in range(10):
            print(i)
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            y1 = model1(x1)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]

    print('Test passed.')
