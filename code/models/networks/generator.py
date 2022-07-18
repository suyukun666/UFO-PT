import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, equal_lr
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import Attention
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d

class TpsGridGen(BaseNetwork):
    def __init__(self, out_h=256, out_w=256, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()

            
    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3))
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        Z = torch.FloatTensor(3,3).fill_(0)       
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)



class FeatureRegression(BaseNetwork):
    def __init__(self, input_nc=256, output_dim=18):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 4 * 4, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        ic = 0 + (3 if 'warp' in self.opt.CBN_intype else 0) + (self.opt.semantic_nc if 'mask' in self.opt.CBN_intype else 0)
        self.fc = nn.Conv2d(ic, 16 * nf, 3, padding=1)
        if opt.eqlr_sn:
            self.fc = equal_lr(self.fc)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        if opt.use_attention:
            self.attn = Attention(4 * nf, 'spectral' in opt.norm_G)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

        self.conv_t = nn.Conv2d(3, 16 * nf, 3, padding=1)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, warp_out=None, grid=None, ref_img=None):
        seg = input if warp_out is None else warp_out

        ref_T = F.grid_sample(ref_img, grid, padding_mode='border')
        ref_T = F.interpolate(ref_T, size=(self.sh, self.sw))
        ref_T = self.conv_t(ref_T)

        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        # x = self.fc(x)
        x = self.fc(x) + ref_T

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)

        x = self.up(x)
        if self.opt.use_attention:
            x = self.attn(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x

class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        # TODO: kernel=4, concat noise, or change architecture to vgg feature pyramid
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(opt.spade_ic, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
        if opt.warp_stride == 2:
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=1, padding=pw))
        else:
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt
        
        nf = opt.ngf

        self.head_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        if opt.adaptor_nonlocal:
            self.attn = Attention(8 * nf, False)
        self.G_middle_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        self.G_middle_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt, use_se=opt.adaptor_se)

        if opt.adaptor_res_deeper:
            self.deeper0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
            if opt.dilation_conv:
                self.deeper1 = SPADEResnetBlock(4 * nf, 4 * nf, opt, dilation=2)
                self.deeper2 = SPADEResnetBlock(4 * nf, 4 * nf, opt, dilation=4)
                self.degridding0 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=2, dilation=2))
                self.degridding1 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1))
            else:
                self.deeper1 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
                self.deeper2 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

    def forward(self, input, seg):
        x = self.layer1(input)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))

        # print('x.shape after layer5',x.shape)[1, 512, 64, 64]
        
        x = self.head_0(x, seg)
        if self.opt.adaptor_nonlocal:
            x = self.attn(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        
        # print('x.shape after G_middle_1',x.shape)[1, 256, 64, 64]

        if self.opt.adaptor_res_deeper:
            x = self.deeper0(x, seg)
            x = self.deeper1(x, seg)
            x = self.deeper2(x, seg)
            if self.opt.dilation_conv:
                x = self.degridding0(x)
                x = self.degridding1(x)

        # print('x.shape before return',x.shape)[1, 256, 64, 64]
        # sys.exit(0)
        return x

class ReverseGenerator(BaseNetwork):
    def __init__(self, opt, ic, oc, size):
        super().__init__()
        self.opt = opt
        self.downsample = True if size == 256 else False
        nf = opt.ngf
        opt.spade_ic = ic
        if opt.warp_reverseG_s:
            self.backbone_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
        else:
            self.backbone_0 = SPADEResnetBlock(4 * nf, 8 * nf, opt)
            self.backbone_1 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
            self.backbone_2 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
            self.backbone_3 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.backbone_4 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.backbone_5 = SPADEResnetBlock(2 * nf, nf, opt)
        del opt.spade_ic
        if self.downsample:
            kw = 3
            pw = int(np.ceil((kw - 1.0) / 2))
            ndf = opt.ngf
            norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
            self.layer1 = norm_layer(nn.Conv2d(ic, ndf, kw, stride=1, padding=pw))
            self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, 4, stride=2, padding=pw))
            self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 4, stride=2, padding=pw))
            self.up = nn.Upsample(scale_factor=2)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.conv_img = nn.Conv2d(nf, oc, 3, padding=1)

    def forward(self, x):
        input = x
        if self.downsample:
            x = self.layer1(input)
            x = self.layer2(self.actvn(x))
            x = self.layer3(self.actvn(x))
            x = self.layer4(self.actvn(x))
        x = self.backbone_0(x, input)
        if not self.opt.warp_reverseG_s:
            x = self.backbone_1(x, input)
            x = self.backbone_2(x, input)
            x = self.backbone_3(x, input)
        if self.downsample:
            x = self.up(x)
        x = self.backbone_4(x, input)
        if self.downsample:
            x = self.up(x)
        x = self.backbone_5(x, input)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

class DomainClassifier(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        nf = opt.ngf
        kw = 4 if opt.domain_rela else 3
        pw = int((kw - 1.0) / 2)
        self.feature = nn.Sequential(nn.Conv2d(4 * nf, 2 * nf, kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(2 * nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(2 * nf, nf, kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(nf, int(nf // 2), kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(int(nf // 2), affine=True),
                                nn.LeakyReLU(0.2, False))  #32*8*8
        model = [nn.Linear(int(nf // 2) * 8 * 8, 100),
                SynchronizedBatchNorm1d(100, affine=True),
                nn.ReLU()]
        if opt.domain_rela:
            model += [nn.Linear(100, 1)]
        else:
            model += [nn.Linear(100, 2),
                      nn.LogSoftmax(dim=1)]
        self.classifier = nn.Sequential(*model)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = self.mu
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
