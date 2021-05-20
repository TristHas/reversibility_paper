#from revunet.Imodules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from rev_block import RevBlock

class IConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, invert=True, **kwargs):
        super().__init__()
        self.set_invert(invert)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.kwargs = kwargs

        if self.invert:
            if self.in_channels == self.out_channels:
                assert (self.in_channels % 2) == 0
                f_conv =  nn.Conv2d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
                g_conv =  nn.Conv2d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
                self.module = RevBlock(f_conv, g_conv, invert=True)
            else:
                raise Exception(f'Cannot inverse convolution with in_channels {self.in_channels} and out_channels {self.out_channels}')
        else:
            self.module = nn.Conv2d(self.in_channels, self.out_channels, *self.args, **self.kwargs)

    def set_invert(self, invert):
    	self.invert = invert

    def forward(self, x):
        return self.module(x)

class IBatchNorm2d(nn.BatchNorm2d):
    """
    """
    def __init__(self, *args, ieps=0, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ieps = ieps
        self.set_invert(invert)
        
    def set_invert(self, invert):
        self.invert = invert
        
    def forward(self, x):
        if self.invert:
            return self.i_forward(x)
        else:
            return super().forward(x)

    def i_forward(self, x):
        with torch.no_grad():
            x_ = x.permute(1,0,2,3).contiguous().view(x.size(1), -1)
            mean, std = x_.mean(1).squeeze(), x_.std(1).squeeze()

        out = F.batch_norm( 
            x, None, None, self.weight.abs() + self.ieps, self.bias, 
            True, 0.0, self.eps
        )

        if self.training and out.requires_grad:
            handle_ref = [0]
            handle_ref_ = out.register_hook(self.get_variable_backward_hook(x, out, std, mean, handle_ref))
            handle_ref[0] = handle_ref_
        x.data.set_()
        return out
        
    def inverse(self, y, x, std, mean):
        with torch.no_grad():
            x_ =  F.batch_norm(
                        y, None, None, std, mean, 
                        True, 0.0, 0
                    )
        x.data.set_(x_)
        y.data.set_()

    def get_variable_backward_hook(self, x, output, std, mean, handle_ref):
        def backward_hook(grad):
            self.inverse(output, x, std, mean)
            handle_ref[0].remove()
        return backward_hook

class IBatchNorm3d(nn.BatchNorm3d):
    """
    """
    def __init__(self, *args, ieps=0, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ieps = ieps
        self.set_invert(invert)
        
    def set_invert(self, invert):
        self.invert = invert
        
    def forward(self, x):
        if self.invert:
            return self.i_forward(x)
        else:
            return super().forward(x)

    def i_forward(self, x):
        with torch.no_grad():
            x_ = x.permute(1, 0, 2, 3, 4).contiguous().view(x.size(1), -1)
            mean, std = x_.mean(1).squeeze(), x_.std(1).squeeze()
        out = F.batch_norm( 
            x, None, None, self.weight.abs() + self.ieps, self.bias, 
            True, 0.0, self.eps
        )

        if self.training and out.requires_grad:
            handle_ref = [0]
            handle_ref_ = out.register_hook(self.get_variable_backward_hook(x, out, std, mean, handle_ref))
            handle_ref[0] = handle_ref_
        x.data.set_()
        return out
        
    def inverse(self, y, x, std, mean):
        with torch.no_grad():
            x_ =  F.batch_norm(
                        y, None, None, std, mean, 
                        True, 0.0, 0
                    )
        x.data.set_(x_)
        y.data.set_()

    def get_variable_backward_hook(self, x, output, std, mean, handle_ref):
        def backward_hook(grad):
            self.inverse(output, x, std, mean)
            handle_ref[0].remove()
        return backward_hook    

class ILeakyReLU(nn.LeakyReLU):
    def __init__(self, *args, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_invert(invert)
        
    def set_invert(self, invert):
        self.invert = invert
                
    def forward(self, x):
        if self.invert:
            return self.i_forward(x)
        else:
            return super().forward(x)

    def i_forward(self, x):
        y=super().forward(x)
        if self.training and y.requires_grad:
            handle_ref = [0]
            handle_ref_ = y.register_hook(self.get_variable_backward_hook(x, y, handle_ref))
            handle_ref[0] = handle_ref_
        x.data.set_()
        return y
        
    def inverse(self, x, y):
        with torch.no_grad():
            x_ = F.leaky_relu(y, 1/self.negative_slope, self.inplace)
        x.data.set_(x_)
        y.data.set_()

    def get_variable_backward_hook(self, x, y, handle_ref):
        def backward_hook(grad):
            self.inverse(x, y)
            handle_ref[0].remove()
        return backward_hook

class IResidual(nn.Module):
    def __init__(self, c, nblock=2, bn_ieps=0.1, negative_slope=0.01, 
                 invert=False, skip_invert=False, downsample=None):
        super().__init__()
        def residual_module_fn(c):
            layers = [iconv_bn(c, c, invert=invert, bn_ieps=bn_ieps, 
                               negative_slope=negative_slope) 
                      for i in range(nblock)]
            mod = nn.Sequential(*layers)
            setattr(mod, 'invert', invert)
            return mod
        self.skip = ISkip(c, residual_module_fn, skip_invert=skip_invert, invert=invert)

    def forward(self, x):
        return self.skip(x)
    
    
class ISkip(nn.Module):
    def __init__(self, channels, skip_module_fn, skip_invert=True, invert=True):
        super().__init__()
        self.set_invert(skip_invert)
        self.skip_invert = skip_invert

        if skip_invert:
            self.module = RevBlock(
                skip_module_fn(channels//2),
                skip_module_fn(channels//2),
                invert = True
            )
        else:
            self.skip = RevAdd(invert)
            self.module = skip_module_fn(channels)
    
    def forward(self, x):
        if self.skip_invert:
            return self.module(x) 
        else:
            x = self.skip.register_skip(x)
            return self.skip(self.module(x))
    
    def set_invert(self, invert):
        self.invert = invert


def iconv_bn(c_in, c_out, invert=False, conv_invert=None, 
             negative_slope=0.01, bn_ieps=0.1, bn_weight_init=1.0, **kw):
    conv_invert= invert if conv_invert is None else conv_invert
    mod = nn.Sequential(
        IConv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False, invert=conv_invert), 
        ibatch_norm(c_out, bn_weight_init=bn_weight_init, invert=invert), 
        ILeakyReLU(invert=invert, negative_slope=negative_slope),
    )
    setattr(mod, 'invert', invert)
    return mod

def ibatch_norm(num_channels, invert=False, ieps=0.0, bn_bias_init=None, 
                bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = IBatchNorm2d(num_channels, ieps=ieps, invert=invert)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
    return m

class ShapePool2DXD(nn.Module):
    def __init__(self, block_size, stack_dim='depth'):
        super().__init__()
        assert stack_dim in ['depth', 'channel']
        self.stack_dim = stack_dim
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, inp):
        (batch_size, d_channel, d_depth, d_height, d_width) = inp.size()
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        
        if self.stack_dim == 'depth':
            s_depth = int(d_depth / self.block_size_sq)
            tmp1 = inp.view(batch_size, d_channel, s_depth, 2, 2, d_height, d_width)
            tmp2 = tmp1.permute(0, 1, 2, 5, 3, 6, 4).contiguous()
            out = tmp2.view(batch_size, d_channel, s_depth, s_height, s_width)
            tmp2.data.set_()
            tmp1.data.set_()
            
        elif self.stack_dim == 'channel':
            s_channel = int(d_channel / self.block_size_sq)
            tmp1 = inp.view(batch_size, 2, 2, s_channel, d_depth, d_height, d_width)
            tmp2 = tmp1.permute(0, 3, 4, 5, 1, 6, 2).contiguous()
            out = tmp2.view(batch_size, s_channel, d_depth, s_height, s_width)
            tmp2.data.set_()
            tmp1.data.set_()
        else:
            raise Exception(f"Invalid stackdim {self.stack_dim}")
        return out

    def forward(self, inp):
        (batch_size, s_channel, s_depth, s_height, s_width) = inp.size()
        d_height = int(s_height / self.block_size)
        d_width = int(s_height / self.block_size)
        
        if self.stack_dim == 'depth':
            d_depth = s_depth * self.block_size_sq
            tmp1 = inp.view(batch_size, s_channel, s_depth, d_height, 2, d_width, 2)
            tmp2 = tmp1.permute(0, 1, 2, 4, 6, 3, 5).contiguous()
            out  = tmp2.view(batch_size, s_channel, d_depth, d_height, d_width)
            tmp2.data.set_()

        elif self.stack_dim == 'channel':
            d_channel = s_channel * self.block_size_sq
            tmp1 = inp.view(batch_size, s_channel, s_depth, d_height, 2, d_width, 2)
            tmp2 = tmp1.permute(0, 4, 6, 1, 2, 3, 5).contiguous()
            out = tmp2.view(batch_size, d_channel, s_depth, d_height, d_width)
            tmp2.data.set_()
        else:
            raise Exception(f"Invalid stackdim {self.stack_dim}")

        if self.training and out.requires_grad:
            handle_ref = [0]
            handle_ref_ = out.register_hook(self.get_variable_backward_hook(inp, out, handle_ref))
            handle_ref[0] = handle_ref_

        inp.data.set_()

        return out
    
    def get_variable_backward_hook(self, x, out, handle_ref):
        def backward_hook(grad):
            x.data.set_(self.inverse(out))
            handle_ref[0].remove()
        return backward_hook

class IConv2dXD(nn.Module):
    def __init__(self, in_channels, out_channels, *args, kernel_size=3, invert=True, **kwargs):
        super().__init__()
        self.set_invert(invert)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.kwargs = kwargs
        self.kwargs['padding']=(0, 1, 1)
        kernel_size = (1, kernel_size, kernel_size)

        if self.invert:
            if self.in_channels == self.out_channels:
                assert (self.in_channels % 2) == 0
                f_conv =  nn.Conv3d(self.in_channels//2, self.out_channels//2, 
                                    *self.args, kernel_size=kernel_size, **self.kwargs)
                g_conv =  nn.Conv3d(self.in_channels//2, self.out_channels//2, 
                                    *self.args, kernel_size=kernel_size, **self.kwargs)
                self.module = RevBlock(f_conv, g_conv, invert=True)
            else:
                raise Exception(f'Cannot inverse convolution with in_channels {self.in_channels} and out_channels {self.out_channels}')
        else:
            self.module = nn.Conv3d(self.in_channels, self.out_channels, *self.args, kernel_size=kernel_size, **self.kwargs)

    def set_invert(self, invert):
    	self.invert = invert

    def forward(self, x):
        return self.module(x)
    
class IResidualXD(nn.Module):
    def __init__(self, c, bn_ieps=0.1, negative_slope=0.01, 
                 invert=False, skip_invert=False, downsample=None):
        super().__init__()
        def residual_module_fn(c):
            mod = nn.Sequential(
                iconv_bn_xd(c, c, invert=invert, bn_ieps=bn_ieps, negative_slope=negative_slope), 
                iconv_bn_xd(c, c, invert=invert, bn_ieps=bn_ieps, negative_slope=negative_slope), 
            )
            setattr(mod, 'invert', invert)
            return mod

        self.skip = ISkip(c, residual_module_fn, skip_invert=skip_invert, invert=invert)

    def forward(self, x):
        return self.skip(x)
                
class IResidualXD(nn.Module):
    def __init__(self, c, nblock=2, bn_ieps=0.1, negative_slope=0.01, 
                 invert=False, skip_invert=False, downsample=None):
        super().__init__()
        def residual_module_fn(c):
            layers = [iconv_bn_xd(c, c, invert=invert, bn_ieps=bn_ieps, 
                                  negative_slope=negative_slope) 
                      for i in range(nblock)]
            mod = nn.Sequential(*layers)
            setattr(mod, 'invert', invert)
            return mod
        self.skip = ISkip(c, residual_module_fn, skip_invert=skip_invert, invert=invert)

    def forward(self, x):
        return self.skip(x)
        
def iconv_bn_xd(c_in, c_out, invert=False, conv_invert=None, negative_slope=0.01, 
                bn_ieps=0.1, bn_weight_init=1.0, **kw):
    conv_invert = invert if conv_invert is None else conv_invert
    mod = nn.Sequential(
        IConv2dXD(c_in, c_out, kernel_size=3, stride=1, 
                  padding=1, bias=False, invert=conv_invert), 
        ibatch_norm_xd(c_out, bn_weight_init=bn_weight_init, invert=invert), 
        ILeakyReLU(invert=invert, negative_slope=negative_slope)
    )
    setattr(mod, 'invert', invert)
    return mod

def iconv_bn_xd_seq(ch, nblk, invert=True, negative_slope=0.01, 
                    bn_ieps=0.1, bn_weight_init=1.0, **kw):
    mod = nn.Sequential(
        *[iconv_bn_xd(ch, ch, invert=invert, negative_slope=negative_slope, 
                      bn_ieps=bn_ieps, bn_weight_init=bn_weight_init) 
          for i in range(nblk)]
    )
    setattr(mod, 'invert', invert)
    return mod

def ibatch_norm_xd(num_channels, invert=False, ieps=0.0, bn_bias_init=None, 
                   bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = IBatchNorm3d(num_channels, ieps=ieps, invert=invert)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
    return m
