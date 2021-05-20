import torch.nn as nn
from layers import iconv_bn_xd, iconv_bn_xd_seq, IResidualXD, ShapePool2DXD

def compute_pooling(pools):
    x = sum(map(lambda x:x=="depth", pools))
    return int(4**x)

def compute_features(infeats, pools):
    features = [infeats]
    for p in pools:
        if p=="channel":
            features+=[features[-1]*4]
        else:
            features+=[features[-1]]
    return features

class IResXD(nn.Module):
    def __init__(self, w=0.125, infeatures=32, inchannels=3,
                 pools  = ["depth", "depth", "depth"],
                 nblock=2, nlayer=[1,1,1], nclass=10,
                 invert=True, skip_invert=True, 
                 negative_slope=0.01, bn_ieps=0.1):
        """
        """
        super().__init__()
        features = compute_features(infeatures, pools)
        prep  = iconv_bn_xd(inchannels, features[0], invert=invert, conv_invert=False, 
                            negative_slope=negative_slope, bn_ieps=bn_ieps)
        layer = IResidualXD(features[0], nblock=1, invert=invert, skip_invert=skip_invert, 
                            bn_ieps=bn_ieps, negative_slope=negative_slope)
        layers= [prep, layer]
        for i,pool in enumerate(pools):
            layer = [ShapePool2DXD(2, pool)]
            layer += [IResidualXD(features[i+1], nblock=nblock, invert=invert, 
                                 skip_invert=skip_invert, bn_ieps=bn_ieps, 
                                 negative_slope=negative_slope) for j in range(nlayer[i])]
            layers.extend(layer)
        self.layers = nn.Sequential(*layers)
        self.pool   = nn.MaxPool3d((compute_pooling(pools), 2**(5-len(pools)), 2**(5-len(pools))))
        self.linear = nn.Linear(features[-1], nclass, bias=False)
        self.w = w

    def forward(self, x):
        x = x[:,:,None,:,:]
        x = self.layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.linear(x)
        return x*self.w
    
class IFlatXD(nn.Module):
    def __init__(self, infeatures=32, invert=False, 
                 pools  = ["depth", "depth", "depth"],
                 layers = [2, 3, 4, 3], w=0.125,
                 negative_slope=0.01, bn_ieps=0.1,
                 inchannels=1):
        """
        """
        super().__init__()
        features = compute_features(infeatures, pools)
        
        prep   = iconv_bn_xd(inchannels, features[0], invert=invert, conv_invert=False, negative_slope=negative_slope, bn_ieps=bn_ieps)
        layer  = iconv_bn_xd_seq(features[0], layers[0]-1, invert=invert, negative_slope=negative_slope, bn_ieps=bn_ieps)

        modules = [prep, layer]
        for i in range(1, len(layers)):
            pool  = ShapePool2DXD(2, pools[i-1])
            layer = iconv_bn_xd_seq(features[i], layers[i], invert=invert, negative_slope=negative_slope, bn_ieps=bn_ieps)
            modules.extend([pool, layer])
            
        self.features = nn.Sequential(*modules)
        self.pool   = nn.MaxPool3d((compute_pooling(pools), 2**(5-len(pools)), 2**(5-len(pools))))
        self.linear = nn.Linear(features[-1], 10, bias=False)
        self.w = w
        
    def forward(self, x):
        """
        """
        x = x[:,:,None,:,:]
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.linear(x)
        return x*self.w