from inclusion import *

class cSE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(cSE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg-pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class sSE_Block(nn.Module):
    def __init__(self, channel):
        super(sSE_Block, self).__init__()
        self.conv_act = nn.Sequential(
                nn.Conv2d(channel, 1, 1),
                nn.Sigmoid())

    def forward(self, x): 
        y = self.conv_act(x)
        return x * y 

class scSE_Block(nn.Module):
    """
    Implementation of Concurrent Spatial and Channel Squeeze & Excitation as discussed
    by Roy et al.
    """

    def __init__(self, channel):
        super(scSE_Block, self).__init__()
        self.cse = cSE_Block(channel)
        self.sse = sSE_Block(channel)

    def forward(self, x):
        return self.cse(x) + self.sse(x)
    
class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out, use_scse=False):
        super().__init__()
        self.use_scse = use_scse
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.scse = scSE_Block(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        if self.use_scse:
            return self.scse(self.bn(F.relu(cat_p)))
        return self.bn(F.relu(cat_p))

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
    
class Unet34(nn.Module):
    def __init__(self, rn, use_hypercols=False):
        super().__init__()
        self.rn = rn
        self.use_hypercols = use_hypercols
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.relu(self.rn(x))
        u1 = self.up1(x, self.sfs[3].features)
        u2 = self.up2(u1, self.sfs[2].features)
        u3 = self.up3(u2, self.sfs[1].features)
        u4 = self.up4(u3, self.sfs[0].features)
        if self.use_hypercols: # Adding hypercolumns as discussed by Hariharan et al.
            f = torch.cat((F.interpolate(u1, scale_factor=8, mode='bilinear', align_corners=False),
                F.interpolate(u2, scale_factor=4, mode='bilinear', align_corners=False),
                F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False),
                F.interpolate(u4, scale_factor=1, mode='bilinear', align_corners=False)), dim=1)
            f = F.dropout2d(f, p=0.5)
            act = F.relu(f)
        else:
            act = u4
        u5 = self.up5(act)
        return u5[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()
            
class UnetModel():
    def __init__(self,model,name='Unet'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]

