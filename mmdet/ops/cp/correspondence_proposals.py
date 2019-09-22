import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


def normal_init(m):
    kaiming_init(m, mode='fan_in')
    m.inited = True
class Cos_correlation(nn.Module):
    """Some Information about MyModule
    given a [B ,C,THW] calculate the THWs correlations
    output [B, THW, THW] correlation tensor
    """
    def __init__(self):
        super(Cos_correlation, self).__init__()

    def forward(self, x):
        B,C,THW=x.size()
        up=torch.matmul(x.permute(0,2,1),x) # [B,THW,C] [B,C,THW]
        assert up.size()==(B,THW,THW)
        x=x**2              #[B,C,THW]
        x=x.sum(1)           #[B,THW]
        x=x**0.5
        assert x.size()==(B,THW)
        x=torch.matmul(x.unsqueeze(2),x.unsqueeze(1)) # [B,THW,1] [B,1,THW]
        x=up/x
        return x

class Correspondence(nn.Module):
    """Some Information about Correspondence"""
    def __init__(self,inplance,norm,topk=5,mode='max'):
        super(Correspondence, self).__init__()
        assert mode in ['max','dense']
        #print('in mode ',mode)
        self.topk=topk
        self.cos=Cos_correlation()
        self.temporal_relu=nn.ReLU(inplace=True)
        self.temporal_norm=norm
        self.global_conv=nn.Conv2d(inplance,inplance,1)
        if mode =='max':
            self.with_max=True
        else:
            self.with_max=False
            self.inplance=inplance
            self.dense=torch.nn.Linear(self.topk*inplance,inplance)
        self.reset_parameters()

    def reset_parameters(self):
        last_zero_init(self.global_conv)
    def slicecollect(self,indice,x,B,S,C,H,W):
        output=[]
        for b in range(B):
            each_batch=[]
            for i in range(self.topk):
                indx=indice[b,i,:]
                each_slice=torch.index_select(x[b,...],1,indx)
                assert each_slice.size()==(C,S*H*W)
                each_batch.append(each_slice)
            each_batch=torch.stack(each_batch)
            assert(each_batch.size()==(self.topk,C,S*H*W))
            output.append(each_batch)
        return torch.stack(output)
    def forward(self, x , snip):
        #input [B_S,C,H,W]
        snip_size=snip
        B_S,C,H,W=x.size()
        B=B_S//snip_size
        identity =x
        x=x.view(-1,snip_size,C,H,W)
        x=x.permute(0,2,1,3,4).contiguous()
        x=x.view(-1,C,snip_size*H*W)

        correlation=self.cos(x) #[B,THW,THW]
        for i in range(snip_size):
            correlation[:,H*W*i:H*W*(i+1):,H*W*i:H*W*(i+1)]=-1
        _,indice = torch.topk(correlation,self.topk,dim=1)
        del correlation
        assert indice.size()==(B,self.topk,snip_size*H*W)

        x=self.slicecollect(indice,x,B_S//snip_size,snip_size,C,H,W)
        assert x.size()==(B,self.topk,C,snip_size*H*W)
        if self.with_max:
            x=x.max(1)[0]
            x=x.view(B_S//snip_size,C,snip_size,H,W)
            x=x.permute(0,2,1,3,4).contiguous()
        else:
            x=x.permute(0,3,1,2).contiguous() #[B,THW,k,C]
            x=x.view(B_S*H*W,self.topk,C)
            x=x.view(B_S*H*W,self.topk*C)

            x=self.dense(x) #[bTHW,C]
            x=x.view(B,snip_size*H*W,C)
            x=x.view(B,snip_size,H*W,C)
            x=x.view(B_S,H*W,C)
            x=x.view(B_S,H*W,C)
            x=x.permute(0,2,1).contiguous()


        x=x.view(B_S,C,H,W)
        x = self.temporal_norm(x)
        x = self.temporal_relu(x)
        x = self.global_conv(x) #[B,C,H,W]
        x+=identity
        return x





# if __name__ == "__main__":
#     # inputs=torch.randn(8,32,120)
#     # inputs.requires_grad_()

#     # cos=Cos_correlation()
#     # cos.train()
#     # cos.zero_grad()
#     # out=cos(inputs)
#     # print(out.shape)
#     # out = sum(out.view(-1))
#     # out.backward()
#     # print(inputs.grad.shape)
#     # print('complete')
#     cp=Correspondence(7,5,29,mode='dense')
#     inputs=torch.randn(21,29,13,11)
#     inputs.requires_grad_()

#     cp.train()
#     cp.zero_grad()
#     out=cp(inputs)
#     print(out.shape)
#     out = sum(out.view(-1))
#     out.backward()
#     print(inputs.grad)
#     print('complete')