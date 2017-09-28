
###################################################
# Nicolo Savioli, 2017 -- Conv-GRU pytorch v 1.0  #
###################################################
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    
    def __init__(self,input_size,hidden_size,kernel_size,cuda_flag):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,2 * self.hidden_size,3,padding=self.kernel_size//2)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,self.hidden_size,3,padding=self.kernel_size//2) 
        dtype            = torch.FloatTensor
    
    def forward(self,input,hidden):
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           if self.cuda_flag  == True:
              hidden    = Variable(torch.zeros(size_h)).cuda() 
           else:
              hidden    = Variable(torch.zeros(size_h))
        c1           = self.ConvGates(torch.cat((input,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = f.sigmoid(rt)
        update_gate  = f.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = f.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h
 
def test(num_seqs,channels_img,\
         size_image,max_epoch,model,cuda_test): 
    input_image   = torch.rand(num_seqs,1,channels_img,size_image,size_image)
    target_image  = torch.rand(num_seqs,1,channels_img,size_image,size_image)
    print('\n\n ==> Create Autograd Variables:')
    input_gru     = Variable(input_image)
    target_gru    = Variable(target_image)
    if cuda_test == True:
       input_gru  = input_gru.cuda() 
       target_gru = target_gru.cuda() 
    print('\n\n ==> Create a MSE criterion:')
    MSE_criterion = nn.MSELoss()
    if cuda_test == True:
        print("==> test on the GPU active")
        MSE_criterion = MSE_criterion.cuda()
    err           = 0
    for e in xrange(max_epoch):
        for time in xrange(num_seqs):
            h_next = model(input_gru[time], None)
            err   += MSE_criterion(h_next [0], target_gru[time])
            print(err.data[0])
    
def main():
    num_seqs     = 10
    hidden_size  = 3
    channels_img = 3 
    size_image   = 256 
    max_epoch    = 10
    cuda_flag    = False
    kernel_size  = 3
    print('Init Conv GRUs model:')
    model = ConvGRUCell(channels_img,hidden_size,kernel_size,cuda_flag)
    if cuda_flag == True:
       model = model.cuda() 
    print(repr(model))
    test(num_seqs,channels_img,size_image,\
         max_epoch,model,cuda_flag)

if __name__ == '__main__':
    main()







