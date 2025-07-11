import torch
import torch.nn as nn
from TenGAN.ggan.layers import GraphConvolution

class Discriminator3d(nn.Module):
    def __init__(self, num_nodes = 500, slices=6, time_steps = None, opt = None):
        super(Discriminator3d, self).__init__()
        self.num_nodes = num_nodes
        self.slices = slices

        if opt is None: return None
        self.opt = opt
        sig_out = self.opt['disc_sig_out']
        do_cnn_slices = self.opt['do_cnn_slices']
        self.time_steps = time_steps

        self.do_cnn_slices = do_cnn_slices
        if self.do_cnn_slices:
            # CNNs for each slice
            self.cnn_slices = nn.ModuleList([
                nn.ModuleList([
                nn.Conv2d(in_channels = 1, out_channels = self.slices, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(in_channels = self.slices, out_channels = self.slices, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(in_channels = self.slices, out_channels = 1, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                ]) for __ in range(self.slices)
            ])
        
        else:
            # Updated CNNs
            self.cnns = nn.Sequential(
                nn.Conv2d(in_channels = self.slices, out_channels = 2*self.slices, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(in_channels = 2*self.slices, out_channels = self.slices, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True)
            )

        self.final_layer = nn.Linear(self.num_nodes ** 2, 1)
        self.sig = nn.Sigmoid()
        
        if sig_out: self.sig_out = nn.Sigmoid()
        else: self.sig_out = lambda x: x

    def add_noise(self, tensor, scale = 0.1, noise_type = 'add'):

        if noise_type == 'multiply':
            if scale == 0: return tensor
            noise_tensor = (torch.rand(tuple(tensor.shape)) * 2 * scale) + (1 - scale)
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor.to(self.opt['device'])
    
        elif noise_type == 'add':
            noise_tensor = torch.rand(tuple(tensor.shape)) * tensor.max().to('cpu') * scale
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor + noise_tensor
    
    def add_noise_list(self, tensor_list, scale = 0.1, noise_type = 'add'):
        return tuple([self.add_noise(tensor, scale, noise_type) for tensor in tensor_list])

    def forward(self, ten):
        
        # _________ Updated CNN architecture _________________________________________________
        if not self.do_cnn_slices:
            stacked = self.cnns(ten)
        # ____________________________________________________________________________________

        
        # _________ CNN Slices architecture __________________________________________________
        else:
            outs = list()
            for k in range(self.slices):
                
                x = ten[:, k, :, :].unsqueeze(1)
                
                for i in range(len(self.cnn_slices[k])):
                    x = self.cnn_slices[k][i](x)
                
                # Maybe add a FC layer here?
                outs.append(x)
            stacked = torch.stack(outs, dim=1).squeeze()

            stacked = stacked.view(ten.size(0), self.slices, self.num_nodes, self.num_nodes)
        # ____________________________________________________________________________________
            
        if self.opt['disc_add_noise'] is not None:
            stacked = self.add_noise(stacked, scale = self.opt['disc_add_noise'][1], noise_type = self.opt['disc_add_noise'][0])
            
        pooled = nn.functional.max_pool3d(stacked, kernel_size=(self.slices, 1, 1))

        # pooled = self.sig(pooled)

        x = pooled.view(pooled.size(0), -1)
        
        # x = self.sig(x)
        
        x = self.final_layer(x)
        
        return self.sig_out(x), x
    
    
class Discriminator4d(nn.Module):
    def __init__(self, num_nodes = 500, slices=6, time_steps=9, opt = None):
        super(Discriminator4d, self).__init__()
        self.num_nodes = num_nodes
        self.slices = slices
        self.time_steps = time_steps

        if opt is None: return None
        self.opt = opt
        sig_out = self.opt['disc_sig_out']
        do_cnn_slices = self.opt['do_cnn_slices']

        self.do_cnn_slices = do_cnn_slices
        if self.do_cnn_slices:
            # CNNs for each slice
            self.cnn_slices = nn.ModuleList([
                nn.ModuleList([
                nn.Conv2d(in_channels = 1, out_channels = self.slices, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(in_channels = self.slices, out_channels = self.slices, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(in_channels = self.slices, out_channels = 1, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                ]) for __ in range(self.slices)
            ])
        
        else:
            # Updated CNNs
            self.cnns = nn.Sequential(
                nn.Conv2d(in_channels = self.slices*self.time_steps, out_channels = 2*self.slices, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(in_channels = 2*self.slices, out_channels = self.slices, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True)
            )

        self.final_layer = nn.Linear(self.num_nodes ** 2, 1)
        self.sig = nn.Sigmoid()
        
        if sig_out: self.sig_out = nn.Sigmoid()
        else: self.sig_out = lambda x: x

    def add_noise(self, tensor, scale = 0.1, noise_type = 'add'):

        if noise_type == 'multiply':
            if scale == 0: return tensor
            noise_tensor = (torch.rand(tuple(tensor.shape)) * 2 * scale) + (1 - scale)
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor.to(self.opt['device'])
    
        elif noise_type == 'add':
            noise_tensor = torch.rand(tuple(tensor.shape)) * tensor.max().to('cpu') * scale
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor + noise_tensor
    
    def add_noise_list(self, tensor_list, scale = 0.1, noise_type = 'add'):
        return tuple([self.add_noise(tensor, scale, noise_type) for tensor in tensor_list])

    def forward(self, ten):
        
        # _________ Updated CNN architecture _________________________________________________
        if not self.do_cnn_slices:
            ten = ten.reshape(-1, self.slices * self.time_steps, self.num_nodes, self.num_nodes)
            stacked = self.cnns(ten)
        # ____________________________________________________________________________________

        
        # _________ CNN Slices architecture __________________________________________________
        else:
            outs = list()
            for k in range(self.slices):

                x = ten[:, k, :, :].unsqueeze(1)
                
                for i in range(len(self.cnn_slices[k])):
                    x = self.cnn_slices[k][i](x)
                
                # Maybe add a FC layer here?
                outs.append(x)
            stacked = torch.stack(outs, dim=1).squeeze()

            stacked = stacked.view(ten.size(0), self.slices, self.num_nodes, self.num_nodes)
        # ____________________________________________________________________________________
            
        if self.opt['disc_add_noise'] is not None:
            stacked = self.add_noise(stacked, scale = self.opt['disc_add_noise'][1], noise_type = self.opt['disc_add_noise'][0])
            
        pooled = nn.functional.max_pool3d(stacked, kernel_size=(self.slices, 1, 1))

        # pooled = self.sig(pooled)

        x = pooled.view(pooled.size(0), -1)
        
        # x = self.sig(x)
        
        x = self.final_layer(x)
        
        return self.sig_out(x), x