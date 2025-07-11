import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from opt_einsum import contract
import tensorly as tl
from tensorly.tucker_tensor import tucker_to_tensor
import sys


class Kernel(nn.Module):
    ''' Implement a kernel smoothing regularization'''
    
    def __init__(self, window, density, sigma = 0.5, device = 'cpu'):
        super().__init__()
        self.sigma = sigma
        self.window = window
        self.density = density
        self.weight = self.gaussian().to(device)
        
    def gaussian(self):
        ''' Make a Gaussian kernel'''

        window = int(self.window-1)/2
        sigma2 = self.sigma * self.sigma
        x = torch.FloatTensor(np.arange(-window, window+1))
        phi_x = torch.exp(-0.5 * abs(x) / sigma2)
        phi_x = phi_x / phi_x.sum()
        return phi_x.view(1, 1, self.window, 1).to(torch.double)

    
    def forward(self,factor):
        ''' Perform a Gaussian kernel smoothing on a temporal factor'''

        factor = factor.double()
        row, col = factor.shape
        conv = F.conv2d(factor.view(1, 1, row, col), self.weight, 
                          padding = (int((self.window-1)/2), 0))
        return conv.view(row, col)


# Updated Tensor Generator
# ___________________________________________________________________________________________________________________________________________________


class TensorGenerator(nn.Module):
    def __init__(self, latent_dim=100, layer_size=128, num_nodes=500, rank=30, num_views=25, decomp_type = 'CPD',
                 num_time_steps = 9, num_tensor_modes = None, output_factors = False, tensor_decomposition = True, opt = None):

        super(TensorGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.latent_dim = latent_dim
        self.num_views = num_views
        self.decomp_type = decomp_type
        self.num_tensor_modes = num_tensor_modes
        self.num_time_steps = num_time_steps
        self.tensor_decomposition = tensor_decomposition

        if opt is None: return None
        self.opt = opt

        self.device = self.opt['device']
        self.smooth = Kernel(self.opt['gen_channel_smooth_window'], density = None, device = self.device).to(self.device)

        if 'gen_smooth_modes' not in self.opt: self.opt['gen_smooth_modes'] = list()
        self.smooth_modes = self.opt['gen_smooth_modes']

        shared_layers = [

            nn.Linear(latent_dim, layer_size),
            nn.ReLU(inplace=True),

            # New block
            nn.Linear(layer_size, layer_size * 2),
            nn.BatchNorm1d(layer_size * 2),
            nn.ReLU(inplace=True),
            
            # New block
            nn.Linear(layer_size * 2, layer_size * 4),
            nn.BatchNorm1d(layer_size * 4),
            nn.ReLU(inplace=True)

        ]
        
        self.shared = nn.Sequential(*shared_layers)
        self.output_factors = output_factors      
        
        if not self.tensor_decomposition:

            output_intermediate_tensor = [
                
                nn.Linear(layer_size * 4, layer_size * 4),  
                nn.BatchNorm1d(layer_size * 4),
                nn.ReLU(),   

            ]

            output_tensor = [  

                nn.Linear(layer_size * 4, layer_size * 8),
                nn.ReLU(),

                nn.Linear(layer_size * 8, num_views * num_nodes),
                nn.ReLU(),
                
                nn.Linear(num_views * num_nodes, num_views * num_nodes * 4),
                nn.ReLU(),

                nn.Linear(num_views * num_nodes * 4, num_nodes * num_nodes * num_views),
                nn.Sigmoid()

            ]

            self.output_intermediate_tensor = nn.Sequential(*output_intermediate_tensor)
            self.output_tensor = nn.Sequential(*output_tensor)

        else:

            output_layers = [
                [
                    nn.Linear(layer_size * 4, layer_size * 2),
                    # nn.BatchNorm1d(layer_size * 8),
                    nn.ReLU(),

                    nn.Linear(layer_size * 2, num_nodes * rank),
                    nn.Sigmoid(),

                ] for _ in range(2)
            ]
            
            view_layer = [

                nn.Linear(layer_size * 4, layer_size * 2),
                # nn.BatchNorm1d(layer_size * 4),
                nn.ReLU(),

                nn.Linear(layer_size * 2, num_views * rank),
                nn.Sigmoid(),

            ]

            if num_tensor_modes == 4:

                time_layer = [
                    nn.Linear(layer_size * 4, layer_size * 2),
                    # nn.BatchNorm1d(layer_size * 4), 
                    nn.ReLU(),

                    nn.Linear(layer_size * 2, num_time_steps * rank),
                    nn.Sigmoid(),
                ]


            self.output1 = nn.Sequential(*output_layers[0])
            self.output2 = nn.Sequential(*output_layers[1])
            self.output3 = nn.Sequential(*view_layer)  
            if num_tensor_modes == 4: self.output4 = nn.Sequential(*time_layer)
        
            if self.decomp_type == 'tucker':
                core_tensor_layer = [
                    nn.Linear(layer_size * 4, layer_size * 4),
                    nn.ReLU(),
                    nn.Linear(layer_size * 4, rank**num_tensor_modes),
                    nn.Sigmoid()
                ]
                
                self.output_core = nn.Sequential(*core_tensor_layer)
    
    def set_factor_output(self, new_val):
        self.output_factors = new_val
        return True
    
    def add_noise(self, tensor, scale = 0.1, noise_type = 'add'):
        
        if noise_type == 'multiply':
            noise_tensor = (torch.rand(tuple(tensor.shape)) * (scale[1]-scale[0])) + scale[0]
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor
    
        elif noise_type == 'add':
            if scale == 0: return tensor
            noise_tensor = torch.rand(tuple(tensor.shape)) * tensor.max().to('cpu') * scale
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor + noise_tensor 
        
        elif noise_type == 'zeros':
            noise_tensor = (torch.rand(tuple(tensor.shape)) * (scale[1]-scale[0])) + scale[0]
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor.round()
        
        elif noise_type == 'add_n':
            mean, std = scale[0]*tensor.mean(), scale[1]*tensor.std()
            noise_tensor = torch.randn(tensor.shape).to(self.opt['device']) * std + mean
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor + noise_tensor
        
        elif noise_type == 'multiply_n':
            mean, std = scale
            noise_tensor = torch.randn(tensor.shape).to(self.opt['device']) * std + mean
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor
        
        
    def add_noise_list(self, tensor_list, scale = 0.1, noise_type = 'add'):
        return tuple([self.add_noise(tensor, scale, noise_type) for tensor in tensor_list])
        

    def forward(self, noise):
        batch_sz = noise.shape[0]
        S = self.shared(noise)
        
        
        if not self.tensor_decomposition:

            tensor = self.output_intermediate_tensor(S)

            tensor = self.add_noise(tensor, scale = [0, 1], noise_type = 'multiply')
            tensor = self.add_noise(tensor, scale = [0, 2], noise_type = 'multiply')
            
            # tensor = self.add_noise(tensor, scale = [0, 1], noise_type = 'zeros')
            # tensor = self.add_noise(tensor, scale = self.opt['gen_add_noise'][1], noise_type = self.opt['gen_add_noise'][0])

            tensor = self.output_tensor(tensor)

            tensor = tensor.view(batch_sz, self.num_views, self.num_nodes, self.num_nodes)

            return tensor
            
        # generate factor matrices
        A = self.output1(S).view(batch_sz, self.num_nodes, self.rank)
        B = self.output2(S).view(batch_sz, self.num_nodes, self.rank)
        C = self.output3(S).view(batch_sz, self.num_views, self.rank)
        if self.num_tensor_modes == 4:
            D = self.output4(S).view(batch_sz, self.num_time_steps, self.rank)

        # add noise
        if self.opt['gen_add_noise'] is not None:
            A, B, C = self.add_noise_list([A, B, C], scale = [0, 1], noise_type = 'zeros')
            A, B, C = self.add_noise_list([A, B, C], scale = self.opt['gen_add_noise'][1], noise_type = self.opt['gen_add_noise'][0])

            if self.num_tensor_modes == 4:
                D = self.add_noise(D, scale = [0, 1], noise_type = 'zeros')
                D = self.add_noise(D, scale = self.opt['gen_add_noise'][1], noise_type = self.opt['gen_add_noise'][0])


        factors = [A, B, C]
        if self.num_tensor_modes == 4: factors.append(D)
        self.smooth_loss = sum([sum([abs(factor - self.smooth(factor)).mean() for factor in factors[i]]) for i in self.smooth_modes])

        # Reconstruct tensor using generated factor matrices
        if self.decomp_type == 'CPD':

            if self.num_tensor_modes == 3:
                out = contract('faz,fbz,fcz->fabc', A, B, C, backend='torch')
                out = out.permute(0, 3, 1, 2)

                if self.output_factors: return (A, B, C)
                else: return out
                
            elif self.num_tensor_modes == 4:
                
                out = contract('faz,fbz,fcz,fdz->fabcd', A, B, C, D, backend='torch')
                out = out.permute(0, 3, 4, 1, 2)
                
                if self.output_factors: return (A, B, C, D)
                else: return out

        elif self.decomp_type == 'tucker':
            D = self.output_core(S).view([batch_sz] + [self.rank for m in range(self.num_tensor_modes)])
            
            if self.opt['gen_add_noise'] is not None:
                D = self.add_noise(D, self.opt['gen_add_noise'][1], noise_type = self.opt['gen_add_noise'][0])
                
            if self.output_factors: return (D, A, B, C)
            else: 
                
                tl.set_backend('pytorch')
                reconstructions = torch.stack([tucker_to_tensor((D[i], [C[i], A[i], B[i]])) 
                                               for i in range(D.shape[0])])
                tl.set_backend('numpy')
                
                return reconstructions
            
        else:
            print("Unknown tensor decomposition type for generating tensor factors!")
            return None

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim)).to(self.device)
    
    def change_device(self, device_name):
        self.device = device_name
        self.opt['device'] = device_name
        self.smooth = self.smooth.to(self.device)
        self.smooth.weight = self.smooth.weight.to(self.device)
        
    def generate(self, num_samples):
        return self(torch.randn((num_samples, self.latent_dim)).to(self.device)).to(self.device)

# ___________________________________________________________________________________________________________________________________________________


class NormalGenerator(nn.Module):
    def __init__(self, generate_tensor_shape = [25, 51, 51], 
                 opt = None):
        
        super(NormalGenerator, self).__init__()


        if opt is None: 
            print("Opt is None!")
            return None

        self.opt = opt
        
        self.smooth = None

        self.device = self.opt['device']
        self.latent_dim = self.opt['latent_dim']
        self.layer_size = self.opt['gen_layer_size']

        try: self.generate_tensor_shape = list(generate_tensor_shape)
        except: 
            print("Generate tensor shape not valid -- cannot convert to list type!")
            return None


        # _______________________
        
        input_latent_dim = [
            nn.Linear(self.latent_dim, self.layer_size * 4),
            nn.ReLU(),
        ]
        
        output_intermediate_tensor_1 = [
                
            nn.Linear(self.layer_size * 4, self.layer_size * 4),  
            nn.BatchNorm1d(self.layer_size * 4),
            nn.ReLU(),   

        ]

        output_intermediate_tensor_2 = [  

            nn.Linear(self.layer_size * 4, min(self.generate_tensor_shape)),
            nn.ReLU(),

            nn.Linear(min(self.generate_tensor_shape), 
                      min(self.generate_tensor_shape) * max(self.generate_tensor_shape)),
            
            nn.ReLU(),


            nn.Linear(min(self.generate_tensor_shape) * max(self.generate_tensor_shape), 
                      min(self.generate_tensor_shape) * max(self.generate_tensor_shape) * 4),

            nn.ReLU()

        ]


        def get_list_product(l):
            return_value = 1
            for i in l: return_value *= i
            return return_value

        output_tensor = [

            nn.Linear(min(self.generate_tensor_shape) * max(self.generate_tensor_shape) * 4, 
                      get_list_product(self.generate_tensor_shape)),

            nn.Sigmoid()

        ]
        
        self.input_latent_dim = nn.Sequential(*input_latent_dim)
        self.output_intermediate_tensor_1 = nn.Sequential(*output_intermediate_tensor_1)
        self.output_intermediate_tensor_2 = nn.Sequential(*output_intermediate_tensor_2)
        self.output_tensor = nn.Sequential(*output_tensor)



    # __________________________________________


    def forward(self, x):
        x = self.input_latent_dim(x)
        # x = self.add_noise(tensor = x, scale = [1, 1], noise_type = 'multiply_n')
        x = self.output_intermediate_tensor_1(x)
        # x = self.add_noise(tensor = x, scale = [1, 0.75], noise_type = 'multiply_n')
        x = self.add_noise(tensor = x, scale = [0.25, 1], noise_type = 'multiply_e_n')
        x = self.output_intermediate_tensor_2(x)
        # x = self.add_noise(tensor = x, scale = [1, 0.5], noise_type = 'multiply_n')
        x = self.output_tensor(x)
        
        output_shape = [x.shape[0]] + self.generate_tensor_shape
        x = x.reshape(output_shape)
        return x
    
    
    # __________________________________________
    
    def add_noise(self, tensor, scale = 0.1, noise_type = 'add'):
        
        if noise_type == 'multiply':
            noise_tensor = (torch.rand(tuple(tensor.shape)) * (scale[1]-scale[0])) + scale[0]
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor
    
        elif noise_type == 'add':
            if scale == 0: return tensor
            noise_tensor = torch.rand(tuple(tensor.shape)) * tensor.max().to('cpu') * scale
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor + noise_tensor 
        
        elif noise_type == 'zeros':
            noise_tensor = (torch.rand(tuple(tensor.shape)) * (scale[1]-scale[0])) + scale[0]
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor.round()
        
        elif noise_type == 'add_n':
            mean, std = scale[0]*tensor.mean(), scale[1]*tensor.std()
            noise_tensor = torch.randn(tensor.shape).to(self.opt['device']) * std + mean
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor + noise_tensor

        elif noise_type == 'multiply_n':
            mean, std = scale
            noise_tensor = torch.randn(tensor.shape).to(self.opt['device']) * std + mean
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor
        
        elif noise_type == 'multiply_e_n':
            mean, std = scale
            noise_tensor = torch.randn(tensor.shape) * std + mean
            noise_tensor = np.exp(noise_tensor)
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

    def change_device(self, device_name):
        self.device = device_name
        self.opt['device'] = device_name

        if self.smooth is not None: 
            self.smooth = self.smooth.to(self.device)
            self.smooth.weight = self.smooth.weight.to(self.device)

    def generate(self, num_samples):
        return self(torch.randn((num_samples, self.latent_dim)).to(self.device)).to(self.device)


# ___________________________________________________________________________________________________________________________________________________


class NormalGeneratorCNN(nn.Module):
    def __init__(self, generate_tensor_shape = [25, 51, 51], 
                 opt = None):
        
        super(NormalGeneratorCNN, self).__init__()


        if opt is None: 
            print("Opt is None!")
            return None

        self.opt = opt
        
        self.smooth = None

        self.device = self.opt['device']
        self.latent_dim = self.opt['latent_dim']
        self.layer_size = self.opt['gen_layer_size']

        try: self.generate_tensor_shape = list(generate_tensor_shape)
        except: 
            print("Generate tensor shape not valid -- cannot convert to list type!")
            return None


        # _______________________
        
        input_latent_dim = [

            nn.Linear(self.latent_dim, self.layer_size * 4),
            nn.ReLU(),

            nn.Linear(self.layer_size * 4, max(self.generate_tensor_shape)**2),
            nn.ReLU()

        ]

        convs_output_channels = self.opt['gen_hidden_channels'] * 4

        conv1 = [

            nn.Conv2d(in_channels = 1, out_channels = self.opt['gen_hidden_channels'], 
                      kernel_size = (3,3), stride = 1, padding = 1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels = self.opt['gen_hidden_channels'], out_channels = self.opt['gen_hidden_channels'] * 2, 
                      kernel_size = (3,3), stride = 1, padding = 1),
            nn.LeakyReLU()

        ]
        
        conv2 = [

            nn.Conv2d(in_channels = self.opt['gen_hidden_channels'] * 2, out_channels = self.opt['gen_hidden_channels'] * 4, 
                      kernel_size = (3,3), stride = 1, padding = 1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels = self.opt['gen_hidden_channels'] * 4, out_channels = self.opt['gen_hidden_channels'] * 2, 
                      kernel_size = (3,3), stride = 1, padding = 1),
            nn.LeakyReLU()

        ]
        
        
        conv3 = [

            nn.Conv2d(in_channels = self.opt['gen_hidden_channels'] * 2, out_channels = self.opt['gen_hidden_channels'], 
                      kernel_size = (2,2), stride = 1, padding = 1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels = self.opt['gen_hidden_channels'], out_channels = convs_output_channels, 
                      kernel_size = (2,2), stride = 1, padding = 0),
            nn.LeakyReLU()

        ]
        
        output_conv = [
            nn.Conv2d(in_channels = convs_output_channels, out_channels = min(self.generate_tensor_shape), 
                      kernel_size = (3,3), stride = 1, padding = 1),
            nn.ReLU()
        ]
        

        
        self.input_latent_dim = nn.Sequential(*input_latent_dim)

        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.conv3 = nn.Sequential(*conv3)

        self.output_conv = nn.Sequential(*output_conv)



    # __________________________________________


    def forward(self, x):
        x = self.input_latent_dim(x)
        x = x.reshape(x.shape[0], 1, max(self.generate_tensor_shape), max(self.generate_tensor_shape))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output_conv(x)
        
        return x
    
    
    # __________________________________________
    
    def add_noise(self, tensor, scale = 0.1, noise_type = 'add'):
        
        if noise_type == 'multiply':
            noise_tensor = (torch.rand(tuple(tensor.shape)) * (scale[1]-scale[0])) + scale[0]
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor
    
        elif noise_type == 'add':
            if scale == 0: return tensor
            noise_tensor = torch.rand(tuple(tensor.shape)) * tensor.max().to('cpu') * scale
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor + noise_tensor 
        
        elif noise_type == 'zeros':
            noise_tensor = (torch.rand(tuple(tensor.shape)) * (scale[1]-scale[0])) + scale[0]
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor.round()
        
        elif noise_type == 'add_n':
            mean, std = scale[0]*tensor.mean(), scale[1]*tensor.std()
            noise_tensor = torch.randn(tensor.shape).to(self.opt['device']) * std + mean
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor + noise_tensor

        elif noise_type == 'multiply_n':
            mean, std = scale
            noise_tensor = torch.randn(tensor.shape).to(self.opt['device']) * std + mean
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor
        
        elif noise_type == 'multiply_e_n':
            mean, std = scale
            noise_tensor = torch.randn(tensor.shape) * std + mean
            noise_tensor = np.exp(noise_tensor)
            noise_tensor = noise_tensor.to(self.opt['device'])
            return tensor * noise_tensor

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

    def change_device(self, device_name):
        self.device = device_name
        self.opt['device'] = device_name

        if self.smooth is not None: 
            self.smooth = self.smooth.to(self.device)
            self.smooth.weight = self.smooth.weight.to(self.device)

    def generate(self, num_samples):
        return self(torch.randn((num_samples, self.latent_dim)).to(self.device)).to(self.device)

# ___________________________________________________________________________________________________________________________________________________
