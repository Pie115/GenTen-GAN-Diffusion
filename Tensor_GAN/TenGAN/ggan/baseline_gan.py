import torch
import torch.nn as nn
import torch.nn.functional as F



class gen_3dGAN(nn.Module):
    def __init__(self, opt = None, verbose = False):
        super(gen_3dGAN, self).__init__()

        if opt is None: 
            print("Opt is required for generator!")
            return

        self.opt = opt
        self.latent_dim = self.opt['latent_dim']
        self.layer_size = self.opt['layer_size']
        self.tensor_shape = self.opt['tensor_shape']
        self.hidden_tensor_shape = self.opt['hidden_tensor_shape']

        def geom_sum(l):
            rv = 1
            for v in l: rv *= v
            return rv

        reshaped_dim = geom_sum(self.hidden_tensor_shape)
        reshape_latent = [
            nn.Linear(self.latent_dim, int(reshaped_dim/2)),
            nn.ReLU(),
            nn.Linear(int(reshaped_dim/2), reshaped_dim),
            nn.ReLU()
        ]

        self.reshape_latent = nn.Sequential(*reshape_latent)


        GAN3D = [
            nn.Conv3d(in_channels = 8, out_channels = 64, kernel_size = (6, 6, 8), padding = 'same'),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features = 64, eps = 1e-6),
            nn.Upsample(scale_factor=(4, 4, 2), mode='nearest'),

            nn.ConstantPad3d((0, 0, 2, 2, 2, 2), 0),
            nn.Conv3d(in_channels = 64, out_channels = 6, kernel_size = (6, 5, 8)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features = 6, eps = 1e-6),
            nn.Upsample(scale_factor=(2, 2, 3), mode='nearest'),

            nn.ConstantPad3d((3, 3, 0, 0, 1, 1), 0),
            nn.Conv3d(in_channels = 6, out_channels = 6, kernel_size = (3, 3, 8)),
            nn.LeakyReLU(),

        ]

        self.GAN3D = nn.Sequential(*GAN3D)


        self.output_conv = nn.Conv3d(in_channels = 6, out_channels = 1, kernel_size = (4, 4, 2))
        nn.init.xavier_normal_(self.output_conv.weight)


        self.final_act = nn.ReLU()
        # self.final_act = lambda x: x

        print("Successfully initialized generator.")


    def forward(self, x):
        num_samples = x.shape[0]

        x = self.reshape_latent(x)
        new_shape = [num_samples] + self.hidden_tensor_shape

        x = x.reshape(new_shape)

        x = self.GAN3D(x)
        x = self.output_conv(x)
        x = self.final_act(x)

        x = x.squeeze()
        
        x = x.permute(0, 3, 1, 2)

        return x


    def sample_latent(self, sample_size = 5):
        return torch.randn(sample_size, self.latent_dim)
    
    def change_device(self, device_name):
        self.device = device_name
        self.opt['device'] = device_name
        
    def generate(self, num_samples):
        return self(torch.randn((num_samples, self.latent_dim)).to(self.device)).to(self.device)
    
# ___________________________________________________________________________________________________________________________________________________

class BaselineGenerator(nn.Module):
    def __init__(self, opt = None):
        super(BaselineGenerator, self).__init__()
        
        if opt is None: return None
        
        self.opt = opt
        self.device = self.opt['device']
        self.latent_dim = self.opt['latent_dim']
        
    
        self.dim = (8, 8, 7, 7)
        
        self.fc = nn.Linear(self.latent_dim, 64 * 7 * 7)
        
        self.conv1 = nn.Conv3d(8, 64, kernel_size=(6, 6, 8), padding='same')
        self.bn1 = nn.BatchNorm3d(64)
        
        self.conv2 = nn.Conv3d(64, 8, kernel_size=(7, 5, 8))
        self.bn2 = nn.BatchNorm3d(8)
        
        self.conv3 = nn.Conv3d(8, 6, kernel_size=(3, 3, 8))
        self.conv4 = nn.Conv3d(6, 1, kernel_size=(2, 2, 2), bias=False)
        
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 3), mode='nearest')
        
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        

        self.new_conv1 = nn.Conv3d(8, 8, kernel_size = (3, 3, 11), bias = False)
        self.new_upsample1 = nn.Upsample(scale_factor = (3, 3, 3), mode='nearest')
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.dim)
        
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.bn1(x)
        x = self.upsample1(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.bn2(x)
        x = self.upsample2(x)
        
        
        # x = self.add_noise(tensor = x, scale = [0, 1], noise_type = 'zeros')
        # x = self.add_noise(tensor = x, scale = [0, 1], noise_type = 'multiply')
            
        
        x = self.new_conv1(x)        
        x = self.new_upsample1(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        
        x = self.sigmoid(x).squeeze()
        
        x = x.permute(0, 3, 1, 2)
        
        return x
    
    
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
    
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim)).to(self.device)
    
    def change_device(self, device_name):
        self.device = device_name
        self.opt['device'] = device_name
        self.smooth = self.smooth.to(self.device)
        self.smooth.weight = self.smooth.weight.to(self.device)
        
    def generate(self, num_samples):
        return self(torch.randn((num_samples, self.latent_dim)).to(self.device)).to(self.device)