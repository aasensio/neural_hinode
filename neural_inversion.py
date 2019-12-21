import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
import h5py
from tqdm import tqdm
import time
import argparse

# Import all models
import model_inversion
import vae
import model_synthesis


class deep_3d_inversion(object):
    def __init__(self, saveplots=True):
        self.cuda = torch.cuda.is_available()

        if (self.cuda):
            print("Using GPU")
        else:
            print("Using CPU")

        self.device = torch.device("cuda" if self.cuda else "cpu")
                       
        self.ltau = np.array([0.0,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0])

        self.variable = ["T", "v$_z$", "h", "log P", "$(B_x^2-B_y^2)^{1/2}$", "$(B_x B_y)^{1/2}$", "B$_z$"]
        self.variable_txt = ["T", "vz", "tau", "logP", "sqrtBx2By2", "sqrtBxBy", "Bz"]
        self.units = ["K", "km s$^{-1}$", "km", "cgs", "kG", "kG", "kG"]
        self.multiplier = [1.0, 1.e-5, 1.e-5, 1.0, 1.0e-3, 1.0e-3, 1.0e-3]

        self.z_tau1 = 1300.0

        self.saveplots = saveplots
        
        self.gammas = 0.001
        self.files_weights = '2019-12-11-10:59:53_-lr_0.0003'       

    def load_weights(self, checkpoint=None):

        self.checkpoint = '{0}.pth'.format(checkpoint)                        
        
        print("  - Defining synthesis NN...")
        self.model_synth = model_synthesis.block(in_planes=7*7, out_planes=40).to(self.device)
    
        print("  - Defining inversion NN...")
        self.model_inversion = model_inversion.block(in_planes=112*4, out_planes=20).to(self.device)
        
        print("  - Defining synthesis VAE...")        
        self.vae_syn = vae.VAE(length=112*4, n_latent=40).to(self.device)

        print("  - Defining model VAE...")        
        self.vae_mod = vae.VAE(length=7*7, n_latent=20).to(self.device)

        tmp = self.checkpoint.split('.')
        f_normal = '{0}.normalization.npz'.format('.'.join(tmp[0:-1]))
        tmp = np.load(f_normal)
        self.phys_min, self.phys_max = tmp['minimum'], tmp['maximum']
                        
        tmp = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        
        self.model_synth.load_state_dict(tmp['synth_state_dict'])        
        print("     => loaded checkpoint for synthesis'{}'".format(self.checkpoint))       
        self.model_synth.eval()
                                        
        self.model_inversion.load_state_dict(tmp['inv_state_dict'])        
        print("     => loaded checkpoint for inversion '{}'".format(self.checkpoint))     
        self.model_inversion.eval()
                                                        
        self.vae_syn.load_state_dict(tmp['vae_syn_state_dict'])        
        print("     => loaded checkpoint for VAE '{}'".format(self.checkpoint))
        self.vae_syn.eval()

        self.vae_mod.load_state_dict(tmp['vae_mod_state_dict'])        
        print("     => loaded checkpoint for VAE '{}'".format(self.checkpoint))
        self.vae_mod.eval()
                       
    def test_hinode(self, parsed):

        print(f"Reading input file {parsed['input']}")

        f = h5py.File(parsed['input'], 'r')

        self.stokes = f['stokes'][:,:,:,:]

        if (parsed['normalize'] is not None):
            x0, x1, y0, y1 = parsed['normalize']
            print(f"Data will be normalized to median value in box : {x0}-{x1},{y0}-{y1}")
            stokes_median = np.median(self.stokes[0,x0:x1,y0:y1,0:3])
        else:
            print(f"Data is already normalized")
            stokes_median = 1.0
        
        f.close()
    
        print(f"Transposing data")
        self.stokes = np.transpose(self.stokes, axes=(0,3,1,2))      

        _, n_lambda, nx, ny = self.stokes.shape

        nx_int = nx // 2**4
        ny_int = ny // 2**4
        nx = nx_int * 2**4
        ny = ny_int * 2**4

        print(f"Cropping map to range (0,{nx})-(0,{ny}) ")

        self.stokes = self.stokes[:,:,0:nx,0:ny]

        print(f"Normalizing data")
        
        self.stokes /= stokes_median

        self.stokes[1,:,:,:] /= 0.1
        self.stokes[2,:,:,:] /= 0.1
        self.stokes[3,:,:,:] /= 0.1

        self.stokes = np.expand_dims(self.stokes.reshape((4*n_lambda,nx,ny)), axis=0)
                                                       
        logtau = np.linspace(0.0, -3.0, 70)

        self.load_weights(checkpoint=self.files_weights)
                
        print("Running neural network inversion...")

        start = time.time()
        input = torch.as_tensor(self.stokes[0:1,:,:,:].astype('float32')).to(self.device)
            
        with torch.no_grad():                
            output_model_latent = self.model_inversion(input)
            output_model = self.vae_mod.decode(output_model_latent)
            output_latent = self.model_synth(output_model)
            output_stokes = self.vae_syn.decode(output_latent)

        end = time.time()
        print(f"Elapsed time : {end-start} s - {1e6*(end-start)/(nx*ny)} us/pixel")

        # Transform the tensors to numpy arrays and undo the transformation needed for the training
        print("Saving results")
        output_model = np.squeeze(output_model.cpu().numpy())
        output_model = output_model * (self.phys_max[:,None,None] - self.phys_min[:,None,None]) + self.phys_min[:,None,None]    
        output_model = output_model.reshape((7,7,nx,ny))

        # Do the same 
        output_stokes = output_stokes.cpu().numpy()
                        
        stokes_output = output_stokes[0,:,:,:].reshape((4,112,nx,ny))
        stokes_output[1:,:] *= 0.1
                
        stokes_original = self.stokes[0,:,:,:].reshape((4,112,nx,ny))        
        stokes_original[1:,:] *= 0.1

        
        tmp = '.'.join(self.checkpoint.split('/')[-1].split('.')[0:2])
        f = h5py.File(f"{parsed['output']}", 'w')
        db_logtau = f.create_dataset('tau_axis', self.ltau.shape)
        db_T = f.create_dataset('T', output_model[0,:,:,:].shape)
        db_vz = f.create_dataset('vz', output_model[1,:,:,:].shape)
        db_tau = f.create_dataset('tau', output_model[2,:,:,:].shape)
        db_logP = f.create_dataset('logP', output_model[3,:,:,:].shape)
        db_Bx2_By2 = f.create_dataset('sqrt_Bx2_By2', output_model[4,:,:,:].shape)
        db_BxBy = f.create_dataset('sqrt_BxBy', output_model[5,:,:,:].shape)
        db_Bz = f.create_dataset('Bz', output_model[6,:,:,:].shape)
        db_Bx = f.create_dataset('Bx', output_model[4,:,:,:].shape)
        db_By = f.create_dataset('By', output_model[5,:,:,:].shape)

        Bx = np.zeros_like(db_Bz[:])
        By = np.zeros_like(db_Bz[:])
                    

        db_logtau[:] = self.ltau
        db_T[:] = output_model[0,:,:,:] * self.multiplier[0]
        db_vz[:] = output_model[1,:,:,:] * self.multiplier[1]
        db_tau[:] = output_model[2,:,:,:] * self.multiplier[2]
        db_logP[:] = output_model[3,:,:,:] * self.multiplier[3]
        db_Bx2_By2[:] = output_model[4,:,:,:] * self.multiplier[4]
        db_BxBy[:] = output_model[5,:,:,:] * self.multiplier[5]
        db_Bz[:] = output_model[6,:,:,:] * self.multiplier[6]

        A = np.sign(db_Bx2_By2[:]) * db_Bx2_By2[:]**2    # I saved sign(Bx^2-By^2) * np.sqrt(Bx^2-By^2)
        B = np.sign(db_BxBy[:]) * db_BxBy[:]**2    # I saved sign(Bx*By) * np.sqrt(Bx*By)

    # This quantity is obviously always >=0
        D = np.sqrt(A**2 + 4.0*B**2)
        
        ind_pos = np.where(B >0)
        ind_neg = np.where(B < 0)
        ind_zero = np.where(B == 0)
        Bx[ind_pos] = np.sign(db_BxBy[:][ind_pos]) * np.sqrt(A[ind_pos] + D[ind_pos]) / np.sqrt(2.0)
        By[ind_pos] = np.sqrt(2.0) * B[ind_pos] / np.sqrt(1e-1 + A[ind_pos] + D[ind_pos])
        Bx[ind_neg] = np.sign(db_BxBy[:][ind_neg]) * np.sqrt(A[ind_neg] + D[ind_neg]) / np.sqrt(2.0)
        By[ind_neg] = -np.sqrt(2.0) * B[ind_neg] / np.sqrt(1e-1 + A[ind_neg] + D[ind_neg])
        Bx[ind_zero] = 0.0
        By[ind_zero] = 0.0

        db_Bx[:] = Bx
        db_By[:] = By

        f.close()
        
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Fast 3D LTE inversion of Hinode datasets')
    parser.add_argument('-i', '--input', default=None, type=str,
                    metavar='INPUT', help='Input file', required=True)
    parser.add_argument('-o', '--output', default=None, type=str,
                    metavar='OUTPUT', help='Output file', required=True)
    parser.add_argument('-n', '--normalize', default=None, type=int, nargs='+',
                    metavar='OUTPUT', help='Output file', required=False)

    parsed = vars(parser.parse_args())

    deep_network = deep_3d_inversion(saveplots=False)

    # ar10933, ar11429, ar11967, qs
    deep_network.test_hinode(parsed)