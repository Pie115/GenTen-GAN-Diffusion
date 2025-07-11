import time
from collections import defaultdict
import networkx as nx
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from tqdm import tqdm
# from differ.multiview import MultiviewDiffer, TensorDiffer
# from differ.multiview import TensorDiffer
# from dsloader.tensor_creator import create_tensors
import TenGAN.graphrnn as grnn
from os import path
from TenGAN.dsloader import util
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import imageio
from math import exp as exp
# from karateclub.graph_embedding.graph2vec import Graph2Vec

import random

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, opt,
                 gp_weight=10.0, ct_weight=2.0, M=0.1, generator_iterations=1, critic_iterations=5, print_every=50, epoch_print_every=5,
                 use_cuda=False, checkpoints=True, checkpoint_every=500, checkpoint_path='models/',
                 file_name='ctgan', rank_lambda=0.0, penalty_type='fro', wandb=None, rank_penalty_method='A',
                 n_graph_sample_batches=10, batch_size=64, eval_every=250, gs=None, eval_method='graph',
                 given_raw_graphs=False):
        
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        
        self.losses = {'G': [], 'D': [], 'GP': [], 'CT': [], 'gradient_norm': [], 
                       'd_generated_pred':[], 'd_real_pred':[], 
                       'g_smooth_loss':[], 'g_inc_loss':[], 'd_acc':[]}
        
        self.inconsistency_iter = list()
        self.epoch_times = list()
        
        self.generated_images_log = dict()
        self.evaluation_metric_log = dict()
        
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.ct_weight = ct_weight
        self.M = M
        self.batch_size = batch_size
        self.critic_iterations = critic_iterations
        self.generator_iterations = generator_iterations
        self.print_every = print_every
        self.epoch_print_every = epoch_print_every
        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.file_name = file_name
        self.rank_lambda = rank_lambda
        self.opt = opt
        self.penalty_type = penalty_type
        self.sigmoid = nn.Sigmoid()
        
        self.disc_loss_fn = nn.BCELoss()
        
        # if gs is None: raise Exception('gs argument required')
        # self.gs = gs

        # self.base_Gs = defaultdict(list)
        # if given_raw_graphs:
        #     self.raw_graphs = self.gs
        #     print('Converting existing graphs')
        #     for G in tqdm(self.raw_graphs):
        #         for k in range(opt['tensor_slices']):
        #             self.base_Gs[k].append(G[k])
                    
                    
        # Methods supported:
        # A: norm(C * C^T)
        # B: norm(A*A^T) + norm(B*B^T)
        # C: norm(I - A*A^T) + norm(I-B*B^T)
        self.rank_penalty_method = rank_penalty_method
        if rank_penalty_method not in ('A', 'B', 'C', 'D', 'E'):
            raise Exception('Invalid rank penalty method')
        self.wandb = wandb
        self.to_log = {}
        self.n_graph_sample_batches = n_graph_sample_batches
        self.eval_every = eval_every
        self.eval_method = eval_method
        self.current_epoch = None

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _remove_unconnected(self, G):
        to_remove = []
        for node in G.nodes():
            if len(G.edges(node)) == 0:
                to_remove.append(node)

        G.remove_nodes_from(to_remove)
        return G

    def wandb_log(self, data, **kwargs):
        if self.wandb is not None:
            self.wandb.log(data, **kwargs)

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda: data = data.cuda()

        d_real = self.D(data)[0]
        d_generated = self.D(generated_data)[0]

        # Get gradient penalty
        # gradient_penalty = self._gradient_penalty(data, generated_data) 
        gradient_penalty = 0

        self.losses['GP'].append(float(gradient_penalty))

        # Get consistency term
        # print(f'Data shape: {data.size()}')
        # consistency_term = self._consistency_term(data)
        consistency_term = 0
        self.losses['CT'].append(float(consistency_term))

        # Create total loss and optimize
        self.D_opt.zero_grad()

# _______________________ Discriminator Loss ________________________________________________________________________________________________________

        # d_loss = d_real.mean() - d_generated.mean()

# ___________________________________________________________________________________________________________________________________________________


                
        d_loss_pred = self.sigmoid(torch.concat((d_generated, d_real))).squeeze().to(float).to(self.opt['device'])

        d_loss_labels = torch.concat((torch.tensor([1 for _ in range(d_generated.shape[0])]),
                                      torch.tensor([0 for _ in range(d_real.shape[0])]))).to(float).to(self.opt['device'])

# _______________________ Discriminator Loss v2 _____________________________________________________________________________________________________

        d_loss = self.disc_loss_fn(d_loss_pred, d_loss_labels)
# ___________________________________________________________________________________________________________________________________________________


        d_acc = accuracy_score(np.round(d_loss_pred.detach().cpu()).to(int), np.round(d_loss_labels.cpu()).to(int))
        self.losses['d_acc'].append(d_acc)

        # d_loss += consistency_term
        # d_loss += gradient_penalty


        d_loss.backward()

        self.D_opt.step()
        
        
        # self.losses['d_generated_pred'].append(float(d_generated.mean()))
        self.losses['d_real_pred'].append(float(d_real.mean()))

        # Record loss
        self.to_log['discriminator_loss'] = d_loss
        self.to_log['consistency_term'] = float(consistency_term)
        self.to_log['gradient_penalty'] = float(gradient_penalty)
        self.losses['D'].append(float(d_loss))

    # Methods supported:
    # A: norm(C * C^T)
    # B: norm(A*A^T) + norm(B*B^T)
    # C: norm(I - A*A^T) + norm(I-B*B^T)
    def _get_rank_penalty(self, M, factor=None):
        if self.rank_penalty_method == 'A':
            if self.penalty_type == 'fro':
                return self.rank_lambda * torch.norm(M @ torch.transpose(M, 1, 2), p=self.penalty_type)
            else:
                whole = M @ torch.transpose(M, 1, 2)
                total = 0.0
                for i in range(whole.size(0)):
                    total += torch.norm(whole[i, :, :], p=self.penalty_type)
                return self.rank_lambda * total
        elif self.rank_penalty_method == 'B':
            # Same as A
            return self.rank_lambda * torch.norm(M @ torch.transpose(M, 1, 2), p=self.penalty_type)
        elif self.rank_penalty_method == 'C':
            return self.rank_lambda * torch.norm(torch.eye(M.size(1), device='cuda') - M @ torch.transpose(M, 1, 2), p=self.penalty_type)
        elif self.rank_penalty_method == 'D':
            if factor == 'A':
                tmp = torch.transpose(M, 1, 2) @ M
                eye = torch.eye(M.size(2), device='cuda')
            elif factor == 'B':
                tmp = M @ torch.transpose(M, 1, 2)
                eye = torch.eye(M.size(1), device='cuda')
            else:
                print('No factor specified for penalty method D')
                return None
            return self.rank_lambda * torch.norm(tmp - eye, p=self.penalty_type)
        elif self.rank_penalty_method == 'E':
            if factor == 'A':
                tmp = torch.transpose(M, 1, 2) @ M
            elif factor == 'B':
                tmp = M @ torch.transpose(M, 1, 2)
            else:
                print('No factor specified for penalty method E')
                return None
            diag = torch.diag_embed(torch.diagonal(tmp, dim1=1, dim2=2))
            return self.rank_lambda * torch.norm(tmp - diag, p=self.penalty_type)
        else:
            raise Exception('Unknown rank penalty type!')

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        if self.rank_penalty_method == 'A':
            generated_data = self.sample_generator(batch_size)
        else:
            self.G.set_factor_output(True)
            generated_data, factors = self.sample_generator(batch_size)
            self.G.set_factor_output(False)
                
        # Get generated data
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda: data = data.cuda()

        # d_real = self.D(data)[0]

        d_generated = self.D(generated_data)[0]

        self.losses['d_generated_pred'].append(float(d_generated.mean()))

# _______________________ Generator Loss ____________________________________________________________________________________________________________

        g_loss = d_generated.mean()
        
        g_smooth_loss = 0

        if self.opt['gen_model'] == 'TensorGenerator':
            if self.G.tensor_decomposition: g_smooth_loss = self.opt['gen_channel_smooth'] * self.G.smooth_loss


        if self.current_epoch+1 < self.opt['gen_epoch_start_smooth']: g_smooth_loss *= 0
        self.losses['g_smooth_loss'].append(float(g_smooth_loss))
        g_loss += g_smooth_loss

        g_inconsistency_loss = 0
        if self.opt['gen_epoch_start_inc'] is not None:
            g_inconsistency_loss += 1/(generated_data[:, :, 15:36, 15:36].mean(axis = (1, 2, 3)).std() + 1e-12)
            g_inconsistency_loss += 1/(generated_data[:, :, 15:36, 15:36].std(axis = (1, 2, 3)).std() + 1e-12)
            g_inconsistency_loss *= self.opt['gen_inconsistency_lambda']
            if self.current_epoch+1 < self.opt['gen_epoch_start_inc']: g_inconsistency_loss *= 0
            else: self.inconsistency_iter.append(self.num_steps)
            
        if 'gen_zero_slice_loss' in self.opt:
            if self.opt['gen_zero_slice_loss'] is not None:
                g_zero_slice_loss = (generated_data[:, :, 15:36, 15:36].sum(axis = (2, 3)) < 5e-3).sum().to(torch.float32) # num slices
                g_zero_slice_loss /= float(generated_data.shape[0] * generated_data.shape[1]) # fraction of total slices
                g_zero_slice_loss *= float(self.opt['gen_zero_slice_loss']) # multiply by lambda value
                g_loss += g_zero_slice_loss # add to total loss
                
                if 'gen_zero_slice_loss' not in self.losses: self.losses['gen_zero_slice_loss'] = [g_zero_slice_loss]
                else: self.losses['gen_zero_slice_loss'].append(g_zero_slice_loss)
            
            
        # if (self.current_epoch+1)%3 != 0: g_inconsistency_loss *= 0
        self.losses['g_inc_loss'].append(float(g_inconsistency_loss))
        g_loss += g_inconsistency_loss

# ___________________________________________________________________________________________________________________________________________________


        g_loss_original = float(g_loss)
        if self.rank_lambda > 0:
            graph_size = data.size()[2]
            if self.rank_penalty_method == 'A':
                A = generated_data.view(batch_size, graph_size, graph_size)
                rank_penalty = self._get_rank_penalty(A)
            else:
                rank_penalty = self._get_rank_penalty(factors[0], factor='A') + self._get_rank_penalty(factors[1], factor='B')
            rank_penalty_original = float(rank_penalty)
            g_loss += rank_penalty
            self.to_log['rank_penalty_loss'] = rank_penalty_original
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.to_log['generator_loss_original'] = g_loss_original
        self.to_log['generator_loss'] = float(g_loss)
        self.losses['G'].append(float(g_loss))

    def _consistency_term(self, real_data):
        d1, d_1 = self.D(real_data.to(self.opt['device']))
        d2, d_2 = self.D(real_data.to(self.opt['device']))

        # why max is needed when norm is positive?
        consistency_term = (d1 - d2).norm(2, dim=1) + 0.1 * \
            (d_1 - d_2).norm(2, dim=1) - self.M
        return consistency_term.mean()

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)[0]

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch

        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(float(gradients.norm(2, dim=1).mean()))

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def get_evaluation(self, metric = 'accuracy', disc_input_type = 'generated'):

        self.G.eval()
        self.D.eval()

        sample_size = self.opt['val_gen_size']

        x = self.sample_generator(sample_size)

        discriminator_output = self.D(x)[0].to('cpu')

        if self.opt['disc_sig_out']: sigmoid = lambda x: x
        else: sigmoid = nn.Sigmoid()

        discriminator_classes = sigmoid(discriminator_output).round().to(int)

        if metric == 'accuracy': metric_value = accuracy_score(discriminator_classes, torch.tensor([1 for _ in range(sample_size)]))
        else: metric_value = accuracy_score(discriminator_classes, torch.tensor([1 for _ in range(sample_size)]))

        del discriminator_output, discriminator_classes

        return metric_value


    def _train_epoch(self, data_loader, print_vals=False, epoch_num=-1):

        start_time = time.time()

        self.current_epoch = epoch_num
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            
            

            # Only update discriminator every [generator_iterations] iterations
            if self.num_steps % self.generator_iterations == 0:
                self._critic_train_iteration(data[0])

            # Only update generator every [critic_iterations] iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data[0])

            if print_vals and (i+1) % self.print_every == 0:
                print(f"Batch {i+1} (epoch {epoch_num+1})")
                if self.num_steps > self.critic_iterations:
                    print("D: {}".format(self.losses['D'][-1]))
                    print("d_generated_pred: {}".format(self.losses['d_generated_pred'][-1]))
                    print("d_real_pred: {}".format(self.losses['d_real_pred'][-1]))
                    print("GP: {}".format(self.losses['GP'][-1]))
                    print("CT: {}".format(self.losses['CT'][-1]))
                    # print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))
                    print("g_smooth_loss: {}".format(self.losses['g_smooth_loss'][-1]))
                    print("g_inc_loss: {}".format(self.losses['g_inc_loss'][-1]))
                    if 'gen_zero_slice_loss' in self.losses:
                        print("g_zero_slice_loss: {}".format(self.losses['gen_zero_slice_loss'][-1]))



                evaluation_metric = self.get_evaluation(metric = 'accuracy')
                print(f"Disc Accuracy on Generated Data: {evaluation_metric}")
                if 'd_acc_gen' not in self.losses: self.losses['d_acc_gen'] = list()
                self.losses['d_acc_gen'].append(evaluation_metric)
                
                print(f"D_acc: {self.losses['d_acc'][-1]}")

                dict_key = f"epoch_{self.current_epoch+1}"
                if dict_key not in self.evaluation_metric_log: self.evaluation_metric_log[dict_key] = [evaluation_metric]
                else: self.evaluation_metric_log[dict_key].append(evaluation_metric)
                del dict_key
                
        if print_vals: 
            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)
            print(f"\n\nEpoch time: {self.epoch_times[-1]:.4f}\n")
            print("_"*150)
        
        if self.save_generated_images is not None:
            if ((self.current_epoch+1) % self.save_generated_images == 0) or (self.current_epoch+1 == self.opt['n_epochs']):
                self.generated_images_log[f'epoch_{self.current_epoch+1}'] = self.sample_generator(num_samples = 5)
                
                epoch_str = str(self.current_epoch+1)
                if len(epoch_str) == 1: epoch_str = f'00{epoch_str}'
                elif len(epoch_str) == 2: epoch_str = f"0{epoch_str}"
                
                epoch_save_path = f"saved_generators/temp_save/generator_epoch_{epoch_str}.pth"
                torch.save(self.G.state_dict(), epoch_save_path)
                
                print("\nSaved generator & generated images!\n")
                
                
        
        # self.to_log['epoch'] = epoch_num
        # if self.eval_every > 0 and (epoch_num+1) % self.eval_every == 0:
        #     res = self.evaluate_gen()
        #     if res is not None and self.eval_method == 'graph':
        #         (deg_stat, clustering_stat, orbit_stat, valid_graphs, rank_mean, rank_std, rank_med) = res
        #         self.to_log['degree_mmd'] = deg_stat
        #         self.to_log['clustering_mmd'] = clustering_stat
        #         self.to_log['orbit_mmd'] = orbit_stat
        #         self.to_log['mean_mmd'] = np.mean(res[:3])
        #         self.to_log['mmd_graphs'] = valid_graphs
        #         self.to_log['rank_mean'] = rank_mean
        #         self.to_log['rank_std'] = rank_std
        #         self.to_log['rank_med'] = rank_med
        #     elif res is not None:
        #         for k, v in res:
        #             if k in self.to_log:
        #                 print(f'WARNING: key conflict in to_log for key: {k}. Existing value: {self.to_log[k]}, new value: {v}')
        #             self.to_log[k] = v
        # self.wandb_log(self.to_log)
        # self.to_log = {}


    def save_checkpoint(self, epoch):
        torch.save(self.G.state_dict(), path.join(self.checkpoint_path,
                                                  '{}_generator_epoch-{}.model'.format(self.file_name, epoch)))
        torch.save(self.D.state_dict(), path.join(self.checkpoint_path,
                                                  '{}_discriminator_epoch-{}.model'.format(self.file_name, epoch)))
        torch.save(self.G_opt.state_dict(), path.join(self.checkpoint_path,
                                                      '{}_optimizerG_epoch-{}.model'.format(self.file_name, epoch)))
        torch.save(self.D_opt.state_dict(), path.join(self.checkpoint_path,
                                                      '{}_optimizerD_epoch-{}.model'.format(self.file_name, epoch)))
            

    def train(self, data_loader, epochs, save_training_gif=False, save_generated_images = None):
        
        self.save_generated_images = save_generated_images
        
        if self.save_generated_images is not None:

            import os
            import shutil

            folder_path = "/home/spaka002/Tensor_GAN/saved_generators/temp_save/"

            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
            
            del folder_path
        
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(self.batch_size))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            
            should_print = (epoch % self.epoch_print_every == 0)
            if should_print:
                print(f"\nEpoch {epoch+1} of {self.opt['n_epochs']}")
            self._train_epoch(data_loader, print_vals=should_print, epoch_num=epoch)

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)

            if (epoch+1) % self.checkpoint_every == 0 and self.checkpoints:
                print(f'========== Saving checkpoint at epoch #{epoch} ========')
                self.save_checkpoint(epoch)


        if save_training_gif:
            imageio.mimsave('{}/../training_{}_epochs.gif'.format(self.checkpoint_path, epochs),
                            training_progress_images)

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda: latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def evaluate_gen(self):
        if (self.eval_method == 'graph'):
            return self.eval_gen_graphs()
        elif (self.eval_method == 'multiview'):
            return self.eval_gen_multiview_new()
        else:
            print(f'ERROR: {self.eval_method} method not currently supported')
            return None

    def eval_gen_multiview_new(self):
        gen_Gs = defaultdict(list)
        # revisit

        for i in range(self.n_graph_sample_batches):
            z = self.G.sample_latent(self.batch_size)
            if self.use_cuda:
                z = z.cuda()
            res = self.G(z)
            g_np = res.detach().cpu().numpy()

            for j in range(g_np.shape[0]):
                tmp_views = []
                # shape[1] is number of views
                for k in range(g_np.shape[1]):
                    tmp = g_np[j, k, :, :].copy()

                    # TODO: change this for non-binary graphs
                    util.graph_threshold(tmp, threshold=0.0001)
                    graph = nx.from_numpy_array(tmp, create_using=nx.DiGraph)
                    # remove_unconnected(graph)
                    # f = plt.figure()
                    if len(graph) > 0:
                        tmp_views.append(graph)
                if len(tmp_views) == g_np.shape[1]:
                    for k in range(len(tmp_views)):
                        gen_Gs[k].append(tmp_views[k])
                else:
                    print('WARNING: skipped invalid graph')
                    
        print(f'Generated {len(gen_Gs[0])} graphs for evaluation')

        degree_stats = defaultdict(list)
        clustering_stats = defaultdict(list)
        orbit_stats = defaultdict(list)

        # print(self.base_Gs[k])
        # print(gen_Gs[k])
        for k in range(g_np.shape[1]):
            print('Calulating degree stats...')
            deg_stat = grnn.degree_stats(self.base_Gs[k], gen_Gs[k])
            print(f'Degree difference at epoch {self.current_epoch} and slice {k}: {deg_stat}')
            degree_stats[k].append(deg_stat)

            print('Calulating clustering stats...')
            clustering_stat = grnn.clustering_stats(self.base_Gs[k], gen_Gs[k])
            print(f'Clustering difference at epoch {self.current_epoch} and slice {k}: {clustering_stat}')
            clustering_stats[k].append(clustering_stat)

            print('Calculating orbit stats...')
            orbit_stat = grnn.orbit_stats_all(self.base_Gs[k], gen_Gs[k])
            print(f'Orbit difference at epoch {self.current_epoch} and slice {k}: {orbit_stat}')
            orbit_stats[k].append(orbit_stat)

        mean_degree = np.mean([degree_stats[x] for x in range(g_np.shape[1])])
        mean_clustering = np.mean([clustering_stats[x] for x in range(g_np.shape[1])])
        mean_orbit = np.mean([orbit_stats[x] for x in range(g_np.shape[1])])


        # Do classifier-based eval
        # print('Running classifier-based eval...')
        zipped_gen_Gs = list(zip(*(gen_Gs[x] for x in range(self.opt['tensor_slices']))))
        
        # differ = MultiviewDiffer(self.raw_graphs, zipped_gen_Gs, use_ensemble_model=True, embedding_model=Graph2Vec(), split_first=False, even_out=True)        
        
        # res = differ.eval()
        # f1_score = res['f1']
        # acc_score = res['accuracy']
        
        # base_sample_count = len(self.raw_graphs)
        # gen_sample_count = len(zipped_gen_Gs)

        # Do tensor-based eval
        print('Running tensor-based eval...')

        differ = TensorDiffer(self.raw_graphs, zipped_gen_Gs)
        dists, self_dists = differ.pairwise_sampled_eval(max_rank=self.opt['max_eval_rank'], n_samples=self.opt['tensor_eval_samples'])
        dists_arr, self_dists_arr = np.array(dists), np.array(self_dists)

        filtered_dists = dists_arr[np.isfinite(dists_arr)]
        filtered_self_dists = self_dists_arr[np.isfinite(self_dists_arr)]
        tensor_score = np.sum(filtered_dists) / np.sum(filtered_self_dists)

        # Save every time we finish an epoch
        return {
            'epoch': self.current_epoch,
            'mean_degree': mean_degree,
            'mean_clustering': mean_clustering,
            'mean_orbit': mean_orbit,
            'degree_stats': degree_stats,
            'orbit_stats': orbit_stats,
            'clustering_stats': clustering_stats,
            # 'f1_score': f1_score,
            # 'acc_score': acc_score,
            # 'base_sample_count': base_sample_count,
            # 'gen_sample_count': gen_sample_count,
            'tensor_score': tensor_score
        }

    def eval_gen_multiview(self):
        gen_G = []
        ranks = []
        for i in range(self.n_graph_sample_batches):
            z = self.G.sample_latent(self.batch_size)
            if self.use_cuda:
                z = z.cuda()
            res = self.G(z)
            g_np = res.detach().cpu().numpy()

            for j in range(g_np.shape[0]):
                for k in range(g_np.shape[1]):
                    tmp = g_np[j, k, :, :].copy()
                    (_, S, _) = np.linalg.svd(tmp)
                    ranks.append(util.approx_rank(S))

                    util.graph_threshold(tmp, threshold=0.0001)
                    graph = nx.from_numpy_array(tmp)
                    self._remove_unconnected(graph)

                    if graph.number_of_nodes() > 0:
                        gen_G.append(graph)

        print(f'Generated {len(gen_G)} graphs for evaluation')
        if len(gen_G) == 0:
            print('WARNING: no graphs were generated for evaluation')
            return None
        try:
            deg_stat = grnn.degree_stats(self.base_Gs, gen_G)
            clustering_stat = grnn.clustering_stats(self.base_Gs, gen_G)
            orbit_stat = grnn.orbit_stats_all(self.base_Gs, gen_G)
        except Exception as e:
            print('WARNING: failed to calculate MMD scores:', e)
            return None
        return (deg_stat, clustering_stat, orbit_stat, len(gen_G), np.mean(ranks), np.std(ranks), np.median(ranks))
    
    def eval_gen_graphs(self):
        gen_G = []
        ranks = []
        for i in range(self.n_graph_sample_batches):
            z = self.G.sample_latent(self.batch_size)
            if self.use_cuda:
                z = z.cuda()
            res = self.G(z)
            g_np = res.detach().cpu().numpy()

            for j in range(g_np.shape[0]):
                (_, S, _) = np.linalg.svd(g_np[j, 0, :, :])
                ranks.append(util.approx_rank(S))

                tmp = g_np[j, 0, :, :].copy()
                util.graph_threshold(tmp, threshold=0.0001)
                graph = nx.from_numpy_array(tmp)
                self._remove_unconnected(graph)
                if graph.number_of_nodes() > 0:
                    gen_G.append(graph)
        print(f'Generated {len(gen_G)} graphs for evaluation')
        if len(gen_G) == 0:
            print('WARNING: no graphs were generated for evaluation')
            return None
        try:
            deg_stat = grnn.degree_stats(self.base_Gs, gen_G)
            clustering_stat = grnn.clustering_stats(self.base_Gs, gen_G)
            orbit_stat = grnn.orbit_stats_all(self.base_Gs, gen_G)
        except Exception as e:
            print('WARNING: failed to calculate MMD scores:', e)
            return None
        return (deg_stat, clustering_stat, orbit_stat, len(gen_G), np.mean(ranks), np.std(ranks), np.median(ranks))
        

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]