import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import matplotlib
import numpy as np
import os
from l5kit.geometry import transform_points
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from models.rl.criticmodel import transform_points_tensor
# matplotlib.use('TkAgg')  # or another supported GUI backend
def vis(pred,ori,image,raster):
    idx1=0
    idx2=10
    idx3=20
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,8))

    raster = raster.detach().cpu().numpy()
    image = image.permute(0,2,3,1).detach().cpu().numpy()
    pred_tensor = torch.stack(pred, dim=0)
    pred = pred_tensor.detach().cpu().numpy()
    ori = ori.detach().cpu().numpy()
    ori_raster = transform_points(ori,raster)



    ax1.imshow(image[idx1,:,:,-3:]*0.5+0.5)


    for i in range(5):
        sample = pred[i,:,:,:]
        recon_raster = transform_points(sample,raster)
        ax1.scatter(recon_raster[idx1,:,0],recon_raster[idx1,:,1],c='b',s=0.1,label='future')
        ax1.scatter(ori_raster[idx1,:,0],ori_raster[idx1,:,1],c='r',s=0.2,label='future')
    ax2.imshow(image[idx2,:,:,-3:]*0.5+0.5)

    for i in range(5):
        sample = pred[i,:,:,:]
        recon_raster = transform_points(sample,raster)
        ax2.scatter(recon_raster[idx2,:,0],recon_raster[idx2,:,1],c='b',s=0.1,label='future')
        ax2.scatter(ori_raster[idx2,:,0],ori_raster[idx2,:,1],c='r',s=0.2,label='future')
    ax3.imshow(image[idx1,:,:,-3:]*0.5+0.5)

    for i in range(5):
        sample = pred[i,:,:,:]
        recon_raster = transform_points(sample,raster)
        ax3.scatter(recon_raster[idx3,:,0],recon_raster[idx3,:,1],c='b',s=0.1,label='future')
        ax3.scatter(ori_raster[idx3,:,0],ori_raster[idx3,:,1],c='r',s=0.2,label='future')
    print("11111")
    
def vis_in_out(image,target_raster, hist_raster,recon_fut,indices=[0]):
   
   
    image_rgb_list = []
    for idx in indices:
        image_rgb = image[idx][...,-3:]*0.5+0.5
        image_rgb_list.append(image_rgb)
    fig, axes = plt.subplots(len(indices), 2, figsize=(20, 10*len(indices)))

    if len(indices) == 1:
        axes = np.array([axes])

    for i, idx in enumerate(indices):
        ax_left = axes[i,0]
        ax_right = axes[i,1]
        image_i = image_rgb_list[i]

        ax_left.imshow(image_i)
        ax_left.scatter(target_raster[idx,:,0],target_raster[idx,:,1],c='b',s=0.5,label='future')
        ax_left.scatter(hist_raster[idx,:,0],hist_raster[idx,:,1],c='r',s=0.5,label='hist')
       
        
        ax_left.set_title(f"batch  (idx={idx})")

        ax_right.imshow(image_i)
        ax_right.scatter(recon_fut[idx,:,0],recon_fut[idx,:,1],c='b',s=0.5,label='recon')
             
        ax_right.set_title(f"Output  (idx={idx})")
    plt.tight_layout()
    return fig



class TrajectoryVisualizationCallback(pl.Callback):
    def __init__(self, cfg,media_dir):
        super().__init__()
        self.plot_interval = cfg.train.plt_interval
        self.save_dir = media_dir
        self.indices = cfg.train.plt_indices

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step % self.plot_interval == 0) and (trainer.global_step > 0):
            if isinstance(outputs, list):
                outputs = outputs[0]
            output = outputs['log_dict']
            # Ensure that we have trajectories to plot
            input = output['input'].detach().cpu().numpy()
            raster_from_agent = output['raster_from_agent'].detach().cpu().numpy()

            input_raster = transform_points(input,raster_from_agent)

            hist_position = output['hist'].detach().cpu().numpy()
            hist_raster  = transform_points(hist_position,raster_from_agent)

            image = output['image'].permute(0,2,3,1).detach().cpu().numpy()

            recon_fut = output['output'].detach().cpu().numpy()
            recon_raster = transform_points(recon_fut,raster_from_agent)


            fig = vis_in_out(image, input_raster, hist_raster,recon_raster, indices=self.indices)

            save_path = os.path.join(self.save_dir, f"trajectory_fig_step{trainer.global_step}.png")
            fig.savefig(save_path, dpi=300)

            # Close the figure to free memory
            plt.close(fig)

class VisierProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("MyCustomTrain")
        return bar
    def get_metrics(self, trainer, model):
        metrics = super().get_metrics(trainer, model)
        metrics = {k: v for k, v in metrics.items() if "v_num" not in k}
        metrics["step"] = f"{trainer.global_step}"
        return metrics


class PPOVisualizationCallback(pl.Callback):
    def __init__(self, cfg,media_dir):
        super().__init__()
        self.plot_interval = cfg.train.plt_interval
        self.save_dir = media_dir
        self.indices = cfg.train.plt_indices

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step % self.plot_interval == 0) and (trainer.global_step > 0):
            if isinstance(outputs, list):
                outputs = outputs[0]
          
            
            # Ensure that we have trajectories to plot
            traj = outputs['traj']
            raster_from_agent = outputs['raster_from_agent']
            traj_raster = transform_points_tensor(traj,raster_from_agent)
            traj_raster = traj_raster.detach().cpu().numpy()
            image = outputs['image'].permute(0,2,3,1).detach().cpu().numpy()

            


            fig = ppo_vis(image, traj_raster, indices=self.indices)

            save_path = os.path.join(self.save_dir, f"trajectory_fig_step{trainer.global_step}.png")
            fig.savefig(save_path, dpi=300)

            # Close the figure to free memory
            plt.close(fig)

def ppo_vis(image,traj_raster,indices=[0]):
   
   
    image_rgb_list = []
    for idx in indices:
        image_rgb = image[idx][...,-3:]*0.5+0.5
        image_rgb_list.append(image_rgb)
    fig, axes = plt.subplots(len(indices), 1, figsize=(10, 5 * len(indices)))

    if len(indices) == 1:
        axes = np.array([axes])

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(image_rgb_list[i])
        
   
        ax.scatter(traj_raster[idx,...,0],traj_raster[idx,...,1],c='b',s=0.5,label='future')
             
        
        ax.set_title(f"batch  (idx={idx})")

    
      
    plt.tight_layout()
    return fig