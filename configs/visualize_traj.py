import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import matplotlib
import numpy as np
import os
from l5kit.geometry import transform_points
# matplotlib.use('TkAgg')  # or another supported GUI backend
# def vis(batch):
#     idx=5
#     fig, ax = plt.subplots(1,1,figsize=(8,8))
#     image = batch['image'][idx].permute(1,2,0).cpu().numpy()
#     ax.imshow(image[...,-3:]*0.5+0.5)
#     target_position = batch['target_positions'][idx].cpu().numpy()
#     raster_from_agent = batch['raster_from_agent'][idx].cpu().numpy()
#     target_raster = transform_points(target_position,raster_from_agent)
#     ax.scatter(target_raster[:,0], target_raster[:,1], c='b', s=1, label='future')
#     print("111")
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
        ax_left.scatter(target_raster[idx,:,0],target_raster[idx,:,1],c='b',s=0.2,label='future')
        ax_left.scatter(hist_raster[idx,:,0],hist_raster[idx,:,1],c='r',s=0.2,label='hist')
       
        
        ax_left.set_title(f"batch  (idx={idx})")

        ax_right.imshow(image_i)
        ax_right.scatter(recon_fut[idx,:,0],recon_fut[idx,:,1],c='b',s=0.2,label='recon')
             
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
            # Ensure that we have trajectories to plot
            target_position = outputs['target'].detach().cpu().numpy()
            raster_from_agent = outputs['raster_from_agent'].detach().cpu().numpy()
            target_raster = transform_points(target_position,raster_from_agent)

            hist_position = outputs['hist'].detach().cpu().numpy()
            hist_raster  = transform_points(hist_position,raster_from_agent)

            image = outputs['image'].permute(0,2,3,1).detach().cpu().numpy()

            recon_fut = outputs['output'].detach().cpu().numpy()
            recon_raster = transform_points(recon_fut,raster_from_agent)


            fig = vis_in_out(image, target_raster, hist_raster,recon_raster, indices=self.indices)

            save_path = os.path.join(self.save_dir, f"trajectory_fig_step{trainer.global_step}.png")
            fig.savefig(save_path, dpi=300)

            # Close the figure to free memory
            plt.close(fig)


