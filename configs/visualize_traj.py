import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import matplotlib
import numpy as np
import os
from l5kit.geometry import transform_points
# matplotlib.use('TkAgg')  # or another supported GUI backend
def vis(batch):
    idx=5
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    image = batch['image'][idx].permute(1,2,0).cpu().numpy()
    ax.imshow(image[...,-3:]*0.5+0.5)
    target_position = batch['target_positions'][idx].cpu().numpy()
    raster_from_agent = batch['raster_from_agent'][idx].cpu().numpy()
    target_raster = transform_points(target_position,raster_from_agent)
    ax.scatter(target_raster[:,0], target_raster[:,1], c='b', s=1, label='future')
    print("111")
def vis_in_out(image,input, output, raster_from_agent,indices=[0]):
   
   
    image_rgb_list = []
    for idx in indices:
        image_rgb = image[idx].transpose(1, 2, 0)[...,-3:]
        image_rgb = image_rgb * 0.5 + 0.5
        image_rgb_list.append(image_rgb)

    fig, axes = plt.subplots(len(indices), 2, figsize=(20, 10*len(indices)))

    if len(indices) == 1:
        axes = np.array([axes])

    for i, idx in enumerate(indices):
        
        input_raster = transform_points(input[i,:,:2],raster_from_agent[i])
        output_raster = transform_points(output[i,:,:2],raster_from_agent[i])
        image_rgb = image_rgb_list[i]

        axes[i, 0].imshow(image_rgb, alpha=0.8)
        axes[i, 0].scatter(input_raster[:, 0], input_raster[:, 1], c='b', s=0.2, label='Input Trajectory')
       
        # axes[i, 0].scatter(input_raster[0, 0], input_raster[0, 1], marker='o', color='green', s=4, label='Start')

        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].legend()
        axes[i, 0].set_title(f"Input Trajectory (idx={idx})")

        # Plot output
        axes[i, 1].imshow(image_rgb, alpha=0.8)
        axes[i, 1].scatter(output_raster[:, 0], output_raster[:, 1], c='b', s=0.2, label='Output Trajectory')
       
        # axes[i, 1].scatter(output_raster[0, 0], output_raster[0, 1], marker='o', color='green', s=4, label='Start')

        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].legend()
        axes[i, 1].set_title(f"Output Trajectory (idx={idx})")

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
            
            recon = outputs['output'].detach().cpu().numpy()
            origin = outputs['input'].detach().cpu().numpy()
            image = outputs['image'].detach().cpu().numpy()
            raster_from_agent = outputs['raster_from_agent'].detach().cpu().numpy()

            fig = vis_in_out(image, origin, recon,raster_from_agent, indices=self.indices)

            save_path = os.path.join(self.save_dir, f"trajectory_fig_step{trainer.global_step}.png")
            fig.savefig(save_path, dpi=300)

            # Close the figure to free memory
            plt.close(fig)


