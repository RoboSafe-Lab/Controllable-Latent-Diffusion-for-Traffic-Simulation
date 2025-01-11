import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import matplotlib
import numpy as np
import os
# matplotlib.use('TkAgg')  # or another supported GUI backend


def vis_in_out(maps,input, output, raster_from_agent,indices=[0]):
   
    raster_from_agent = raster_from_agent
    maps_rgb_list = []
    for idx in indices:
        map_idx = maps[idx]
        maps_rgb = map_idx.transpose(1, 2, 0)
        maps_rgb = maps_rgb * 0.4 + 0.6
        maps_rgb_list.append(maps_rgb)


    input = input[..., :2]
    output = output[..., :2]
    

    B, T, _ = input.shape
    ones = np.ones((B, T, 1))
    fig, axes = plt.subplots(len(indices), 2, figsize=(12, 6 * len(indices)))

    if len(indices) == 1:
        axes = np.array([axes])

    for i, idx in enumerate(indices):

        input_homo = np.concatenate([input, ones], axis=-1)
        input_homo_T = np.transpose(input_homo, (0, 2, 1))
        input_raster = np.matmul(raster_from_agent, input_homo_T)
        input_raster = np.transpose(input_raster, (0, 2, 1))
        input_xy = input_raster[..., :2][idx]

  

        output_homo = np.concatenate([output, ones], axis=-1)
        output_homo_T = np.transpose(output_homo, (0, 2, 1))
        output_raster = np.matmul(raster_from_agent, output_homo_T)
        output_raster = np.transpose(output_raster, (0, 2, 1))
        output_xy = output_raster[..., :2][idx]

        maps_rgb = maps_rgb_list[i]


        axes[i, 0].imshow(maps_rgb, alpha=0.8)
        axes[i, 0].plot(input_xy[:, 0], input_xy[:, 1], color='red', linewidth=1, label='Input Trajectory')
        axes[i, 0].plot(input_xy[0, 0], input_xy[0, 1], marker='o', color='green', markersize=2, label='Start')
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].legend()
        axes[i, 0].set_title(f"Input Trajectory (idx={idx})")

        # Plot output
        axes[i, 1].imshow(maps_rgb, alpha=0.8)
        axes[i, 1].plot(output_xy[:, 0], output_xy[:, 1], color='blue', linewidth=1, label='Output Trajectory')
        axes[i, 1].plot(output_xy[0, 0], output_xy[0, 1], marker='o', color='green', markersize=2, label='Start')
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].legend()
        axes[i, 1].set_title(f"Output Trajectory (idx={idx})")

    plt.tight_layout()
    return fig

# def vis_in_out_list(maps,input, outputs_list, raster_from_agent,indices=[0]):
   
#     raster_from_agent = raster_from_agent
#     maps_rgb_list = []
#     for idx in indices:
#         map_idx = maps[idx]
#         maps_rgb = map_idx.transpose(1, 2, 0)
#         maps_rgb = maps_rgb * 0.4 + 0.6
#         maps_rgb_list.append(maps_rgb)


#     input = input[..., :2]
#     output = output[..., :2]
    

#     B, T, _ = input.shape
#     ones = np.ones((B, T, 1))
#     fig, axes = plt.subplots(len(indices), 2, figsize=(12, 6 * len(indices)))

#     if len(indices) == 1:
#         axes = np.array([axes])

#     for i, idx in enumerate(indices):

#         input_homo = np.concatenate([input, ones], axis=-1)
#         input_homo_T = np.transpose(input_homo, (0, 2, 1))
#         input_raster = np.matmul(raster_from_agent, input_homo_T)
#         input_raster = np.transpose(input_raster, (0, 2, 1))
#         input_xy = input_raster[..., :2][idx]

  

#         output_homo = np.concatenate([output, ones], axis=-1)
#         output_homo_T = np.transpose(output_homo, (0, 2, 1))
#         output_raster = np.matmul(raster_from_agent, output_homo_T)
#         output_raster = np.transpose(output_raster, (0, 2, 1))
#         output_xy = output_raster[..., :2][idx]

#         maps_rgb = maps_rgb_list[i]


#         axes[i, 0].imshow(maps_rgb, alpha=0.8)
#         axes[i, 0].plot(input_xy[:, 0], input_xy[:, 1], color='red', linewidth=1, label='Input Trajectory')
#         axes[i, 0].plot(input_xy[0, 0], input_xy[0, 1], marker='o', color='green', markersize=2, label='Start')
#         axes[i, 0].axis('off')
#         if i == 0:
#             axes[i, 0].legend()
#         axes[i, 0].set_title(f"Input Trajectory (idx={idx})")

#         # Plot output
#         axes[i, 1].imshow(maps_rgb, alpha=0.8)
#         axes[i, 1].plot(output_xy[:, 0], output_xy[:, 1], color='blue', linewidth=1, label='Output Trajectory')
#         axes[i, 1].plot(output_xy[0, 0], output_xy[0, 1], marker='o', color='green', markersize=2, label='Start')
#         axes[i, 1].axis('off')
#         if i == 0:
#             axes[i, 1].legend()
#         axes[i, 1].set_title(f"Output Trajectory (idx={idx})")

#     plt.tight_layout()
#     return fig




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
            
            recon_traj = outputs['output'].detach().cpu().numpy()
            origin_traj = outputs['input'].detach().cpu().numpy()
            maps = outputs['maps'].detach().cpu().numpy()
            raster_from_agent = outputs['raster_from_agent'].detach().cpu().numpy()

            fig = vis_in_out(maps, origin_traj, recon_traj,raster_from_agent, indices=self.indices)

            save_path = os.path.join(self.save_dir, f"trajectory_fig_step{trainer.global_step}.png")
            fig.savefig(save_path, dpi=300)

            # Close the figure to free memory
            plt.close(fig)


