import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import os

def visualize_multiple_trajectories_scatter(
    origin_traj, 
    recon_state_descaled, 
    indices=[0,15,50,75]
):
  
    if isinstance(origin_traj, torch.Tensor):
        origin_traj = origin_traj.detach().cpu().numpy()
    if isinstance(recon_state_descaled, torch.Tensor):
        recon_state_descaled = recon_state_descaled.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()  # Flatten so we can index as a 1D array

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Extract x, y for original and reconstructed
        original_x = origin_traj[idx, :, 0]
        original_y = origin_traj[idx, :, 1]
        reconstructed_x = recon_state_descaled[idx, :, 0]
        reconstructed_y = recon_state_descaled[idx, :, 1]

        # Plot original trajectory as a line
        ax.plot(original_x, original_y, color='blue', label='Original', alpha=0.7)

        # Mark start and end points for original trajectory
        ax.plot(original_x[0], original_y[0], 'g*', markersize=10, label='Start')
        ax.plot(original_x[-1], original_y[-1], 'ks', markersize=8, label='End')

        # Plot reconstructed trajectory as a line
        ax.plot(reconstructed_x, reconstructed_y, color='red', label='Reconstructed', alpha=0.7)

        # Mark start and end points for reconstructed trajectory
        ax.plot(reconstructed_x[0], reconstructed_y[0], 'g*', markersize=10)
        ax.plot(reconstructed_x[-1], reconstructed_y[-1], 'ks', markersize=8)

        ax.set_title(f"Trajectory Index: {idx}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)

        # Only show legend on the first subplot to reduce clutter
        if i == 0:
            ax.legend()

    plt.tight_layout()
    return fig

    # # Create the figure and a 2x2 grid of subplots
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # axes = axes.flatten()  # Flatten so we can index as a 1D array

    # for i, idx in enumerate(indices):
    #     ax = axes[i]
    #     # Extract x, y for original and reconstructed
    #     original_x = origin_traj[idx, :, 0]
    #     original_y = origin_traj[idx, :, 1]
    #     reconstructed_x = recon_state_descaled[idx, :, 0]
    #     reconstructed_y = recon_state_descaled[idx, :, 1]

    #     # Scatter original trajectory points
    #     ax.scatter(original_x, original_y, c='blue', s=10, label='Original', alpha=0.7)

    #     # Scatter reconstructed trajectory points
    #     ax.scatter(reconstructed_x, reconstructed_y, c='red', s=10, label='Reconstructed', alpha=0.7)

    #     # Mark start and end points for original
    #     ax.scatter(original_x[0], original_y[0], c='green', marker='*', s=100, label='Start')
    #     ax.scatter(original_x[-1], original_y[-1], c='black', marker='s', s=50, label='End')

    #     # Mark start and end points for reconstructed (using same markers for consistency)
    #     ax.scatter(reconstructed_x[0], reconstructed_y[0], c='green', marker='*', s=100)
    #     ax.scatter(reconstructed_x[-1], reconstructed_y[-1], c='black', marker='s', s=50)

    #     ax.set_title(f"Trajectory Index: {idx}")
    #     ax.set_xlabel("X Position")
    #     ax.set_ylabel("Y Position")
    #     ax.grid(True)

    #     # Only show legend on the first subplot to reduce clutter
    #     if i == 0:
    #         ax.legend()

    # plt.tight_layout()
    # return fig


class TrajectoryVisualizationCallback(pl.Callback):
    def __init__(self, plot_interval=50):
        """
        Args:
            plot_interval (int): Every how many steps to plot and log the figure.
        """
        super().__init__()
        self.plot_interval = plot_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if we are at a plotting step
        if (trainer.global_step % self.plot_interval == 0) and (trainer.global_step > 0):
            # Ensure that we have trajectories to plot
            if hasattr(pl_module, 'latest_origin_traj') and hasattr(pl_module, 'latest_recon_traj'):
                origin_traj = pl_module.latest_origin_traj
                recon_traj = pl_module.latest_recon_traj

                # Select indices to visualize
                indices = [0, 25, 50, 75] if origin_traj.size(0) >= 4 else list(range(origin_traj.size(0)))

                # Call the visualization function and get the figure
                fig = visualize_multiple_trajectories_scatter(
                    origin_traj,
                    recon_traj,
                    indices=indices
                )

                # Log this figure to wandb
                if hasattr(trainer.logger, 'experiment'):
                    import wandb
                    trainer.logger.experiment.log({
                        "trajectory_fig": wandb.Image(fig),
                        "global_step": trainer.global_step
                    })

                # Close the figure after logging to free resources
                plt.close(fig)
