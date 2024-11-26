from tbsim.algos.algos import DiffuserTrafficModel
import torch.optim as optim
import torch
import pytorch_lightning as pl
class Diffusion_Decoder(DiffuserTrafficModel):
    def __init__(self, algo_config, modality_shapes, registered_name,
                 do_log=True, guidance_config=None, constraint_config=None):

        super(Diffusion_Decoder, self).__init__(algo_config, modality_shapes,
                                                registered_name, do_log=do_log,
                                                guidance_config=guidance_config,
                                                constraint_config=constraint_config)
        deprecated_hooks = ["validation_epoch_end", "training_epoch_end"]
        for hook in deprecated_hooks:
            if hasattr(DiffuserTrafficModel, hook):
                delattr(DiffuserTrafficModel, hook)
        self.nets['dm_vae']=None


    def forward_vae_decoder(self, x):
        pass
    def forward_dm(self,x):
        pass
    def forward(self, obs_dict, plan=None, step_index=0, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False, global_t=0):
        pass

    def set_guidance(self, guidance_config, example_batch=None):
        pass

    def on_train_batch_start(self, batch, logs=None):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        print("training_step_start--------------------------------------------------")
        dm_output, vae_output = self.nets['dm_vae'](batch)

        if optimizer_idx == 0:  # 针对 DM 的优化器
            loss_dm = self.compute_dm_loss(dm_output, batch)
            self.log("dm_loss", loss_dm)
            return loss_dm
        elif optimizer_idx == 1:  # 针对 VAE 的优化器
            loss_vae = self.compute_vae_loss(vae_output, batch)
            self.log("vae_loss", loss_vae)
            return loss_vae



    def compute_dm_loss(self, dm_output, batch):
        pass
    def compute_vae_loss(self, vae_output, batch):
        pass


    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        print("Optimizer 1 state:", self.optimizers()[0].state_dict())
        print("Optimizer 2 state:", self.optimizers()[1].state_dict())






    def configure_optimizers(self):
        optim_params_dm = self.algo_config.optim_params["dm"]
        optim_params_vae = self.algo_config.optim_params["vae"]

        optimizer_dm = optim.Adam(
            params=self.nets["dm_vae"].dm.parameters(),
            lr=optim_params_dm["learning_rate"]["initial"],
            weight_decay=optim_params_dm["regularization"]["L2"],
        )
        optimizer_vae = optim.Adam(
            params=self.nets["dm_vae"].vae.parameters(),
            lr=optim_params_vae["learning_rate"]["initial"],
            weight_decay=optim_params_vae["regularization"]["L2"],
        )

        scheduler_dm = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_dm,
            mode='min',
            factor=self.algo_config.optim_params["dm"]["learning_rate"]["decay_factor"],
            patience=5,
            verbose=True
        )
        scheduler_vae = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_vae,
            milestones=optim_params_vae["learning_rate"]["epoch_schedule"],
            gamma=optim_params_vae["learning_rate"]["decay_factor"],
        )

        return (
            [optimizer_dm, optimizer_vae],
            [{"scheduler": scheduler_dm, "interval": "epoch","monitor": "val_loss"},
             {"scheduler": scheduler_vae, "interval": "epoch"}],
        )






