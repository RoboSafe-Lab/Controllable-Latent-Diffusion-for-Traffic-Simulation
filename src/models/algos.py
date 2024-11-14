from tbsim.algos.algos import DiffuserTrafficModel
import torch.optim as optim

class Diffusion_Decoder(DiffuserTrafficModel):
    def __init__(self, algo_config, modality_shapes, registered_name,
                 do_log=True, guidance_config=None, constraint_config=None):

        super(Diffusion_Decoder, self).__init__(algo_config, modality_shapes,
                                                registered_name, do_log=do_log,
                                                guidance_config=guidance_config,
                                                constraint_config=constraint_config)
        self.nets["vae"]=None

    def forward_vae_decoder(self, x):
        pass
    def forward_dm(self,x):
        pass
    def forward(self, obs_dict, plan=None, step_index=0, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False, global_t=0):
        pass

    def set_guidance(self, guidance_config, example_batch=None):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_step_end(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optim_params_dm = self.algo_config.optim_params["policy"]
        optim_params_vae = self.algo_config.optim_params["vae"]
        return [
            optim.Adam(params=self.nets["policy"].parameters(),
                       lr=optim_params_dm["learning_rate"]["initial"]),
            optim.Adam(params=self.nets["vae"].parameters(),
                       lr=optim_params_vae["learning_rate"]["initial"])
            ]

