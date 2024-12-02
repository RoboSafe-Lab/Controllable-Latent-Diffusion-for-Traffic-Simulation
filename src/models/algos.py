from tbsim.algos.algos import DiffuserTrafficModel
import torch.optim as optim
import torch,copy
from models.dm_vae import  DMVAE
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.trajdata_utils import convert_scene_data_to_agent_coordinates,  add_scene_dim_to_agent_data, get_stationary_mask
import pytorch_lightning as pl
import torch.nn as nn
import tbsim.utils.tensor_utils as TensorUtils
import numpy as np
from tbsim.policies.common import Plan, Action
from tbsim.utils.guidance_loss import choose_action_from_guidance, choose_action_from_gt
from tbsim.models.diffuser_helpers import EMA

class UnifiedTrainer(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, registered_name,
                 do_log=True, guidance_config=None, constraint_config=None,train_mode="vae", vae_model_path=None):

        super(UnifiedTrainer, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        # assigned at run-time according to the given data batch
        self.data_centric = None
        # ['agent_centric', 'scene_centric']
        self.coordinate = algo_config.coordinate
        # used only when data_centric == 'scene' and coordinate == 'agent'
        self.scene_agent_max_neighbor_dist = algo_config.scene_agent_max_neighbor_dist
        # to help control stationary agent's behavior
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        self.moving_speed_th = algo_config.moving_speed_th
        self.stationary_mask = None
        print(f"algo_config_diffuser_input_mode: {algo_config.diffuser_input_mode}")

        if algo_config.diffuser_input_mode == 'state':
            observation_dim = 0
            action_dim = 3  # x, y, yaw
            output_dim = 3  # x, y, yaw
        elif algo_config.diffuser_input_mode == 'action':
            observation_dim = 0
            action_dim = 2  # acc, yawvel
            output_dim = 2  # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action':
            observation_dim = 4  # x, y, vel, yaw
            action_dim = 2  # acc, yawvel

            output_dim = 2  # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action_no_dyn':
            observation_dim = 4  # x, y, vel, yaw
            action_dim = 2  # acc, yawvel
            output_dim = 6  # x, y, vel, yaw, acc, yawvel
        else:
            raise

        print(f"outputdim={observation_dim}, actiondim={action_dim}, outputdim={output_dim}")


        diffuser_norm_info = algo_config.nusc_norm_info['diffuser']
        agent_hist_norm_info = algo_config.nusc_norm_info['agent_hist']
        neighbor_hist_norm_info = algo_config.nusc_norm_info['neighbor_hist']

        self.cond_drop_map_p = algo_config.conditioning_drop_map_p
        self.cond_drop_neighbor_p = algo_config.conditioning_drop_neighbor_p
        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        assert min_cond_drop_p >= 0.0 and max_cond_drop_p <= 1.0
        self.use_cond = self.cond_drop_map_p < 1.0 and self.cond_drop_neighbor_p < 1.0  # no need for conditioning arch if always dropping
        self.cond_fill_val = algo_config.conditioning_drop_fill

        self.use_rasterized_map = algo_config.rasterized_map
        self.use_rasterized_hist = algo_config.rasterized_history

        if self.use_cond:
            if self.cond_drop_map_p > 0:
                print(
                    'DIFFUSER: Dropping map input conditioning with p = %f during training...' % (self.cond_drop_map_p))
            if self.cond_drop_neighbor_p > 0:
                print('DIFFUSER: Dropping neighbor traj input conditioning with p = %f during training...' % (
                    self.cond_drop_neighbor_p))

        self.nets['dm_vae'] = DMVAE(
            step_time = algo_config.step_time,
            trajectory_shape = algo_config.trajectory_shape,
            condition_dim = algo_config.condition_dim,
            latent_dim = algo_config.latent_dim,
            rasterized_map=algo_config.rasterized_map,
            use_map_feat_global=algo_config.use_map_feat_global,
            use_map_feat_grid=algo_config.use_map_feat_grid,
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,

            rasterized_hist=algo_config.rasterized_history,
            hist_num_frames=algo_config.history_num_frames + 1,  # the current step is concat to the history
            hist_feature_dim=algo_config.history_feature_dim,

            cond_feature_dim=algo_config.cond_feat_dim,
            curr_state_feature_dim=algo_config.curr_state_feat_dim,

            diffuser_model_arch=algo_config.diffuser_model_arch,
            horizon=algo_config.horizon,

            observation_dim=observation_dim,
            action_dim=action_dim,

            output_dim=output_dim,

            n_timesteps=algo_config.n_diffusion_steps,

            loss_type=algo_config.loss_type,
            clip_denoised=algo_config.clip_denoised,

            predict_epsilon=algo_config.predict_epsilon,
            action_weight=algo_config.action_weight,
            loss_discount=algo_config.loss_discount,
            loss_weights=algo_config.diffusor_loss_weights,

            dim_mults=algo_config.dim_mults,

            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,

            base_dim=algo_config.base_dim,
            diffuser_building_block=algo_config.diffuser_building_block,

            action_loss_only=algo_config.action_loss_only,

            diffuser_input_mode=algo_config.diffuser_input_mode,
            use_reconstructed_state=algo_config.use_reconstructed_state,

            use_conditioning=self.use_cond,
            cond_fill_value=self.cond_fill_val,

            diffuser_norm_info=diffuser_norm_info,
            agent_hist_norm_info=agent_hist_norm_info,
            neighbor_hist_norm_info=neighbor_hist_norm_info,

            disable_control_on_stationary=self.disable_control_on_stationary,


        )
        if guidance_config is not None:
            self.set_guidance(guidance_config)
        if constraint_config is not None:
            self.set_constraints(constraint_config)

        # set up EMA
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            self.ema_policy = copy.deepcopy(self.nets["dm_vae"])
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()

        self.cur_train_step = 0
        self.train_mode = train_mode  # "vae" or "dm"
        if train_mode == "dm":
            # Load pre-trained VAE model and freeze parameters
            if vae_model_path:
                self.nets["dm_vae"].vae.load_state_dict(torch.load(vae_model_path))
                for param in self.nets["dm_vae"].vae.parameters():
                    param.requires_grad = False

    def on_validation_epoch_end(self, *args, **kwargs):
        pass

    def on_train_epoch_end(self, *args, **kwargs):
        pass



    def forward(self, batch, plan=None, step_index=0, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False, global_t=0):
        """
        Forward pass based on training mode.
        """
        if self.train_mode == "vae":
            vae_output = self.nets["dm_vae"].get_vaeloss(batch)
            return vae_output
        elif self.train_mode == "dm":
            dm_output = self.nets["dm_vae"].get_dmloss(batch)
            return dm_output

    def set_guidance(self, guidance_config, example_batch=None):
        pass

    def on_train_batch_start(self, batch, logs=None):
        pass

    def training_step(self, batch, batch_idx):
        """
        Training step based on training mode.
        """
        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            pass
        elif self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, merge_BM=True,
                                                            max_neighbor_dist=self.scene_agent_max_neighbor_dist)
        else:
            raise NotImplementedError

            # drop out conditioning if desired
        if self.use_cond:
            if self.use_rasterized_map:
                num_sem_layers = batch['maps'].size(
                    1)  # NOTE: this assumes a trajdata-based loader. Will not work with lyft-specific loader.
                if self.cond_drop_map_p > 0:
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_map_p
                    # only fill the last num_sem_layers as these correspond to semantic map
                    batch["image"][drop_mask, -num_sem_layers:] = self.cond_fill_val

            if self.use_rasterized_hist:
                # drop layers of map corresponding to histories
                # NOTE: this assumes a trajdata-based loader. Will not work with lyft-specific loader.
                num_sem_layers = batch['maps'].size(1) if batch['maps'] is not None else None
                if self.cond_drop_neighbor_p > 0:
                    # sample different mask so sometimes both get dropped, sometimes only one
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_neighbor_p
                    if num_sem_layers is None:
                        batch["image"][drop_mask] = self.cond_fill_val
                    else:
                        # only fill the layers before semantic map corresponding to trajectories (neighbors and ego)
                        batch["image"][drop_mask, :-num_sem_layers] = self.cond_fill_val
            else:
                if self.cond_drop_neighbor_p > 0:
                    # drop actual neighbor trajectories instead
                    # set availability to False, will be zeroed out in model
                    B = batch["all_other_agents_history_availabilities"].size(0)
                    drop_mask = torch.rand((B)) < self.cond_drop_neighbor_p
                    batch["all_other_agents_history_availabilities"][drop_mask] = 0
        if self.train_mode == "vae":
            vae_output = self(batch)
            loss = self.compute_vae_loss(vae_output, batch)
            self.log("vae_loss", loss)
            return loss

        elif self.train_mode == "dm":
            dm_output = self(batch)
            loss = self.compute_dm_loss(dm_output, batch)
            self.log("dm_loss", loss)
            return loss



    def compute_vae_loss(self, vae_output, batch):
        recon_loss = torch.nn.functional.mse_loss(vae_output["recons"], batch["target"])
        kl_loss = vae_output["kl"]
        return recon_loss + self.algo_config.vae_loss_weight * kl_loss

    def compute_dm_loss(self, dm_output, batch):
        """
        Compute DM loss.
        """
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        print("Optimizer 1 state:", self.optimizers()[0].state_dict())
        print("Optimizer 2 state:", self.optimizers()[1].state_dict())

    def configure_optimizers(self):
        """
        Configure optimizers based on training mode.
        """
        if self.train_mode == "vae":
            optim_params_vae = self.algo_config.optim_params["vae"]
            optimizer = optim.Adam(
                params=self.nets["dm_vae"].vae.parameters(),
                lr=optim_params_vae["learning_rate"]["initial"],
                weight_decay=optim_params_vae["regularization"]["L2"],
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=optim_params_vae["learning_rate"]["epoch_schedule"],
                gamma=optim_params_vae["learning_rate"]["decay_factor"],
            )
        elif self.train_mode == "dm":
            optim_params_dm = self.algo_config.optim_params["dm"]
            optimizer = optim.Adam(
                params=self.nets["dm_vae"].dm.parameters(),
                lr=optim_params_dm["learning_rate"]["initial"],
                weight_decay=optim_params_dm["regularization"]["L2"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=optim_params_dm["learning_rate"]["decay_factor"],
                patience=5,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown training mode: {self.train_mode}")

        # 直接返回优化器和调度器，而不是列表
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # 根据您的验证指标
            },
        }

    def on_train_epoch_end(self):
        # Switch training phases
        if self.training_phase == "vae":
            print("VAE pretraining completed. Switching to DM training.")
            self.training_phase = "dm"
            # Freeze VAE
            for param in self.nets["dm_vae"].vae.parameters():
                param.requires_grad = False
        elif self.training_phase == "dm":
            print("DM training ongoing.")

    @property
    def checkpoint_monitor_keys(self):
        if self.use_ema:
            return {"valLoss": "val/ema_losses_diffusion_loss"}
        else:
            return {"valLoss": "val/losses_diffusion_loss"}
    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.nets["dm_vae"].state_dict())
    def step_ema(self, step):#fixme:需要修改一下
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["dm_vae"])
    def on_training_step_end(self, batch_parts):
        self.cur_train_step += 1

    def validation_step(self, batch, batch_idx):
        cur_policy = self.nets["dm_vae"]
        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            pass
        elif self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, merge_BM=True, max_neighbor_dist=self.scene_agent_max_neighbor_dist)
        else:
            raise NotImplementedError

        losses =TensorUtils.detach(cur_policy.compute_losses(batch))

        pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False)
        metrics = self._compute_metrics(pout, batch)
        return_dict =  {"losses": losses, "metrics": metrics}

        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
            ema_losses = TensorUtils.detach(cur_policy.compute_losses(batch))
            pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False)
            ema_metrics = self._compute_metrics(pout, batch)
            return_dict["ema_losses"] = ema_losses
            return_dict["ema_metrics"] = ema_metrics

        return return_dict

    def on_validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)
        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

        if self.use_ema:
            for k in outputs[0]["ema_losses"]:
                m = torch.stack([o["ema_losses"][k] for o in outputs]).mean()
                self.log("val/ema_losses_" + k, m)
            for k in outputs[0]["ema_metrics"]:
                m = np.stack([o["ema_metrics"][k] for o in outputs]).mean()
                self.log("val/ema_metrics_" + k, m)

    def get_plan(self, obs_dict, **kwargs):
        plan = kwargs.get("plan", None)
        preds = self(obs_dict, plan)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}

    def get_action(self, obs_dict,
                   num_action_samples=1,
                   class_free_guide_w=0.0,
                   guide_as_filter_only=False,
                   guide_with_gt=False,
                   guide_clean=False,
                   **kwargs):
        plan = kwargs.get("plan", None)

        cur_policy = self.nets["dm_vae"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy

            # sanity chech that policies are different
            # for current_params, ma_params in zip(self.nets["policy"].parameters(), self.ema_policy.parameters()):
            #     old_weight, up_weight = ma_params.data, current_params.data
            #     print(torch.sum(old_weight - up_weight))
            # exit()

        # already called in policy_composer, but just for good measure...
        cur_policy.eval()

        # update with current "global" timestep
        cur_policy.update_guidance(global_t=kwargs['step_index'])

        preds = self(obs_dict, plan, num_samp=num_action_samples,
                     class_free_guide_w=class_free_guide_w, guide_as_filter_only=guide_as_filter_only,
                     guide_clean=guide_clean, global_t=kwargs['step_index'])
        # [B, N, T, 2]
        B, N, _, _ = preds["positions"].size()

        # arbitrarily use the first sample as the action by default
        act_idx = torch.zeros((B), dtype=torch.long, device=preds["positions"].device)
        if guide_with_gt and "target_positions" in obs_dict:
            act_idx = choose_action_from_gt(preds, obs_dict)
        elif cur_policy.current_perturbation_guidance.current_guidance is not None:
            # choose sample closest to desired guidance
            guide_losses = preds.pop("guide_losses", None)

            # from tbsim.models.diffuser_helpers import choose_act_using_guide_loss
            # act_idx = choose_act_using_guide_loss(guide_losses, cur_policy.current_perturbation_guidance.current_guidance.guide_configs, act_idx)
            act_idx = choose_action_from_guidance(preds, obs_dict,
                                                  cur_policy.current_perturbation_guidance.current_guidance.guide_configs,
                                                  guide_losses)

        action_preds = TensorUtils.map_tensor(preds, lambda x: x[torch.arange(B), act_idx])

        preds_positions = preds["positions"]
        preds_yaws = preds["yaws"]

        action_preds_positions = action_preds["positions"]
        action_preds_yaws = action_preds["yaws"]

        if self.disable_control_on_stationary and self.stationary_mask is not None:
            stationary_mask_expand = self.stationary_mask.unsqueeze(1).expand(B, N)

            preds_positions[stationary_mask_expand] = 0
            preds_yaws[stationary_mask_expand] = 0

            action_preds_positions[self.stationary_mask] = 0
            action_preds_yaws[self.stationary_mask] = 0

        info = dict(
            action_samples=Action(
                positions=preds_positions,
                yaws=preds_yaws
            ).to_dict(),
            # diffusion_steps={
            #     'traj' : action_preds["diffusion_steps"] # full state for the first sample
            # },
        )
        action = Action(
            positions=action_preds_positions,
            yaws=action_preds_yaws
        )
        return action, info

    def set_guidance(self, guidance_config, example_batch=None):

        cur_policy = self.nets["dm_vae"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance(guidance_config, example_batch)

    def clear_guidance(self):
        cur_policy = self.nets["dm_vae"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.clear_guidance()

    def set_constraints(self, constraint_config):
        '''
        Resets the test-time hard constraints to follow during prediction.
        '''
        cur_policy = self.nets["dm_vae"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_constraints(constraint_config)

    def set_guidance_optimization_params(self, guidance_optimization_params):
        '''
        Resets the test-time guidance_optimization_params.
        '''
        cur_policy = self.nets["dm_vae"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_optimization_params(guidance_optimization_params)

    def set_diffusion_specific_params(self, diffusion_specific_params):
        cur_policy = self.nets["dm_vae"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_diffusion_specific_params(diffusion_specific_params)


