import torch.nn as nn
import tbsim.models.base_models as base_models
from tbsim.models.diffuser_helpers import unicyle_forward_dynamics,MapEncoder,convert_state_to_state_and_action
import torch
from tbsim.utils.batch_utils import batch_utils
from tbsim.models.diffuser_helpers import convert_state_to_state_and_action,unicyle_forward_dynamics
import tbsim.utils.tensor_utils as TensorUtils
class ContextEncoder(nn.Module):
    def __init__(self,state_in_dim,algo_config,modality_shapes,dyn):
        super().__init__()
        self.dyn=dyn
        cond_in_feat_size=0
        curr_state_feature_dim = algo_config.curr_state_feat_dim
        layer_dims = (curr_state_feature_dim, curr_state_feature_dim)
        self.agent_state_encoder = base_models.MLP(state_in_dim,
                                                   curr_state_feature_dim,
                                                   layer_dims,
                                                   normalization=True,)
        cond_in_feat_size += curr_state_feature_dim

        self.map_encoder = MapEncoder(
            model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],
            global_feature_dim=algo_config.map_feature_dim,
            grid_feature_dim=None,
        )
        
        cond_in_feat_size+= algo_config.map_feature_dim

        cond_out_feat_size = algo_config.cond_feat_dim
        combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, cond_out_feat_size, cond_out_feat_size)

        self.process_cond_mlp = base_models.MLP(
            cond_in_feat_size,
            cond_out_feat_size,
            combine_layer_dims,
            normalization=True,
        )

    def forward(self,data_batch):
        N = data_batch["history_positions"].size(0)
        device = data_batch["history_positions"].device
        cond_feat_in = torch.empty((N,0)).to(device)
        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        curr_state_feat = self.agent_state_encoder(curr_states)

        cond_feat_in = torch.cat([cond_feat_in, curr_state_feat], dim=-1)

        image_batch = data_batch['image']
        map_global_feat, _ = self.map_encoder(image_batch)
        cond_feat_in = torch.cat([cond_feat_in,map_global_feat], dim=-1)

        cond_feat = self.process_cond_mlp(cond_feat_in)

        aux_info = {
            'cond_feat': cond_feat,
            'curr_states': curr_states,
            'image': image_batch,
        }

        return aux_info
    

def get_state_and_action_from_data_batch(batch, chosen_inds=[]):

    if len(chosen_inds) == 0:
            chosen_inds = [0, 1, 2, 3, 4, 5] 
    traj_state = torch.cat((batch["target_positions"][:, :52 :], batch["target_yaws"][:, :52, :]), dim=2)
    traj_state_and_action = convert_state_to_state_and_action(traj_state,batch['curr_speed'],0.1)
    return traj_state_and_action[...,chosen_inds]

