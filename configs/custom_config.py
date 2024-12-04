from tbsim.configs.trajdata_nusc_config import NuscTrajdataTrainConfig,NuscTrajdataEnvConfig
from tbsim.configs.algo_config import DiffuserConfig

from tbsim.configs.trajdata_nusc_config import NuscTrajdataTrainConfig
from tbsim.configs.nusc_config import NuscEnvConfig
from tbsim.configs.algo_config import DiffuserConfig


class HfCustomTrainConfig(NuscTrajdataTrainConfig):
    def __init__(self):
        super(HfCustomTrainConfig, self).__init__()


        #self.trajdata_source_train = ["nusc_mini-mini_train"]
        #self.trajdata_source_valid = ["nusc_mini-mini_val"]

        self.trajdata_data_dirs = {
            "nusc_trainval" : "/home/visier/nuscenes",
            "nusc_test"     : "/home/visier/nuscenes",
            "nusc_mini"     : "/home/visier/nuscenes",
        }

        self.trajdata_cache_location = "~/my_custom_cache_location"
        self.training.batch_size = 100
        self.training.num_steps = 100000

        self.validation.every_n_steps =4000   #一个step就是一个epoch

        self.logging.log_tb = False
        self.logging.log_wandb = True
        self.logging.wandb_project_name = "Hf_visier"
        self.training.num_data_workers = 20


        self.validation.batch_size=300
        self.validation.num_data_workers=20
        self.save.best_k=3

        self.save.every_n_steps=5000


class HfCustomEnvConfig(NuscTrajdataEnvConfig):
    def __init__(self):
        super(HfCustomEnvConfig, self).__init__()
        pass

class HfCustomAlgoConfig(DiffuserConfig):
    def __init__(self):
        super(HfCustomAlgoConfig, self).__init__()
        self.diffuser_input_mode = 'state_and_action'
        if hasattr(self.optim_params, "policy"):
            del self.optim_params.policy

        self.trajectory_shape = (52,6)
        self.condition_dim = 256
        self.latent_dim = 64

        self.optim_params.dm = {
            "learning_rate": {
                "initial": 2e-4,
                "decay_factor": 0.1,
                "epoch_schedule": [15, 30],
            },
            "regularization": {
                "L2": 0.0,
            },
        }

        self.optim_params.vae = {
            "learning_rate": {
                "initial": 1e-3,
                "decay_factor": 0.5,
                "epoch_schedule": [10, 20],
            },
            "regularization": {
                "L2": 0.01,
            },
        }


