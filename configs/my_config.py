from tbsim.configs.trajdata_nusc_config import NuscTrajdataTrainConfig,NuscTrajdataEnvConfig

class MyCustomTrainConfig(NuscTrajdataTrainConfig):
    def __init__(self):
        super().__init__()
        self.trajdata_data_dirs = {
            "nusc_trainval": "../datasets/nuscenes",
            "nusc_test": "../datasets/nuscenes",
            "nusc_mini": "../datasets/nuscenes/mini",
        }


class MyCustomEnvConfig(NuscTrajdataEnvConfig):
    def __init__(self):
        super().__init__()
        pass