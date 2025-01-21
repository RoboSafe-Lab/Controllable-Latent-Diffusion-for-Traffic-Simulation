from tbsim.configs.scene_edit_config import SceneEditingConfig


class Hf_SceneEditingConfig(SceneEditingConfig):
    def __init__(self, registered_name='trajdata_nusc_diff'):
        super().__init__(registered_name)
        self.name = "visier_visulization"
        self.policy_config = None
        self.trajdata.trajdata_data_dirs = {
                    "nusc_trainval" : "/home/visier/nuscenes",
                }
        self.trajdata.num_scenes_to_evaluate=1
        self.trajdata.trajdata_cache_location = '~/my_custom_cache_location'
        # self.eval_class='DmComposer'
        