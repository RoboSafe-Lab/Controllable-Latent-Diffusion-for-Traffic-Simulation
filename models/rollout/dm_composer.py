from tbsim.evaluation.policy_composers import PolicyComposer
from trainers.dm_trainer import DMLightningModule
from tbsim.utils.experiment_utils import get_checkpoint
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.policies.wrappers import PolicyWrapper
class DmComposer(PolicyComposer):
    def get_policy(self, policy=None):
        policy_ckpt_path, policy_config_path = get_checkpoint(
            ngc_job_id=self.eval_config.ckpt.policy.ngc_job_id,
            ckpt_key=self.eval_config.ckpt.policy.ckpt_key,
            ckpt_dir=self.eval_config.ckpt.policy.ckpt_dir,
            ckpt_root_dir=self.ckpt_root_dir,
        )
        policy_cfg = get_experiment_config_from_file(policy_config_path)
        policy = DMLightningModule.load_from_checkpoint(
            policy_ckpt_path,
            algo_config=policy_cfg.algo,
            train_config=policy_cfg.train,
            modality_shapes=self.get_modality_shapes(policy_cfg),
            registered_name=policy_cfg["registered_name"],
        ).to(self.device).eval()
        policy_cfg = policy_cfg.clone()
        policy = PolicyWrapper.wrap_controller(
            policy,
            num_action_samples=self.eval_config.policy.num_action_samples,
            class_free_guide_w=self.eval_config.policy.class_free_guide_w,
            guide_as_filter_only=self.eval_config.policy.guide_as_filter_only,
            guide_with_gt=self.eval_config.policy.guide_with_gt,
            guide_clean=self.eval_config.policy.guide_clean,
        )
        # TBD: for debugging purpose
        # policy = SceneCentricToAgentCentricWrapper(policy)
        # policy = AgentCentricToSceneCentricWrapper(policy)
        return policy, policy_cfg
