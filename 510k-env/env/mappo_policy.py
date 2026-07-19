"""
MAPPO custom policy: centralized critic using global observation.
Actor uses local obs only; critic uses global obs.
"""
import torch as th
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MAPPOCentralizedCriticPolicy(MaskableActorCriticPolicy):
    """Actor uses local obs; critic uses global obs (via custom feature extractor)."""

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        kwargs['share_features_extractor'] = True
        kwargs['features_extractor_class'] = MAPPOFeatureExtractor
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def extract_features(self, obs, features_extractor=None):
        # Override: return (actor_features, critic_features) tuple
        # SB3's forward() will pass this tuple into mlp_extractor.forward()
        return self.features_extractor(obs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = MAPPOMlpExtractor(
            actor_dim=256, critic_dim=512,
        )

    def predict_values(self, obs):
        features = self.extract_features(obs)
        _, global_feat = features
        latent_vf = self.mlp_extractor.forward_critic(global_feat)
        return self.value_net(latent_vf)

    def get_distribution(self, obs, action_masks=None):
        features = self.extract_features(obs)
        local_feat, _ = features
        latent_pi = self.mlp_extractor.forward_actor(local_feat)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution


class MAPPOFeatureExtractor(BaseFeaturesExtractor):
    """Separate encoders for local (actor) and global (critic) observations.
    Returns a tuple (actor_features, critic_features)."""

    def __init__(self, observation_space):
        local_dim = observation_space['local'].shape[0]
        global_dim = observation_space['global'].shape[0]
        super().__init__(observation_space, features_dim=256 + 512)

        self.local_net = nn.Sequential(
            nn.Linear(local_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.global_net = nn.Sequential(
            nn.Linear(global_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )

    def forward(self, obs):
        local_feat = self.local_net(obs['local'])
        global_feat = self.global_net(obs['global'])
        return local_feat, global_feat


class MAPPOMlpExtractor(nn.Module):
    """Separate MLP heads: actor processes local features, critic processes global features."""

    def __init__(self, actor_dim=256, critic_dim=512):
        super().__init__()
        self.latent_dim_pi = 128
        self.latent_dim_vf = 256

        self.policy_net = nn.Sequential(
            nn.Linear(actor_dim, 256), nn.ReLU(),
            nn.Linear(256, self.latent_dim_pi),
        )
        self.value_net = nn.Sequential(
            nn.Linear(critic_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, self.latent_dim_vf),
        )

    def forward(self, features):
        """Called by SB3 when share_features_extractor=True.
        Expects a tuple (actor_feat, critic_feat)."""
        local_feat, global_feat = features
        return self.policy_net(local_feat), self.value_net(global_feat)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)
