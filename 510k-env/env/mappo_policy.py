"""
MAPPO custom policy: centralized critic using global observation.
Relies on parent MaskableActorCriticPolicy for most logic.
"""
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MAPPOCentralizedCriticPolicy(MaskableActorCriticPolicy):
    """Actor uses local obs; critic uses global obs (via custom feature extractor)."""

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        kwargs['share_features_extractor'] = False
        kwargs['features_extractor_class'] = MAPPOFeatureExtractor
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self):
        # Override: build custom MLP that processes local+global features
        self.mlp_extractor = MAPPOMlpExtractor(
            local_dim=256, global_dim=512,
        )


class MAPPOFeatureExtractor(BaseFeaturesExtractor):
    """Separate encoders for local and global observations."""

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
        local = self.local_net(obs['local'])
        global_ = self.global_net(obs['global'])
        return nn.functional.relu(torch.cat([local, global_], dim=-1))


class MAPPOMlpExtractor(nn.Module):
    """Shared MLP: processes concatenated local+global features."""

    def __init__(self, local_dim=256, global_dim=512):
        super().__init__()
        self.latent_dim_pi = 128
        self.latent_dim_vf = 256
        self.shared_net = nn.Sequential(
            nn.Linear(local_dim + global_dim, 512), nn.ReLU(),
        )
        self.policy_net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, self.latent_dim_pi), nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, self.latent_dim_vf), nn.ReLU(),
        )

    def forward_actor(self, features):
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features):
        return self.value_net(self.shared_net(features))
