"""Quick test of transfer evaluation pipeline."""
import sys, warnings
warnings.filterwarnings('ignore')

from sb3_contrib import MaskablePPO
from transfer import run_episode

model = MaskablePPO.load('../models/510k_single_final.zip')

for mode in ['single', 'static', 'dynamic']:
    result = run_episode(model, mode, seed=42)
    traj_lens = {p: len(t) for p, t in result['trajectories'].items()}
    print(f'[{mode}] finish={result["finish_order"]} '
          f'rewards={result["rewards"]} '
          f'lens={traj_lens} '
          f'steps={result["episode_length"]}')
