"""Linear vs Nonlinear IRL comparison.

Trains a small MLP discriminator (expert vs random) and compares
its implied reward structure with linear IRL weights via feature importance.
"""
import json, os, gzip, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

FEATURE_NAMES = ['MyScore', 'MyHandSize', 'MyStrength', 'TrickScore', 'PassCount']
FEATURE_DIM = len(FEATURE_NAMES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_state_features(traj_dir, prefix):
    """Load feature vectors from trajectory files."""
    traj_path = os.path.join(traj_dir, f'{prefix}_trajectories.pkl.gz')
    with gzip.open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)
    features = []
    for ep_data in trajectories:
        for traj in ep_data['trajectories'].values():
            for entry in traj:
                features.append(entry['features'])
    return np.array(features, dtype=np.float32)


class RewardMLP(nn.Module):
    """Small MLP reward function: state -> scalar reward."""
    def __init__(self, input_dim=FEATURE_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_discriminator(expert_feat, random_feat, epochs=500, lr=1e-3):
    """Train MLP to distinguish expert from random states.
    
    The discriminator output (logit for "this is expert") serves as
    an implicit nonlinear reward function.
    """
    n_expert = len(expert_feat)
    n_random = len(random_feat)
    n_use = min(n_expert, n_random, 20000)  # subsample for speed
    
    idx_exp = np.random.choice(n_expert, n_use, replace=False)
    idx_rand = np.random.choice(n_random, n_use, replace=False)
    
    X = np.vstack([expert_feat[idx_exp], random_feat[idx_rand]])
    y = np.hstack([np.ones(n_use), np.zeros(n_use)])
    
    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    
    X_t = torch.FloatTensor(X).to(DEVICE)
    y_t = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
    
    model = RewardMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    split = int(0.8 * len(X_t))
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t[:split])
        loss = criterion(logits, y_t[:split])
        loss.backward()
        optimizer.step()
        
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_t[split:])
                val_loss = criterion(val_logits, y_t[split:])
                train_acc = ((logits > 0).float() == y_t[:split]).float().mean()
                val_acc = ((val_logits > 0).float() == y_t[split:]).float().mean()
            print(f'  Epoch {ep+1}: train_loss={loss.item():.4f} val_loss={val_loss.item():.4f} '
                  f'train_acc={train_acc.item():.3f} val_acc={val_acc.item():.3f}')
    
    return model


def feature_importance(model, expert_feat, n_samples=5000):
    """Compute permutation importance for each feature.
    
    Measures how much discriminator accuracy drops when each feature is shuffled.
    Higher drop = more important feature.
    """
    model.eval()
    idx = np.random.choice(len(expert_feat), min(n_samples, len(expert_feat)), replace=False)
    X_base = torch.FloatTensor(expert_feat[idx]).to(DEVICE)
    
    with torch.no_grad():
        base_logits = model(X_base)
        base_prob = torch.sigmoid(base_logits).cpu().numpy()
    
    importances = {}
    for i, name in enumerate(FEATURE_NAMES):
        X_perm = X_base.clone()
        perm_idx = torch.randperm(X_perm.size(0))
        X_perm[:, i] = X_perm[perm_idx, i]
        
        with torch.no_grad():
            perm_logits = model(X_perm)
            perm_prob = torch.sigmoid(perm_logits).cpu().numpy()
        
        drop = float((base_prob - perm_prob).mean())
        importances[name] = drop
    
    return importances


# ============================================================
# Main comparison
# ============================================================
print('Loading trajectory data...')
expert_data = {}
# For each policy, load its NATIVE mode trajectories
for mode, dirname, prefix in [
    ('single', 'transfer_data', '510k_single_final_single'),
    ('static', 'transfer_data_pi_static', '510k_static_1818624_steps_static'),
    ('dynamic', 'transfer_data_pi_dynamic', '510k_dynamic_final_dynamic'),
]:
    expert_data[mode] = load_state_features(dirname, prefix)
    print(f'  Expert ({mode}): {len(expert_data[mode])} states')

# For random baseline, we need to generate features from random play
# Run a quick random collection
print('\nCollecting random baseline features (500 episodes)...')
from env.game import Game, GameMode
from env.features import extract_features
import random
random_feat = []
for ep in range(500):
    random.seed(ep + 9999)
    np.random.seed(ep + 9999)
    game = Game(mode=GameMode.SINGLE, num_players=4)
    while not game.is_over:
        pid = game.current_player
        feat = extract_features(game, pid)
        random_feat.append(feat)
        valid = game.get_valid_actions(pid)
        if valid:
            game.play_cards(pid, random.choice(valid).cards)
        elif game.can_pass(pid):
            game.pass_turn(pid)
    if (ep + 1) % 200 == 0:
        print(f'  {ep+1}/500')
random_feat = np.array(random_feat, dtype=np.float32)
print(f'  Random: {len(random_feat)} states\n')

# Load linear IRL weights for comparison
linear_weights = {}
with open('transfer_data/irl_results.json') as f:
    irl_data = json.load(f)
    for mode in ['single', 'static', 'dynamic']:
        w = np.array(irl_data['contrastive_weights'][mode])
        linear_weights[mode] = w

print('=== Linear IRL Weights (contrastive, normalized) ===')
for mode in ['single', 'static', 'dynamic']:
    w = linear_weights[mode]
    print(f'  {mode:<8}: {np.round(w / np.linalg.norm(w), 3)}  (unit vector)')

print('\n' + '=' * 60)
print('Nonlinear IRL via Discriminator')
print('=' * 60)

for mode in ['single', 'static', 'dynamic']:
    print(f'\n--- Training discriminator for {mode} ---')
    model = train_discriminator(expert_data[mode], random_feat, epochs=500)
    
    # Feature importance
    imp = feature_importance(model, expert_data[mode])
    
    # Normalize importance to sum to 1 for comparison
    total = sum(imp.values()) or 1.0
    imp_norm = {k: v / total for k, v in imp.items()}
    
    # Linear weight (normalized)
    w = linear_weights[mode]
    w_norm = w / np.linalg.norm(w)
    
    print(f'\n  {mode.upper()} — Feature Importance Comparison:')
    print(f'  {"Feature":<14} {"Linear Weight":<14} {"NN Importance":<14} {"Agree?":<8}')
    print(f'  {"-"*50}')
    for i, name in enumerate(FEATURE_NAMES):
        lw = w_norm[i]
        ni = imp_norm.get(name, 0)
        # Check if they agree on sign/direction
        agree = 'YES' if (lw > 0) == (ni > 0) else 'NO'
        print(f'  {name:<14} {lw:<+14.3f} {ni:<+14.4f} {agree:<8}')
    
    # Overall agreement: rank correlation
    lw_ranks = np.argsort(-np.abs(w_norm))
    ni_ranks = np.argsort(-np.abs([imp_norm[n] for n in FEATURE_NAMES]))
    rank_agree = np.mean(lw_ranks == ni_ranks)
    print(f'  --- Rank agreement (top feature match): {"YES" if lw_ranks[0] == ni_ranks[0] else "NO"} ---')
    print(f'  Linear top: {FEATURE_NAMES[lw_ranks[0]]} ({w_norm[lw_ranks[0]]:.3f})')
    print(f'  NN     top: {FEATURE_NAMES[ni_ranks[0]]} ({imp_norm[FEATURE_NAMES[ni_ranks[0]]]:.4f})')

print('\nDone.')
