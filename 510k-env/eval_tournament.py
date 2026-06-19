"""
Checkpoint 锦标赛评估：让不同 checkpoint 在 4 人局中对战。
使用 finish_order 作为零和指标（谁先出完牌）。
"""
import os, sys, random
import numpy as np
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(__file__))
from env.card import card_to_id
from env.game import Game, GameMode
from env.scorer import Scorer


def _old_obs(obs_116):
    """Convert 116-dim obs (current) -> 112-dim obs (model format)."""
    return np.concatenate([obs_116[0:54], obs_116[54:108], obs_116[109:113]])


def get_obs(game, agent_pid):
    """Build 116-dim observation from game state for given player."""
    hand = np.zeros(54, dtype=np.float32)
    for c in game.players[agent_pid].hand:
        hand[card_to_id(c)] = 1.0
    last_play = np.zeros(54, dtype=np.float32)
    last_type = np.float32(0.0)
    if game.last_trick and game.last_trick.pattern:
        for c in game.last_trick.cards:
            last_play[card_to_id(c)] = 1.0
        last_type = np.float32(game.last_trick.pattern.type.value)
    hand_sizes = np.array([len(p.hand) for p in game.players], dtype=np.float32)
    cp = np.float32(game.current_player)
    pc = np.float32(game.pass_count)
    score = np.float32(game.player_510k_scores[agent_pid])
    return np.concatenate([hand, last_play, [last_type], hand_sizes, [cp], [pc], [score]])


def make_model_bot(model, deterministic=True):
    def bot(pid, actions, game):
        if not actions:
            return None
        obs = get_obs(game, pid)
        mask = np.zeros(300, dtype=np.int64)
        mask[0] = 1
        for i in range(len(actions)):
            if i + 1 < 300:
                mask[i + 1] = 1
        action, _ = model.predict(_old_obs(obs), action_masks=mask, deterministic=deterministic)
        if action == 0:
            return None
        idx = action - 1
        if 0 <= idx < len(actions):
            return actions[idx]
        return None
    return bot


def run_match(players, n_games=200, seed=42):
    """
    players: list of (name, bot_fn_or_None) for each player slot.
    Reports: first-finish rate (zero-sum), mean reward, 510K score.
    """
    n = len(players)
    first_finish = {name: 0 for name, _ in players}
    second_finish = {name: 0 for name, _ in players}
    rewards = {name: [] for name, _ in players}
    scores_510k = {name: [] for name, _ in players}
    rng = random.Random(seed)

    for g in range(n_games):
        game = Game(mode=GameMode.SINGLE, num_players=n, include_jokers=False)

        while not game.is_over:
            pid = game.current_player
            name, bot_fn = players[pid]
            actions = game.get_valid_actions(pid)
            if not actions:
                game.pass_turn(pid)
            else:
                if bot_fn is None:
                    chosen = rng.choice(actions)
                else:
                    chosen = bot_fn(pid, actions, game)
                if chosen is None:
                    if game.can_pass(pid):
                        game.pass_turn(pid)
                    else:
                        game.play_cards(pid, rng.choice(actions).cards)
                else:
                    game.play_cards(pid, chosen.cards)

        # Record finish order
        if len(game.finish_order) >= 1:
            idx = game.finish_order[0]
            first_finish[players[idx][0]] += 1
        if len(game.finish_order) >= 2:
            idx = game.finish_order[1]
            second_finish[players[idx][0]] += 1

        # Rewards
        scorer = Scorer(game)
        all_rewards = scorer.compute_rewards()
        for i, (name, _) in enumerate(players):
            rewards[name].append(all_rewards.get(i, 0.0))
            scores_510k[name].append(float(game.player_510k_scores[i]))

    print(f'--- {n_games} games ---')
    results = []
    for name, _ in players:
        fr = first_finish[name] / n_games
        sr = second_finish[name] / n_games
        mr = np.mean(rewards[name])
        ms = np.mean(scores_510k[name])
        results.append((fr, name, sr, mr, ms))
    results.sort(key=lambda x: -x[0])
    for fr, name, sr, mr, ms in results:
        print(f'{name:25s}  first_finish={fr:.1%}  second={sr:.1%}  510K={ms:.1f}  reward={mr:+5.1f}')
    print(f'  Total first_finish: {sum(r[0] for r in results):.1%}')
    return results


def run_head2head(name_a, bot_a, name_b, bot_b, n_games=400, seed=42):
    """Two models battle, positions swapped every game for fairness."""
    first_a, first_b = 0, 0
    second_a, second_b = 0, 0
    reward_a, reward_b = [], []
    for g in range(n_games):
        if g % 2 == 0:
            lineup = [(name_a, bot_a), (name_b, bot_b), ('rand', None), ('rand', None)]
        else:
            lineup = [(name_b, bot_b), (name_a, bot_a), ('rand', None), ('rand', None)]
        game = Game(mode=GameMode.SINGLE, num_players=4, include_jokers=False)
        while not game.is_over:
            pid = game.current_player
            _, bot_fn = lineup[pid]
            actions = game.get_valid_actions(pid)
            if not actions:
                game.pass_turn(pid)
            else:
                if bot_fn is None:
                    chosen = random.choice(actions)
                else:
                    chosen = bot_fn(pid, actions, game)
                if chosen is None:
                    if game.can_pass(pid):
                        game.pass_turn(pid)
                    else:
                        game.play_cards(pid, random.choice(actions).cards)
                else:
                    game.play_cards(pid, chosen.cards)
        scorer = Scorer(game)
        all_rewards = scorer.compute_rewards()
        if lineup[0][0] == name_a:
            reward_a.append(all_rewards.get(0, 0.0))
        else:
            reward_a.append(all_rewards.get(1, 0.0))
        if lineup[0][0] == name_b:
            reward_b.append(all_rewards.get(0, 0.0))
        else:
            reward_b.append(all_rewards.get(1, 0.0))
        if len(game.finish_order) >= 1:
            fp = game.finish_order[0]
            if lineup[fp][0] == name_a:
                first_a += 1
            elif lineup[fp][0] == name_b:
                first_b += 1

    print(f'\n--- {name_a} vs {name_b} ({n_games} games, position-swapped) ---')
    fr_a, fr_b = first_a / n_games, first_b / n_games
    print(f'{name_a:25s}  first_finish={fr_a:.1%}  mean_reward={np.mean(reward_a):+5.1f}')
    print(f'{name_b:25s}  first_finish={fr_b:.1%}  mean_reward={np.mean(reward_b):+5.1f}')
    print(f'{name_a} {"BEATS" if fr_a > fr_b else "LOSES TO"} {name_b} (gap: {abs(fr_a-fr_b):.1%})')


if __name__ == '__main__':
    model_dir = '../models'
    selected = [
        ('016k', '510k_single_16384_steps.zip'),
        ('163k', '510k_single_163840_steps.zip'),
        ('327k', '510k_single_327680_steps.zip'),
        ('655k', '510k_single_655360_steps.zip'),
        ('999k', '510k_single_999424_steps.zip'),
        ('final', '510k_single_final.zip'),
    ]

    models = {}
    for name, fname in selected:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            models[name] = MaskablePPO.load(path)

    all_names = list(models.keys())

    # Head-to-head: early vs late
    run_head2head('016k', make_model_bot(models['016k']),
                  '999k', make_model_bot(models['999k']), n_games=400)

    run_head2head('016k', make_model_bot(models['016k']),
                  'final', make_model_bot(models['final']), n_games=400)

    run_head2head('655k', make_model_bot(models['655k']),
                  '999k', make_model_bot(models['999k']), n_games=400)

    run_head2head('327k', make_model_bot(models['327k']),
                  '999k', make_model_bot(models['999k']), n_games=400)

    # All checkpoints together (4 at a time)
    print('\n========== All 6 checkpoints, 4 per game ==========')
    print('\n--- Early 4 (016k, 163k, 327k, 655k) ---')
    lineup = [(n, make_model_bot(models[n])) for n in all_names[:4]]
    run_match(lineup, n_games=400)

    print('\n--- Late 4 (327k, 655k, 999k, final) ---')
    lineup = [(n, make_model_bot(models[n])) for n in all_names[-4:]]
    run_match(lineup, n_games=400)

    print('\n--- Mix (016k, 655k, 999k, final) ---')
    lineup = [(n, make_model_bot(models[n])) for n in ['016k', '655k', '999k', 'final']]
    run_match(lineup, n_games=400)

    # Each vs 3 random bots
    print('\n========== Each checkpoint as P0 vs 3 random bots ==========')
    for name in all_names:
        lineup = [(name, make_model_bot(models[name])),
                  ('rand', None), ('rand', None), ('rand', None)]
        run_match(lineup, n_games=200)
