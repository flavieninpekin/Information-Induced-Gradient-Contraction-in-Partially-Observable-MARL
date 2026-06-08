"""
打印完整对局历史，验证游戏逻辑
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
import numpy as np
from env.env_510k import FiveTenKEnv
from env.card import Rank, Suit


def card_str(c):
    if c.rank == Rank.SMALL_JOKER: return '小王'
    if c.rank == Rank.BIG_JOKER: return '大王'
    suit_map = {0: '♠', 1: '♥', 2: '♣', 3: '♦'}
    return f'{c.rank}{suit_map[c.suit.value]}'


def cards_str(cards):
    return ' '.join(card_str(c) for c in cards)


def format_hand(cards):
    return cards_str(sorted(cards, key=lambda c: (c.rank, c.suit.value)))


def print_game_log(game):
    log = game.actions_log
    if not log:
        return

    deal = log[0]
    print('=' * 72)
    print(f'模式: {deal["mode"]}')
    print(f'先出: P{deal["starter"]}（持3♦）')
    if deal.get('red_a_team'):
        print(f'红A队: P{", P".join(str(p) for p in deal["red_a_team"])}')
    print()

    for i, hand in enumerate(deal['hands']):
        print(f'  P{i} 初始手牌 ({len(hand)}张): {format_hand(hand)}')
    print()

    trick_end_cards = []
    step = 0
    for entry in log[1:]:
        if entry['action'] == 'trick_end':
            trick_end_cards = entry.get('cards', [])
            continue

        if entry['action'] == 'play':
            hand_size = entry['hand_size']
            finished = entry['finished']
            ct = entry.get('cards', [])
            line = f'  P{entry["player"]} ▶ {entry["pattern_type"]:16s}'
            if ct:
                line += cards_str(ct)
            else:
                line += '-'

            # Show remaining hand for agent only to limit output
            line += f'  ({hand_size}张)'
            if finished:
                line += ' ★ 出完！'
            print(line)

        elif entry['action'] == 'pass':
            print(f'  P{entry["player"]}   过')

        step += 1

    print()
    print(f'出完顺序: {game.finish_order} (P{", P".join(str(p) for p in game.finish_order)})')
    print('=' * 72)


def run_and_show(mode='single', seed=42):
    env = FiveTenKEnv(mode=mode)
    obs, info = env.reset(seed=seed)
    while True:
        mask = info['action_mask']
        valid = np.where(mask)[0]
        action = int(np.random.choice(valid))
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    print_game_log(env.game)
    return env


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'static', 'dynamic'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--games', type=int, default=3)
    args = parser.parse_args()

    for g in range(args.games):
        env = run_and_show(args.mode, args.seed + g)
        print(f'\nReward P0 (agent): {env._compute_reward():.0f}\n')
