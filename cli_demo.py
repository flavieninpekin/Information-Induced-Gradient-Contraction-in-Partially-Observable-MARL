"""
510K CLI Demo — 人机对战，验证规则是否正确实现
"""
import sys, re, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '510k-env'))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from env.env_510k import FiveTenKEnv
from env.card import Card, Rank, Suit
from env.patterns import PatternType
from env.scorer import Scorer
import numpy as np

SUIT_SYMBOLS = ['♠', '♥', '♣', '♦']
RANK_NAMES = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
              10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: '2',
              16: '小王', 17: '大王'}

def c_str(c):
    if c.is_joker:
        return RANK_NAMES[c.rank.value]
    return RANK_NAMES[c.rank.value] + SUIT_SYMBOLS[c.suit.value]

def hand_str(cards):
    return ' '.join(c_str(c) for c in sorted(cards, key=lambda x: (x.rank, x.suit)))

PATTERN_NAMES = {
    PatternType.SINGLE: '单张', PatternType.PAIR: '对子',
    PatternType.THREE: '三条', PatternType.THREE_ONE: '三带一',
    PatternType.THREE_PAIR: '三带一对', PatternType.STRAIGHT: '顺子',
    PatternType.CONSECUTIVE_PAIRS: '连对', PatternType.AIRPLANE_NONE: '飞机',
    PatternType.AIRPLANE_SINGLE: '飞机带单', PatternType.AIRPLANE_PAIR: '飞机带对',
    PatternType.BOMB: '炸弹', PatternType.NONSUIT_510K: '异花510K',
    PatternType.SUIT_510K: '同花510K', PatternType.RED_A_SINGLE: '红A单张',
    PatternType.RED_A_PAIR: '红A对',
}

def match_cards(hand, tokens):
    suit_map = {'s': Suit.SPADE, 'h': Suit.HEART, 'c': Suit.CLUB, 'd': Suit.DIAMOND,
                '♠': Suit.SPADE, '♥': Suit.HEART, '♣': Suit.CLUB, '♦': Suit.DIAMOND}
    rank_map = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                '10': 10, 'j': 11, 'J': 11, 'q': 12, 'Q': 12, 'k': 13, 'K': 13,
                'a': 14, 'A': 14, '2': 15}
    result = []
    for t in tokens:
        t = t.strip()
        if len(t) < 2:
            return None
        if t.startswith('10'):
            rank_str, suit_str = '10', t[2:]
        else:
            rank_str, suit_str = t[0], t[1:]
        rank = rank_map.get(rank_str)
        suit = suit_map.get(suit_str)
        if rank is None or suit is None:
            return None
        card = Card(Rank(rank), suit)
        if card not in hand:
            print(f'  ❌ 手牌中没有 {c_str(card)}')
            return None
        result.append(card)
    return result


def cli_demo(mode='single'):
    env = FiveTenKEnv(mode=mode, num_players=(3 if mode == '3p' else 4))
    obs, info = env.reset()
    human_id = 0
    last_log_idx = 0  # track which log entries we've shown

    # Print initial deal
    game = env.game
    log = game.actions_log
    deal = log[0]
    np_str = '18张' if game.num_players == 3 else '13张'
    print('\n' + '=' * 60)
    print(f'模式: {mode}  先出: P{deal["starter"]}（持3♦）  每人{np_str}')
    print(f'你的手牌: {hand_str(game.players[human_id].hand)}')
    last_log_idx = 1

    while True:
        game = env.game

        # Print new log entries since last human turn
        log = game.actions_log
        while last_log_idx < len(log):
            entry = log[last_log_idx]
            last_log_idx += 1

            if entry['action'] == 'play':
                pid = entry['player']
                pt = entry['pattern_type']
                cards = ' '.join(c_str(c) for c in entry['cards'])
                fin = ' ★ 出完！' if entry['finished'] else ''
                lab = PATTERN_NAMES.get(PatternType[pt], pt) if pt in [p.name for p in PatternType] else pt
                print(f'  P{pid} [{lab}] {cards}{fin}')

            elif entry['action'] == 'pass':
                print(f'  P{entry["player"]}  过')

            elif entry['action'] == 'trick_end':
                winner = entry['winner']
                sc = entry.get('score', 0)
                if sc > 0:
                    print(f'  → P{winner} 赢得此轮，得 {sc} 分')
                else:
                    print(f'  → P{winner} 赢得此轮')

        if game.is_over:
            break

        if game.current_player != human_id:
            continue  # bots will auto-play in env.step

        # Human turn — show hand & legal actions
        print()
        score = game.player_510k_scores[human_id]
        print(f'你的手牌 ({len(game.players[human_id].hand)} 张)  [510K分: {score}]')
        print(f'  {hand_str(game.players[human_id].hand)}')

        # Show opponent status (finished only, no hand size)
        n = env.game.num_players
        finished = [i for i in range(n) if i != human_id and game.players[i].finished]
        if finished:
            print(f'  已出完: P{", P".join(str(p) for p in finished)}')

        valid = game.get_valid_actions(human_id)
        if not valid:
            print('无法出牌，自动过牌')
            action = 0
        else:
            action = None
            while action is None:
                print(f'\n合法动作 ({len(valid)} 种):')
                for i, p in enumerate(valid):
                    cards = ' '.join(c_str(c) for c in p.cards)
                    print(f'  [{i+1}] [{PATTERN_NAMES.get(p.type, p.type.name)}] {cards}')
                if game.can_pass(human_id):
                    print(f'  [0] 过牌')

                inp = input('\n选择 (编号) 或出牌 (如 3s 4h 5c): ').strip()
                if not inp:
                    continue

                try:
                    idx = int(inp)
                    if idx == 0:
                        if game.can_pass(human_id):
                            action = 0
                        else:
                            print('当前不能过牌')
                    elif 1 <= idx <= len(valid):
                        action = idx
                    else:
                        print(f'输入 0-{len(valid)}')
                except ValueError:
                    tokens = inp.replace(',', ' ').split()
                    cards_list = match_cards(game.players[human_id].hand, tokens)
                    if cards_list is None:
                        print('牌面格式: 3s=3♠, 10h=10♥, kd=K♦, ac=A♣')
                        continue
                    found = [i for i, v in enumerate(valid) if sorted(v.cards) == sorted(cards_list)]
                    if not found:
                        print('这组牌不合法或不足以压制上家')
                        continue
                    action = found[0] + 1

        obs, reward, done, truncated, info = env.step(action)

    # Game over
    print()
    print('=' * 60)
    print('游戏结束！')
    print(f'出完顺序: P{", P".join(str(p) for p in game.finish_order)}')

    scorer = Scorer(game)
    rewards = scorer.compute_rewards()
    print()
    for i in range(game.num_players):
        status = '★ 出完' if game.players[i].finished else f'{len(game.players[i].hand)}张'
        r = rewards.get(i, 0)
        s = game.player_510k_scores[i]
        sign = '+' if r >= 0 else ''
        print(f'  P{i}: {status}  510K分: {s}  总得分: {sign}{r:.0f}')

    if rewards.get(human_id, 0) > 0:
        print('\n🎉 你赢了！')
    else:
        print('\n你输了')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='510K CLI Demo')
    parser.add_argument('--mode', default='single', choices=['single', 'static', 'dynamic', '3p'])
    args = parser.parse_args()
    cli_demo(args.mode)
