"""
Custom Overcooked partner agents with differentiated roles.

Roles:
  chef    - cooks onions into soup; never delivers
  waiter  - delivers finished soups; never cooks
  chaos   - random movement + occasional item interference
"""
import numpy as np
import itertools
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import Agent


class ChefAgent(Agent):
    """
    Specialized cook: picks up onions, puts them in pots, starts cooking.
    NEVER picks up plates or delivers soups.
    Agent should complement by delivering.
    """

    def __init__(self, mlam):
        super().__init__()
        self.mlam = mlam
        self.mdp = mlam.mdp

    def action(self, state):
        player = state.players[self.agent_index]
        am = self.mlam
        mp = self.mlam.motion_planner
        pot_states = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(
            state, list(self.mdp.terrain_pos_dict.get('X', []))
        )

        if not player.has_object():
            # Check if any pot needs onions
            empty_or_partial = pot_states.get('empty', []) + \
                               pot_states.get('1_items', []) + \
                               pot_states.get('2_items', [])
            if empty_or_partial:
                # Go get an onion and put it in
                onion_goals = am.pickup_onion_actions(counter_objects)
                goals = [g for g in onion_goals if mp.is_valid_motion_start_goal_pair(
                    player.pos_and_or, g)]
                if goals:
                    return self._move_to_goal(player, goals)
            # If pots are cooking, wait near the onion supply
            onion_positions = self.mdp.terrain_pos_dict.get('O', [])
            if onion_positions:
                return self._move_to_pos(player, onion_positions[0])
        else:
            obj = player.get_object()
            if obj.name in ('onion', 'tomato'):
                # Put it in a pot
                put_actions = am.put_onion_in_pot_actions(pot_states)
                goals = [g for g in put_actions if mp.is_valid_motion_start_goal_pair(
                    player.pos_and_or, g)]
                if goals:
                    return self._move_to_goal(player, goals)

        return self._random_move(player, state)

    def _move_to_goal(self, player, goals):
        mp = self.mlam.motion_planner
        best_goal, best_cost = None, float('inf')
        for g in goals:
            _, _, cost = mp.get_plan(player.pos_and_or, g)
            if cost < best_cost:
                best_cost, best_goal = cost, g
        if best_goal:
            plan, _, _ = mp.get_plan(player.pos_and_or, best_goal)
            if plan:
                return plan[0], {"action_probs": Agent.a_probs_from_action(plan[0])}
        return self._random_action()

    def _move_to_pos(self, player, target_pos):
        mp = self.mlam.motion_planner
        for d in Direction.ALL_DIRECTIONS:
            goal = (target_pos, d)
            if mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal):
                plan, _, _ = mp.get_plan(player.pos_and_or, goal)
                if plan:
                    return plan[0], {"action_probs": Agent.a_probs_from_action(plan[0])}
        return self._random_action()

    def _random_move(self, player, state):
        return self._random_action()

    def _random_action(self):
        legal = [a for a in Action.MOTION_ACTIONS if a != Action.STAY]
        a = legal[np.random.randint(len(legal))]
        return a, {"action_probs": Agent.a_probs_from_action(a)}


class WaiterAgent(Agent):
    """
    Specialized deliverer: picks up plates, gets finished soups, delivers.
    NEVER picks up onions or puts them in pots.
    Agent should complement by cooking.
    """

    def __init__(self, mlam):
        super().__init__()
        self.mlam = mlam
        self.mdp = mlam.mdp

    def action(self, state):
        player = state.players[self.agent_index]
        am = self.mlam
        mp = self.mlam.motion_planner
        pot_states = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(
            state, list(self.mdp.terrain_pos_dict.get('X', []))
        )

        if not player.has_object():
            ready = pot_states.get('ready', [])
            if ready:
                # Get a plate, then get the soup
                dish_goals = am.pickup_dish_actions(counter_objects)
                goals = [g for g in dish_goals if mp.is_valid_motion_start_goal_pair(
                    player.pos_and_or, g)]
                if goals:
                    return self._move_to_goal(player, goals)
            else:
                # Wait near serving area
                serving = self.mdp.terrain_pos_dict.get('S', [])
                if serving:
                    return self._move_to_pos(player, serving[0])
        else:
            obj = player.get_object()
            if obj.name == 'dish':
                # Pick up finished soup
                soup_goals = am.pickup_soup_with_dish_actions(pot_states)
                goals = [g for g in soup_goals if mp.is_valid_motion_start_goal_pair(
                    player.pos_and_or, g)]
                if goals:
                    return self._move_to_goal(player, goals)
            elif obj.name == 'soup':
                # Deliver
                deliver_goals = am.deliver_soup_actions()
                goals = [g for g in deliver_goals if mp.is_valid_motion_start_goal_pair(
                    player.pos_and_or, g)]
                if goals:
                    return self._move_to_goal(player, goals)

        return self._random_action()

    def _move_to_goal(self, player, goals):
        mp = self.mlam.motion_planner
        best_goal, best_cost = None, float('inf')
        for g in goals:
            _, _, cost = mp.get_plan(player.pos_and_or, g)
            if cost < best_cost:
                best_cost, best_goal = cost, g
        if best_goal:
            plan, _, _ = mp.get_plan(player.pos_and_or, best_goal)
            if plan:
                return plan[0], {"action_probs": Agent.a_probs_from_action(plan[0])}
        return self._random_action()

    def _move_to_pos(self, player, target_pos):
        mp = self.mlam.motion_planner
        for d in Direction.ALL_DIRECTIONS:
            goal = (target_pos, d)
            if mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal):
                plan, _, _ = mp.get_plan(player.pos_and_or, goal)
                if plan:
                    return plan[0], {"action_probs": Agent.a_probs_from_action(plan[0])}
        return self._random_action()

    def _random_action(self):
        legal = [a for a in Action.MOTION_ACTIONS if a != Action.STAY]
        a = legal[np.random.randint(len(legal))]
        return a, {"action_probs": Agent.a_probs_from_action(a)}


class ChaosAgent(Agent):
    """
    Disruptive partner: moves randomly, occasionally interacts with objects
    (picks up items and drops them, blocks paths).
    Agent must work around this partner.
    """

    def __init__(self):
        super().__init__()

    def action(self, state):
        p = np.random.random()
        if p < 0.3:
            # Random interact
            return Action.INTERACT, {"action_probs": Agent.a_probs_from_action(Action.INTERACT)}
        elif p < 0.7:
            # Random motion (not stay)
            legal = [a for a in Action.MOTION_ACTIONS if a != Action.STAY]
            a = legal[np.random.randint(len(legal))]
            return a, {"action_probs": Agent.a_probs_from_action(a)}
        else:
            # Stay (blocking)
            return Action.STAY, {"action_probs": Agent.a_probs_from_action(Action.STAY)}
