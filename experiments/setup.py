"""
B200 setup script for AAAI-27 experiments.

Run once on the remote machine before starting experiments.

Usage:
    pip install -r experiments/requirements.txt
    python experiments/setup.py
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATCHES = {
    # Overcooked has numpy 2.x incompatibilities
    'overcooked_env.py': [
        ('np.Inf', 'np.inf'),
    ],
    'agents/agent.py': [
        ('np.Inf', 'np.inf'),
    ],
    'planning/planners.py': [
        ('np.Inf', 'np.inf'),
    ],
    'actions.py': [
        ('return np.random.choice(Action.ALL_ACTIONS, p=action_probs)',
         'return Action.INDEX_TO_ACTION[np.random.choice(len(action_probs), p=action_probs)]'),
    ],
    'overcooked_mdp.py': [
        ('def has_object(self, pos):\n        return pos in self.objects',
         'def has_object(self, pos):\n        return tuple(pos) in self.objects'),
        ('def get_object(self, pos):\n        assert self.has_object(pos)\n        return self.objects[pos]',
         'def get_object(self, pos):\n        pos = tuple(pos)\n        assert self.has_object(pos)\n        return self.objects[pos]'),
        ('def add_object(self, obj, pos=None):\n        if pos is None:\n            pos = obj.position\n\n        assert not self.has_object(pos)',
         'def add_object(self, obj, pos=None):\n        if pos is None:\n            pos = obj.position\n\n        pos = tuple(pos)\n        assert not self.has_object(pos)'),
        ('def remove_object(self, pos):\n        assert self.has_object(pos)\n        obj = self.objects[pos]\n        del self.objects[pos]\n        return obj',
         'def remove_object(self, pos):\n        pos = tuple(pos)\n        assert self.has_object(pos)\n        obj = self.objects[pos]\n        del self.objects[pos]\n        return obj'),
    ],
}

PATTERNS_PATCH = {
    '510k-env/env/patterns.py': [
        ("sorted(self.cards)",
         "sorted(self.cards, key=lambda c: (c.rank.value, c.suit.value))"),
    ],
}


def find_overcooked_path():
    """Locate the installed overcooked_ai_py package."""
    try:
        import overcooked_ai_py
        return os.path.dirname(overcooked_ai_py.__file__)
    except ImportError:
        print('ERROR: overcooked_ai_py not installed. Run: pip install overcooked_ai')
        sys.exit(1)


def apply_patches(pkg_path, patches):
    for filename, replacements in patches.items():
        filepath = os.path.join(pkg_path, 'mdp', filename) if 'mdp' not in filepath else filepath
        if not os.path.exists(filepath):
            for sub in ['mdp', 'planning', 'agents', '']:
                fp = os.path.join(pkg_path, sub, filename)
                if os.path.exists(fp):
                    filepath = fp
                    break

        if not os.path.exists(filepath):
            print(f'  WARN: {filename} not found')
            continue

        with open(filepath) as f:
            content = f.read()

        modified = False
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                modified = True
                print(f'  PATCHED: {os.path.basename(filepath)} ({old[:50]}...)')

        if modified:
            with open(filepath, 'w') as f:
                f.write(content)


def apply_project_patches():
    """Apply patches to project files."""
    for filepath, replacements in PATTERNS_PATCH.items():
        fp = os.path.join(ROOT, filepath)
        if not os.path.exists(fp):
            print(f'  WARN: {fp} not found')
            continue
        with open(fp) as f:
            content = f.read()
        modified = False
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                modified = True
                print(f'  PATCHED: {filepath}')
        if modified:
            with open(fp, 'w') as f:
                f.write(content)


if __name__ == '__main__':
    print('Applying patches for numpy 2.x compatibility...')

    # Patch overcooked_ai_py
    pkg_path = find_overcooked_path()
    print(f'Overcooked package at: {pkg_path}')
    apply_patches(pkg_path, PATCHES)

    # Patch project files
    apply_project_patches()

    print('\nSetup complete. Run: python experiments/run_all.py --all')
