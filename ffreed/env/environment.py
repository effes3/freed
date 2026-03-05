from rdkit import Chem
from operator import methodcaller, itemgetter
from functools import partial

import numpy as np

from ffreed.env.state import State
from ffreed.utils import dmap, lmap, dsuf
from ffreed.env.utils import connect_mols


class Environment(object):
    def __init__(self, *, atom_vocab, bond_vocab, frag_vocab, timelimit=4,
                 rewards, starting_smile='c1([*:1])c([*:2])ccc([*:3])c1', 
                 action_size=40, fragmentation='crem', **kwargs):
        self.atom_vocab = atom_vocab
        self.frag_vocab = frag_vocab
        self.bond_vocab = bond_vocab

        assert fragmentation in ['crem', 'brics']
        self.fragmentation = fragmentation
        if fragmentation == 'crem':
            attach_vocab = ['*']
        elif fragmentation == 'brics':
            attach_vocab = [f"[{i}*]" for i in range(31)]
 
        self.attach_vocab = attach_vocab
        self.num_att_types = len(attach_vocab)
        # atom representation have 18 meta features
        self.atom_dim = len(atom_vocab) + len(attach_vocab) + 18
        self.bond_dim = len(self.bond_vocab)
        self.action_size = action_size
        self.starting_smile = starting_smile
        self.state_args = {
            'fragmentation': fragmentation,
            'atom_dim': self.atom_dim,
            'bond_dim': self.bond_dim,
            'atom_vocab': atom_vocab,
            'bond_vocab': bond_vocab,
            'attach_vocab': attach_vocab
        }
        self.num_steps = 0
        self.state = State(starting_smile, self.num_steps, **self.state_args)
        self.rewards = rewards
        self.timelimit = timelimit
        self.fragments = []
        print(f"Loading and filtering {len(self.frag_vocab)} fragments...")
        for i, frag in enumerate(self.frag_vocab):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(self.frag_vocab)} fragments...")
            try:
                # Fast check: Parse molecule and check for bridging attachments first
                mol = Chem.MolFromSmiles(frag)
                if mol is None:
                    continue
                
                is_valid = True
                sticker_count = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == "*" or atom.GetAtomicNum() == 0:
                        sticker_count += 1
                        if len(atom.GetNeighbors()) != 1:
                            is_valid = False
                            break
                
                if is_valid and sticker_count > 0:
                    # Only create the expensive State object for valid fragments
                    state = State(frag, 0, **self.state_args)
                    self.fragments.append(state)
            except Exception as e:
                pass # Silent skip for malformed SMILES in large libraries
        
        print(f"Finished loading. {len(self.fragments)} valid fragments found.")
        
        from collections import Counter
        type_counts = Counter()
        for frag in self.fragments:
            for t in frag.attachment_types:
                type_counts[t] += 1
        
        print("Fragment Library Distribution:")
        for t, count in sorted(type_counts.items()):
            label = attach_vocab[t] if t < len(attach_vocab) else f"Type {t}"
            print(f"  {label}: {count} fragments")

        num_att = [len(frag.attachments) for frag in self.fragments]
        S, T = len(self.state.attachments), timelimit
        N, M = len(self.fragments), max(num_att) if num_att else 0
        self.actions_dim = (S + T * (M - 1), N, M)

    def reward_batch(self, smiles):
        objectives = dmap(methodcaller('__call__', smiles), self.rewards)
        rewards = dsuf('Reward', dmap(partial(lmap, itemgetter(0)), objectives))
        properties = dsuf('Property', dmap(partial(lmap, itemgetter(1)), objectives))
        rewards['Reward'] = np.sum(list(rewards.values()), axis=0).tolist()
        return {**rewards, **properties}

    def step(self, action):
        self.attach_fragment(action)
        self.num_steps += 1
        terminated = not self.state.attachments
        truncated = self.num_steps >= self.timelimit
        # reward calculated only for terminal states in "reward_batch" call
        reward = 0.
        state = self.state
        info = dict()
        return state, reward, terminated, truncated, info

    def reset(self, starting_smile=None):
        self.num_steps = 0
        self.state = State(self.starting_smile, self.num_steps, **self.state_args)
        return self.state



    def attach_fragment(self, action):
        a1, a2, a3 = action
        mol = self.state.molecule
        frag_state = self.fragments[a2]
        frag = frag_state.molecule
        mol_attachments = self.state.attachment_ids
        mol_attachment = mol.GetAtomWithIdx(mol_attachments[a1])
        frag_attachments = frag_state.attachment_ids
        frag_attachment = frag.GetAtomWithIdx(frag_attachments[a3])
        new_mol = connect_mols(mol, frag, mol_attachment, frag_attachment)
        self.state = State(Chem.MolToSmiles(new_mol), self.num_steps + 1, **self.state_args)
