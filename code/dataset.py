from itertools import product

from random import choice
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            same_prim_sample=False
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        self.same_prim_sample = same_prim_sample

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

        if self.phase == 'train' and self.same_prim_sample:
            self.same_attr_diff_obj_dict = {pair: list() for pair in self.train_pairs}
            self.same_obj_diff_attr_dict = {pair: list() for pair in self.train_pairs}
            for i_sample, sample in enumerate(self.train_data):
                sample_attr, sample_obj = sample[1], sample[2]
                for pair_key in self.same_attr_diff_obj_dict.keys():
                    if (pair_key[1] == sample_obj) and (pair_key[0] != sample_attr):
                        self.same_obj_diff_attr_dict[pair_key].append(i_sample)
                    elif (pair_key[1] != sample_obj) and (pair_key[0] == sample_attr):
                        self.same_attr_diff_obj_dict[pair_key].append(i_sample)


    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        if self.phase == 'train' and self.same_prim_sample:
            [same_attr_image, same_attr, diff_obj], same_attr_mask = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr')
            [same_obj_image, diff_attr, same_obj], same_obj_mask = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj')
            same_attr_img = self.transform(self.loader(same_attr_image))
            same_obj_img = self.transform(self.loader(same_obj_image))
            data += [same_attr_img, self.attr2idx[same_attr], self.obj2idx[diff_obj], 
                     self.train_pair_to_idx[(same_attr, diff_obj)], same_attr_mask,
                     same_obj_img, self.attr2idx[diff_attr], self.obj2idx[same_obj], 
                     self.train_pair_to_idx[(diff_attr, same_obj)], same_obj_mask]

        return data

    def same_A_diff_B(self, label_A, label_B, phase='attr'):
        if phase=='attr':
            candidate_list = self.same_attr_diff_obj_dict[(label_A, label_B)]
        else:
            candidate_list = self.same_obj_diff_attr_dict[(label_B, label_A)]
        if len(candidate_list) != 0:
            idx = choice(candidate_list)
            mask = 1
        else:
            idx = choice(list(range(len(self.data))))
            mask = 0
        return self.data[idx], mask

    def __len__(self):
        return len(self.data)


class ActionAdverbDataset(Dataset):
    """Dataset for action-adverb classification using pre-extracted I3D features with variable temporal lengths"""
    def __init__(self, data_dir, features_dir, split='train'):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.split = split

        self.data = pd.read_csv(os.path.join(data_dir, f'{split}.csv'))

        # Build vocab from both train and test splits to keep indices consistent
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        all_df = pd.concat([train_df, test_df])
        self.all_df = all_df

        self.adverbs = sorted(all_df['clustered_adverb'].unique().tolist())
        self.actions = sorted(all_df['clustered_action'].unique().tolist())
        self.adverb2idx = {adverb: idx for idx, adverb in enumerate(self.adverbs)}
        self.action2idx = {action: idx for idx, action in enumerate(self.actions)}

        # Create all possible action-adverb pairs
        self.all_pairs = [(a, av) for a in self.actions for av in self.adverbs]
        self.pair2idx = {pair: i for i, pair in enumerate(self.all_pairs)}

        # Track pairs that appear in training data (for seen/unseen split)
        train_pairs_set = set(zip(train_df['clustered_action'], train_df['clustered_adverb']))
        self.train_pairs = sorted(list(train_pairs_set))

        # Track pairs in current split
        current_pairs_set = set(zip(self.data['clustered_action'], self.data['clustered_adverb']))
        self.phase = split  # For compatibility with CompositionDataset
        if split == 'test':
            self.test_pairs = sorted(list(current_pairs_set))
            # Determine unseen pairs (in test but not in train)
            unseen_pairs_set = current_pairs_set - train_pairs_set
            self.val_pairs = sorted(list(unseen_pairs_set))  # For compatibility
        else:
            self.test_pairs = []
            self.val_pairs = []

        # Compute max temporal length and feature dimensions
        self.temporal_len = self._compute_max_temporal_len()
        print(f'ActionAdverbDataset [{split}]:')
        print(f'  - Actions: {len(self.actions)}')
        print(f'  - Adverbs: {len(self.adverbs)}')
        print(f'  - Pairs: {len(self.all_pairs)}')
        print(f'  - Train pairs: {len(self.train_pairs)}')
        if split == 'test':
            print(f'  - Test pairs: {len(self.test_pairs)} (Seen: {len(train_pairs_set & current_pairs_set)}, Unseen: {len(unseen_pairs_set)})')
        print(f'  - Samples: {len(self.data)}')
        print(f'  - Max temporal length: {self.temporal_len}')
        print(f'  - Flow dim: {self.flow_dim}, RGB dim: {self.rgb_dim}')

    def _load_features(self, clip_id: str, modality: str) -> np.ndarray:
        """Load features for a given clip and modality"""
        path = os.path.join(self.features_dir, f'{clip_id}_{modality}.npz')
        with np.load(path, allow_pickle=True) as f:
            key = next((k for k in ('features', 'arr_0') if k in f), f.files[0])
            features = np.array(f[key], dtype=np.float32)
        # Collapse spatial dims (T, H, W, C) -> (T, C) if present
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1, features.shape[-1]).mean(axis=1)
        return features

    def _compute_max_temporal_len(self) -> int:
        """Compute maximum temporal length across all videos"""
        max_len = 0
        first_sample_processed = False

        for clip_id in self.all_df['clip_id']:
            rgb = self._load_features(clip_id, 'rgb')
            flow = self._load_features(clip_id, 'flow')

            # Store feature dimensions from first sample
            if not first_sample_processed:
                self.rgb_dim = rgb.shape[-1]
                self.flow_dim = flow.shape[-1]
                first_sample_processed = True

            # Use min across modalities, consistent with __getitem__
            max_len = max(max_len, min(rgb.shape[0], flow.shape[0]))

        return max_len

    def _pad_to_fixed_length(self, features: np.ndarray) -> np.ndarray:
        """Pad features to fixed temporal length"""
        T, C = features.shape
        if T == self.temporal_len:
            return features
        padded = np.zeros((self.temporal_len, C), dtype=np.float32)
        padded[:T] = features
        return padded

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        clip_id = row['clip_id']
        action = row['clustered_action']
        adverb = row['clustered_adverb']

        rgb = self._load_features(clip_id, 'rgb')
        flow = self._load_features(clip_id, 'flow')

        # Align modalities before padding
        min_T = min(rgb.shape[0], flow.shape[0])
        rgb, flow = rgb[:min_T], flow[:min_T]

        action_idx = self.action2idx[action]
        adverb_idx = self.adverb2idx[adverb]
        pair_idx = self.pair2idx[(action, adverb)]

        return {
            'clip_id': clip_id,
            'rgb_features': torch.from_numpy(self._pad_to_fixed_length(rgb)),   # (L, C)
            'flow_features': torch.from_numpy(self._pad_to_fixed_length(flow)),  # (L, C)
            'action': action,
            'adverb': adverb,
            'action_idx': action_idx,
            'adverb_idx': adverb_idx,
            'pair_idx': pair_idx,
        }
