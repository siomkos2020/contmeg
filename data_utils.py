import json
import torch
import os
import numpy as np
import math
import random
from torch.utils.data import Dataset, DataLoader


class HealthEventData(Dataset):
    def __init__(self,
                 data_path,
                 label_vocab,
                 is_train = True,
                 time_level = 'day',
                 multi_test=False
                 ):
        super(HealthEventData, self).__init__()
        self.is_train = is_train
        self.data_path = data_path
        self.label_vocab = label_vocab
        assert time_level in ['hour', 'day', 'week', 'month'], "time_level must in ['day', 'week', 'month']"
        self.time_level = time_level
        # Load all data from the disk (small size).
        self.examples = self._load_data()
        if multi_test:
            ids = list(range(len(self.examples)))
            random.shuffle(ids)
            ids = ids[:int(0.8*len(ids))]
            self.examples = [self.examples[i] for i in ids]
        self.converted_data = None
        self._convert_time_units(self.time_level)
        self.statistics = None
        self.compute_statistics()
        self._data_normalization()

    def __len__(self):
        return len(self.converted_data)

    def __getitem__(self, index):
        return self.converted_data[index]
    
    def _load_data(self):
        # Load data from data_path.
        assert os.path.exists(self.data_path), "Please check the data path."
        data = json.load(open(self.data_path, 'r', encoding='utf-8'))
        return data

    def compute_statistics(self):
        r"""Compute necessary statistics of the dataset including:
        mean_diag_length: avgerage length of diagnosis sequences;
        mean_medication_length: avgerage length of medication sequences;
        mean_lab_length: average length of lab tests sequences;
        mean_label_length: average length of label sequences;
        unit_time_interval_dist: distribution of unit times;
        label_last_time_dist: distribution of label happening times;
        label_support_examples: number of examples for each disease label;
        global_mean_demo: mean vector of demographic features;
        global_std_demo: std vector of demographic features; 
        """ 
        if self.converted_data is not None:
            diag_lengths, med_lengths, lab_lengths = [], [], []
            unit_time_intervals = []
            label_length = {}
            label_last_times = {}
            label_min_times = {}
            label_support_examples = {}
            demo_matrix = []
            for example in self.converted_data:
                # Count sequential features.
                diag_time, med_time, lab_time = example['features']['diag_time'],\
                                            example['features']['med_time'], \
                                            example['features']['lab_time']
                diag_lengths.append(len(set(diag_time)))
                med_lengths.append(len(set(med_time)))
                lab_lengths.append(len(set(lab_time)))
                # Count demographic features.
                demo_matrix.append(example['demographic'])
                # Count unit time intervals.
                unit_time_intervals.append(example['unit_interval'])
                # Count label length and last times.
                label_set = set()
                for label, time in example['label']:
                    if label not in label_length:
                        label_length[label] = 0
                    label_length[label] += 1
                    if label not in label_min_times:
                        label_min_times[label] = []
                    if label not in label_last_times:
                        label_last_times[label] = []
                    if label not in label_set:
                        if label not in label_support_examples:
                            label_support_examples[label] = 0
                        label_support_examples[label] += 1
                    label_last_times[label].append(time)
                    label_min_times[label].append(time)
                    label_set.add(label)
            
            mean_diag_lengths, mean_med_lengths, mean_lab_lengths = np.mean(diag_lengths),\
                                                    np.mean(med_lengths), \
                                                    np.mean(lab_lengths)
            demo_matrix = np.array(demo_matrix)
            global_mean_demo = demo_matrix.mean(0)
            global_std_demo = demo_matrix.std(0)
            mean_label_length = {k:v/len(self.examples) for k, v in label_length.items()}
            unit_time_intervals = np.array(unit_time_intervals)
            unit_time_dist = {k:np.quantile(unit_time_intervals, k) for k in np.arange(0, 1, 0.1)}
            label_last_times = {
                l: {k: np.quantile(v, k) for k in np.arange(0, 1, 0.1)}
                for l, v in label_last_times.items()
            }
            label_min_times = {l:min(v) for l, v in label_min_times.items()}
            statistics = {
                'mean_diag_lengths': mean_diag_lengths, 
                'mean_med_lengths': mean_med_lengths, 
                'mean_lab_lengths': mean_lab_lengths,
                'global_mean_demo': global_mean_demo,
                'global_std_demo': global_std_demo,
                'mean_label_length': mean_label_length,
                'unit_time_dist': unit_time_dist,
                'label_last_times': label_last_times,
                'label_support_examples': label_support_examples,
                'time_level': self.time_level,
                'total_examples': len(self.converted_data),
                'label_min_time': label_min_times
            }
            self.statistics = statistics

    def labeltime_translation(self, label_time):
        if label_time >= 0 and label_time <= 0.2:
            return 0
        elif label_time > 0.2 and label_time <= 1:
            return 1
        elif label_time > 1 and label_time <= 5:
            return 2
        else:
            return 3

    def _convert_time_units(self, unit_level = 'day'):
        # Convert time units into specified unit level (day, week, month).
        converted_data = []
        # Define the unit level for convertion.
        if unit_level == 'day':
            denominator = 24*60
        elif unit_level == 'week':
            denominator = 24*60*7
        elif unit_level == 'month':
            denominator = 24*60*30
        elif unit_level == 'hour':
            denominator = 60
        assert denominator is not None
        for example in self.examples:
            diag_time, med_time, lab_time = example['features']['diag_time'],\
                                            example['features']['med_time'], \
                                            example['features']['lab_time']
            if len(diag_time) + len(med_time) + len(lab_time) == 0:
                continue
            assert len(diag_time) + len(med_time) + len(lab_time) > 0
            unit_end_time = max(diag_time+med_time+lab_time)
            diag_time = [int(x/denominator) for x in diag_time] if len(diag_time) > 0 else []
            med_time = [int(x/denominator) for x in med_time] if len(med_time) > 0 else []
            lab_time = [int(x/denominator) for x in lab_time] if len(lab_time) > 0 else []
            unit_interval = (unit_end_time // denominator - example['features']['unit_time'])
            example['unit_interval'] = unit_interval
            example['features']['diag_time'] = diag_time
            example['features']['med_time'] = med_time
            example['features']['lab_time'] = lab_time
            example['label'] = sorted(list(set([(x[0], round((x[1]-unit_end_time)/denominator, 4)) 
                                for x in example['label']])), key=lambda x:x[1])
            # Translate continuous label to discrete label.
            translated = []
            for i, x in enumerate(example['label']):
                if i == 0:
                    translated.append((x[0], self.labeltime_translation(x[1])))
                else:
                    cur_x = x[1]-example['label'][i-1][1]
                    translated.append((x[0], self.labeltime_translation(cur_x)))
            example['label'] = translated
            
            example['features']['unit_time'] = int(example['features']['unit_time']/denominator)
            example['features']['time_level'] = self.time_level
            converted_data.append(example)
        self.converted_data = converted_data
                   
    def _data_normalization(self, global_statistics = None):
        # TODO: Normalize the dataset based on mean value and standard deviation.
        # global_statistics works when self.is_train == False
        global_mean_demo = self.statistics['global_mean_demo']
        global_std_demo = self.statistics['global_std_demo']
        for i, example in enumerate(self.converted_data):
            example['demographic'] = np.array(example['demographic'])
            example['demographic'] = (example['demographic'] - global_mean_demo) / global_std_demo
            self.converted_data[i]['demographic'] = example['demographic']



class PrivateHealthEventData(HealthEventData):
    def __init__(self, data_path, label_vocab, is_train=True, time_level='day', multi_test=False):
        self.is_train = is_train
        self.data_path = data_path
        self.label_vocab = label_vocab
        assert time_level in ['hour', 'day', 'week', 'month'], "time_level must in ['day', 'week', 'month']"
        self.time_level = time_level
        # Load all data from the disk (small size).
        self.examples = self._load_data()
        if multi_test:
            ids = list(range(len(self.examples)))
            random.shuffle(ids)
            ids = ids[:int(0.95*len(ids))]
            self.examples = [self.examples[i] for i in ids]
        self.converted_data = None
        self._convert_time_units(self.time_level)
        self.statistics = None
        self.compute_statistics()
        self._data_normalization()
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)

    def _convert_time_units(self, unit_level='day'):
        # Convert time units into specified unit level (day, week, month).
        converted_data = []
        # Define the unit level for convertion.
        input_denominator = 24*60
        denominator = 24*60*30
        assert denominator is not None
        for example in self.examples:
            example['features'] = example['feature']
            diag_time, med_time, lab_time = example['features']['diag_time'],\
                                            example['features']['med_time'], \
                                            example['features']['lab_time']
            if len(diag_time) + len(med_time) + len(lab_time) == 0:
                continue
            assert len(diag_time) + len(med_time) + len(lab_time) > 0
            unit_end_time = max(diag_time+med_time+lab_time)
            diag_time = [int(x/input_denominator) for x in diag_time] if len(diag_time) > 0 else []
            med_time = [int(x/input_denominator) for x in med_time] if len(med_time) > 0 else []
            lab_time = [int(x/input_denominator) for x in lab_time] if len(lab_time) > 0 else []
            example['unit_interval'] = -1
            example['features']['diag_time'] = diag_time
            example['features']['med_time'] = med_time
            example['features']['lab_time'] = lab_time
            example['label'] = sorted(list(set([(x[0], int(x[1]/denominator)-1) 
                                for x in example['label']])), key=lambda x:x[1])
            # Translate continuous label to discrete label.
            translated = []
            for i, x in enumerate(example['label']):
                if i == 0:
                    translated.append((x[0], self.labeltime_translation(x[1])))
                else:
                    cur_x = x[1]-example['label'][i-1][1]
                    translated.append((x[0], self.labeltime_translation(cur_x)))
            example['label'] = translated
            example['features']['unit_time'] = -1
            example['features']['time_level'] = self.time_level
            converted_data.append(example)
        self.converted_data = converted_data


def eicu_hed_collate_fn_for_class(batch_data):
    # Collect and transform batch_data into tensors.
    batch_size = len(batch_data)
    max_diag_lens = max([len(x['features']['diag_seq']) for x in batch_data])
    max_diag_nums = max([len(y) for x in batch_data for y in x['features']['diag_seq']])
    max_med_lens = max([len(x['features']['med_seq']) for x in batch_data])
    if max_med_lens == 0:
        print(max_med_lens)
    max_med_nums = max([len(y) for x in batch_data for y in x['features']['med_seq']]) if max_med_lens > 0 else 2
    max_lab_lens = max([len(x['features']['lab_seq']) for x in batch_data])
    max_lab_nums = max([len(y) for x in batch_data for y in x['features']['lab_seq']]) if max_lab_lens > 0 else 2
    max_label_lens = max([len(x['label']) for x in batch_data])

    batch_demo_features = torch.FloatTensor([x['demographic'] for x in batch_data])
    # Transform sequential features
    batch_diag_seq = torch.zeros((batch_size, max_diag_lens, max_diag_nums), dtype=torch.long)
    batch_diag_times = torch.zeros((batch_size, max_diag_lens), dtype=torch.float64)
    batch_diag_seq_mask = torch.zeros((batch_size, max_diag_lens, max_diag_nums), dtype=torch.long)
    # Final diagnosis codes.
    batch_final_diags = torch.zeros((batch_size, max_diag_nums), dtype=torch.long)
    batch_final_diags_mask = torch.zeros((batch_size, max_diag_nums), dtype=torch.long)

    batch_med_seq = torch.zeros((batch_size, max_med_lens, max_med_nums), dtype=torch.long)
    batch_med_times = torch.zeros((batch_size, max_med_lens), dtype=torch.float64)
    batch_med_seq_mask = torch.zeros((batch_size, max_med_lens, max_med_nums), dtype=torch.long)

    batch_lab_seq = torch.zeros((batch_size, max_lab_lens, max_lab_nums, 2), dtype=torch.long)
    batch_lab_times = torch.zeros((batch_size, max_lab_lens), dtype=torch.float64)
    batch_lab_seq_mask = torch.zeros((batch_size, max_lab_lens, max_lab_nums), dtype=torch.long)
    batch_lab_time_series = torch.zeros((batch_size, max_lab_lens, 150), dtype=torch.float)
    for i, example in enumerate(batch_data):
        # Fill diagnosis sequence.
        batch_diag_times[i, :len(example['features']['diag_time'])] = torch.tensor(example['features']['diag_time'])
        last_time = example['features']['diag_time'][-1] if len(example['features']['diag_time']) > 0 else 0
        batch_diag_times[i, len(example['features']['diag_time']):] = last_time
        for j in range(len(example['features']['diag_seq'])):
            batch_diag_seq[i, j, :len(example['features']['diag_seq'][j])] = torch.tensor(example['features']['diag_seq'][j])
            batch_diag_seq_mask[i, j, :len(example['features']['diag_seq'][j])] = 1
            final_len = len(example['features']['diag_seq'])-1
            if j == final_len:
                batch_final_diags[i, :] = batch_diag_seq[i, j, :]
                batch_final_diags_mask[i, :] =  batch_diag_seq_mask[i, j, :]
        # Fill medication sequence.
        batch_med_times[i, :len(example['features']['med_time'])] = torch.tensor(example['features']['med_time'])
        last_time = example['features']['med_time'][-1] if len(example['features']['med_time']) > 0 else 0
        batch_med_times[i, len(example['features']['med_time']):] = last_time
        for j in range(len(example['features']['med_seq'])):
            batch_med_seq[i, j, :len(example['features']['med_seq'][j])] = torch.tensor(example['features']['med_seq'][j])
            batch_med_seq_mask[i, j, :len(example['features']['med_seq'][j])] = 1
        # Fill lab tests sequence.
        batch_lab_times[i, :len(example['features']['lab_time'])] = torch.tensor(example['features']['lab_time'])
        last_time = example['features']['lab_time'][-1] if len(example['features']['lab_time']) > 0 else 0
        batch_lab_times[i, len(example['features']['lab_time']):] = last_time
        for j in range(len(example['features']['lab_seq'])):
            lab_types = torch.tensor([x[0] for x in example['features']['lab_seq'][j]])
            lab_values = torch.tensor([x[1] for x in example['features']['lab_seq'][j]])
            batch_lab_seq[i, j, :len(example['features']['lab_seq'][j]), 0] = lab_types
            batch_lab_seq[i, j, :len(example['features']['lab_seq'][j]), 1] = lab_values
            batch_lab_seq_mask[i, j, :len(example['features']['lab_seq'][j])] = 1
            batch_lab_time_series[i, j, lab_types] = lab_values

    # Consider the probability in the future 60 days.
    batch_label_seq = torch.zeros((batch_size, max_label_lens+1), dtype=torch.long)
    batch_label_times = torch.zeros((batch_size, max_label_lens+1), dtype=torch.long)   # discrete times.
    batch_label_mask = torch.zeros((batch_size, max_label_lens+1), dtype=torch.float)
    for i, example in enumerate(batch_data):
        label_seq = example['label']
        batch_label_mask[i, :len(label_seq)-1] = 1
        batch_label_seq[i, len(label_seq):] = label_seq[-1][0]
        batch_label_times[i, len(label_seq):] = label_seq[-1][1] if len(label_seq) > 0 else 0.
        for j, (label, label_time) in enumerate(label_seq):
            batch_label_seq[i, j] = int(label)
            batch_label_times[i, j] = int(label_time)

    # Our model predicts the first time and follow-up time intervals.
    return {
        'demo': batch_demo_features,
        'diag_seq': batch_diag_seq,
        'diag_times': batch_diag_times,
        'diag_mask': batch_diag_seq_mask,
        'final_diag': batch_final_diags,
        'final_diag_mask': batch_final_diags_mask,
        'med_seq': batch_med_seq,
        'med_times': batch_med_times,
        'med_mask': batch_med_seq_mask,
        'lab_seq': batch_lab_seq,
        'lab_times': batch_lab_times,
        'lab_mask': batch_lab_seq_mask,
        'lab_ts': batch_lab_time_series,
        'label_seq': batch_label_seq,
        'label_times': batch_label_times,
        'label_mask': batch_label_mask,
    }


def prepare_HED_dataloader(dataset, batch_size, num_workers=4):
    # Build a dataloader for dataset.
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=eicu_hed_collate_fn_for_class,
        num_workers=num_workers
    )

