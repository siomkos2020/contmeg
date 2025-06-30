
import collections
import copy
import os
import json
import random
from tqdm import tqdm
import asyncio
import re
import sys
sys.path.append(".")
from database import EICUConfig


class EICUMedCodeMapper:
    r"""A class which merges a complex medication string to simple format."""
    def __init__(self, ) -> None:
        # Extract letters before a string.
        self.letter_pattern = r'^([a-zA-Z]+(?:\s+[a-zA-Z]+)*)'
        self.raw2merged = dict()

    def merge(self, med_code: str):
        matched = re.match(self.letter_pattern, med_code)
        if matched:
            matched = matched.group(1)
            if len(matched) < 2: return None
            if "in d" in matched:
                matched = matched.split("in d")[0]
            if med_code not in self.raw2merged:
                self.raw2merged[med_code] = matched
            return matched
        else:
            return None
    
    def map(self, med_code: str):
        return self.raw2merged.get(med_code, None)


class VocabBuilder:
    def __init__(self, db_config: EICUConfig) -> None:
        # Label names defined by user (ICD-9).
        self.db_config = db_config
        self.vocabulary = None
        self.freq_count = None
        self.code_merger = EICUMedCodeMapper()
    
    def save(self, target_dir):
        voc_path = os.path.join(target_dir, 'voc.json')
        with open(voc_path, 'w', encoding='utf-8') as fp:
            json.dump(self.vocabulary, fp)

    def build_voc_from_data(self, json_data, verbose=True):
        voc_names = ['diag_voc', 'med_voc', 'lab_voc', 'demographic', 'label']
        vocabulary = dict(zip(voc_names, [dict() for _ in voc_names]))
        # Build the label voc.
        vocabulary['label'] = {name:i for i, name in enumerate(self.db_config.label_names)}
        freq_count = dict(zip(voc_names, [dict() for _ in voc_names]))
        # Build feature voc.
        print("Building vocabulary...")
        for host_id in tqdm(json_data):
            for demo in json_data[host_id]['demographic']:
                if demo != 'dischargeweight' and demo not in vocabulary['demographic']:
                    vocabulary['demographic'][demo] = len(vocabulary['demographic']) + 1
            for unit_id in json_data[host_id]['unit_records']:
                unit_record = json_data[host_id]['unit_records'][unit_id]
                for code in unit_record['diag_seq']:
                    proc_code = self._select_icd_level(code)
                    self._add_new_code_to_voc(proc_code, freq_count['diag_voc'])

                for code in unit_record['med_seq']:
                    proc_code = code[0]
                    self._add_new_code_to_voc(proc_code, freq_count['med_voc'])
                
                for code in unit_record['lab_test_seq']:
                    proc_code = code[0]
                    self._add_new_code_to_voc(proc_code, freq_count['lab_voc'])
        self.freq_count = freq_count
        self.vocabulary = vocabulary
        self._feature_selection_by_freq(100)
        self._merge_med_codes()
    
    def _merge_med_codes(self):
        r"""Merge complex and redundant medication codes."""
        merged_med_voc = {}
        for code in self.vocabulary['med_voc']:
            merged_code = self.code_merger.merge(code.lower())
            if merged_code and merged_code not in merged_med_voc:
                merged_med_voc[merged_code] = len(merged_med_voc)
        self.vocabulary['med_voc'] = merged_med_voc

    
    def _feature_selection_by_freq(self, freq_thre = 100):
        selected_feat = copy.deepcopy(self.freq_count)
        for k in selected_feat:
            if k not in ['label', 'demographic']:
                selected_feat[k] = sorted(selected_feat[k].items(),
                                          key=lambda x: x[1], reverse=True)
                self.vocabulary[k] = {kx:i for i, (kx, kv) in enumerate(selected_feat[k])
                                       if kv > freq_thre}
                
    def _add_new_code_to_voc(self, code, voc):
        if code not in voc:
            voc[code] = 0
        voc[code] += 1
    
    def _select_icd_level(self, code, level = 3):
        if level == 3:
            return code[0].split(",")[0].split(".")[0]
        else:
            return code[0].split(",")[0]


class FeatureBuilder:
    def __init__(self, db_config: EICUConfig) -> None:
        # Label names defined by user (ICD-9).
        self.db_config = db_config
    
    def _map_demo_to_values(self, demo_feat, demo_vocab):
        # Map demographic data into a vector, -1 means the data is lost.
        demo_feat['gender'] = 1 if demo_feat['gender'] == 'Male' else 0
        demo_list = [-1] * len(demo_vocab)
        for name, idx in demo_vocab.items():
            if name == 'age' and '> 'in demo_feat[name]:
                demo_feat[name] = demo_feat[name].split("> ")[-1]
            demo_list[idx-1] = float(demo_feat[name])
        return demo_list
    
    def _map_seq_data_to_ids(self, seq_list, vocab, seq_type, med_merger = None):
        if seq_type == 'diag':
            idx_seq = list(set([
                    (vocab[code[0].split(",")[0].split(".")[0]], code[1])
                    for code in seq_list if code[0].split(",")[0].split(".")[0] in vocab
                ]))
        elif seq_type == 'med':
            if med_merger is None:
                idx_seq = list(set([
                        (vocab[code[0]], code[2])
                        for code in seq_list if code[0] in vocab
                    ]))
            else:
                idx_seq = list(set([
                    (vocab[med_merger.map(code[0].lower())], code[2])
                        for code in seq_list if med_merger.map(code[0].lower()) in vocab
                ]))
        elif seq_type == 'lab':
            idx_seq = list(set([
                    (vocab[code[0]], float(code[1]), code[2])
                    for code in seq_list if code[0] in vocab
                ]))
        return idx_seq

    def save(self, target_dir):
        voc_path = os.path.join(target_dir, 'id_json_data.json')
        with open(voc_path, 'w', encoding='utf-8') as fp:
            json.dump(self.processed_data, fp)

    def translate_raw_to_ids(self, json_data, vocab, med_merger = None):
        r"""Map all data into id format."""
        # Get all vocabularies.
        diag_vocab = vocab['diag_voc']
        med_vocab = vocab['med_voc']
        lab_vocab = vocab['lab_voc']
        demo_vocab = vocab['demographic']
        num_patient_more1 = 0
        # Mapping process.
        for host_id in tqdm(json_data, total=len(json_data)):
            if len(json_data[host_id]['unit_records']) > 1:
                num_patient_more1 += 1
            # Map demographic data into a vector, -1 means the data is lost.
            json_data[host_id]['demographic'] = self._map_demo_to_values(json_data[host_id]['demographic'],
                                                                         demo_vocab)
            for unit_id in json_data[host_id]['unit_records']:
                unit_record = json_data[host_id]['unit_records'][unit_id]
                # Map diagnosis data into ids.
                unit_record['diag_seq'] = self._map_seq_data_to_ids(unit_record['diag_seq'], diag_vocab, 'diag') 
                # Map medication data into ids.
                unit_record['med_seq'] = self._map_seq_data_to_ids(unit_record['med_seq'], med_vocab, 'med', med_merger) 
                # Map lab test data into ids.
                unit_record['lab_test_seq'] = self._map_seq_data_to_ids(unit_record['lab_test_seq'], lab_vocab, 'lab') 
        # Group features by timestamp.
        new_json_data = []
        for host_id in tqdm(json_data, total=len(json_data)):
            patient_example = {'unit_records':list(), 'demo': json_data[host_id]['demographic']}
            for unit_id in json_data[host_id]['unit_records']:
                unit_example = {}
                # Group diseases by their corresponding timestamp.
                temp = {}
                for code in json_data[host_id]['unit_records'][unit_id]['diag_seq']:
                    if code[1] not in temp:
                        temp[code[1]] = []
                    temp[code[1]].append(code[0])
                unit_example['diag_time'] = sorted(list(temp.keys()))
                unit_example['diag_seq'] = [tuple(temp[x]) for x in unit_example['diag_time']]
                # Group medications and their corresponding timestamp.
                temp = {}
                for code in json_data[host_id]['unit_records'][unit_id]['med_seq']:
                    if code[1] not in temp:
                        temp[code[1]] = []
                    temp[code[1]].append(code[0])
                unit_example['med_time'] = sorted(list(temp.keys()))
                unit_example['med_seq'] = [tuple(temp[x]) for x in unit_example['med_time']]
                # Group lab test results and their corresponding timestamp.
                temp = {}
                for code in json_data[host_id]['unit_records'][unit_id]['lab_test_seq']:
                    if code[2] not in temp:
                        temp[code[2]] = []
                    temp[code[2]].append((code[0], code[1]))
                unit_example['lab_time'] = sorted(list(temp.keys()))
                unit_example['lab_seq'] = [tuple(temp[x]) for x in unit_example['lab_time']]
                unit_example['discharge_time'] = int(json_data[host_id]['unit_records'][unit_id]['tocharge']) + \
                                                json_data[host_id]['unit_records'][unit_id]['unit_time']
                unit_example['discharge_status'] = json_data[host_id]['unit_records'][unit_id]['dischargestatus']
                unit_example['unit_time'] = json_data[host_id]['unit_records'][unit_id]['unit_time']
                patient_example['unit_records'].append(unit_example)
            patient_example['unit_records'] = sorted(patient_example['unit_records'], key=lambda x: x['unit_time'])
            new_json_data.append(patient_example)
        self.processed_data = new_json_data


class LabelBuilder:
    def __init__(self, db_config: EICUConfig) -> None:
        # Label names defined by user (ICD-9).
        self.db_config = db_config
    
    def save(self, target_dir):
        voc_path = os.path.join(target_dir, 'label_add_json_data.json')
        with open(voc_path, 'w', encoding='utf-8') as fp:
            json.dump(self.processed_data, fp)
    
    def label_processing_with_neg_samples(self, json_data, vocabulary):
        diag_voc = vocabulary['diag_voc']
        label_voc = vocabulary['label']
        label_set = {diag_voc[k]:label_voc[k] for k in label_voc if k in diag_voc}
        new_pos_json_data, new_neg_json_data, total_units = [], [], 0
        for patient in tqdm(json_data):
            unit_records = patient['unit_records']
            if len(unit_records) > 1:
                total_units += 1
                for i in range(len(unit_records)-1):
                    example = {}
                    unit_labels = []
                    # Check next ICU units.
                    for j, diags in enumerate(unit_records[i+1]['diag_seq']):
                        for diag in list(diags):
                            i_diag_set = set([y for x in unit_records[i]['diag_seq'] for y in x])
                            if diag in label_set:
                                unit_labels.append((label_set[diag], unit_records[i+1]['diag_time'][j]))
                    example['demographic'] = copy.deepcopy(patient['demo'])
                    example['features'] = copy.deepcopy(unit_records[i])
                    if unit_records[i+1]['discharge_status'] == 'Expired':
                        unit_labels.append((label_voc['Expired'], unit_records[i+1]['discharge_time']))
                    example['label'] = sorted(unit_labels, key=lambda x: x[1])
                    # Samples who won't be observed bad events.
                    if len(unit_labels) != 0:
                        example['label'].append((label_voc['END'], example['label'][-1][1]))
                        assert example['label'][-1][0] == label_voc['END']
                        new_pos_json_data.append(example)
            else:
                example = {}
                example['demographic'] = copy.deepcopy(patient['demo'])
                example['features'] = copy.deepcopy(unit_records[0])
                unit_labels = []
                if unit_records[0]['discharge_status'] == 'Expired':
                    unit_labels.append((label_voc['Expired'], unit_records[0]['discharge_time']))
                else:
                    unit_labels.append((label_voc['Normal'], 0))

                unit_labels.append((label_voc['END'], unit_labels[-1][1]+0.1))
                example['label'] = unit_labels
                assert example['label'][-1][0] == label_voc['END']
                # Samples who won't be observed bad events.
                new_neg_json_data.append(example)

        print("We get %d pos examples, %d neg examples, %d units" % (len(new_pos_json_data), len(new_neg_json_data),
                                                                     total_units))
        self.processed_data = new_pos_json_data + new_neg_json_data


class DataSampler:
    def __init__(self, db_config: EICUConfig) -> None:
        # Label names defined by user (ICD-9).
        self.db_config = db_config
    
    def save(self, target_dir):
        voc_path = os.path.join(target_dir, 'sampled_json_data.json')
        with open(voc_path, 'w', encoding='utf-8') as fp:
            json.dump(self.processed_data, fp)
    
    def is_filtered(self, sample):
        label_set = set([x[0] for x in sample['label']])
        # Filter patients without diagnosis codes recorded.
        if len(sample['features']['diag_seq']) == 0:
            return True
        # Filter normal patients.
        if len(sample['label']) == 2:
            if 5 not in label_set and 6 not in label_set:
                return False
            return True
        if 5 in label_set:
            return True
        return False
    
    def sampling(self, json_data):
        new_data = []
        total, filtered = 0, 0
        for sample in json_data:
            total += 1
            # Filter samples with only one element.
            if self.is_filtered(sample): 
                filtered += 1
                continue
            # Process the label sequence.
            new_label = []
            for i, (disease, time) in enumerate(sample['label']):
                if i == 0: new_label.append((disease, time))
                # Merge the same next IDs.
                if sample['label'][i][0] == new_label[-1][0] and \
                    (sample['label'][i][1] - new_label[-1][1]) < 24*60:
                    continue
                # Set minimum time interval as 0.1 days.
                if sample['label'][i][1] < new_label[-1][1]+int(0.1*24*60):
                    new_label.append((sample['label'][i][0], new_label[-1][1]+int(0.1*24*60)))
                    continue
                # Normal situations.
                new_label.append((disease, time))
            sample['label'] = new_label
            new_data.append(sample)
        print("Filtered %d/%d=%.4f" % (filtered, total, filtered/total))
        self.processed_data = new_data


class DataSpliter:
    def __init__(self, db_config: EICUConfig) -> None:
        # Label names defined by user (ICD-9).
        self.db_config = db_config
    
    def split_data(self, json_data, target_dir):
        train, eval, test = self.split_train_val_test_dataset(json_data)
        json.dump(train, open(os.path.join(target_dir, 'train.json'), 'w', encoding='utf-8'))
        json.dump(eval, open(os.path.join(target_dir, 'eval.json'), 'w', encoding='utf-8'))
        json.dump(test, open(os.path.join(target_dir, 'test.json'), 'w', encoding='utf-8'))
        print("We get %d/%d/%d" % (len(train), len(eval), len(test)))
    
    def split_train_val_test_dataset(self, json_data, test_ratio = 0.1):
        r"""Split a json-format data into train, val and test set."""
        assert type(json_data) == list
        assert test_ratio < 1, "Test ratio must be less than 1."
        print(json_data[0]['label'])
        random.shuffle(json_data)
        print(json_data[0]['label'])
        train_size = int(len(json_data)*(1-2*test_ratio))
        test_size = int(len(json_data)*test_ratio)
        return json_data[:train_size], \
                json_data[train_size:train_size+test_size], \
                json_data[train_size+test_size:]


if __name__ == '__main__':
    med_merger = EICUMedCodeMapper()
    print(med_merger.merge("metoprolol tartrate 25 mg po tabs"))