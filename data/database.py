import os
import collections
import random
import json
import asyncio
import pandas as pd
from tqdm import tqdm

class EICUConfig:
    def __init__(self, meta_config) -> None:
        assert "database_dir" in meta_config,        "set database_dir!"
        assert "table_name_cols_map" in meta_config, "table_name_cols_map should be provided"
        self.meta_config = meta_config
        self.db_name = meta_config['db_name']
        self.label_names = meta_config['label_names']
        self.database_dir = meta_config['database_dir']
        self.table_name_cols_map = meta_config['table_name_cols_map']
    
    @classmethod
    def load_from_default_settings(cls, setting):
        meta_config = {
            'db_name': setting.db_name,
            'label_names': setting.label_names,
            'database_dir': setting.db_dir,
            'table_name_cols_map': setting.table_name_cols_map
        }
        return EICUConfig(meta_config)



class EICUDataReader:
    def __init__(self, db_config: EICUConfig) -> None:
        self.db_config = db_config
        self.name_df_mapper = None
        self.name_df_row_counts = None

    def read_from_database(self, verbose = True):
        db_config = self.db_config
        assert db_config.meta_config is not None
        # Read dataframes from csv files.
        name_df_mapper = collections.defaultdict(pd.DataFrame)
        name_df_row_counts = collections.defaultdict(int)
        for tb_name, tb_cols in db_config.table_name_cols_map.items():
            tb_abs_path = os.path.join(db_config.database_dir, tb_name + ".csv")
            df = pd.read_csv(tb_abs_path)[tb_cols].dropna().reset_index()
            name_df_mapper[tb_name] = self.process_abs_timestamp(df, tb_name)
            name_df_row_counts[tb_name] += len(name_df_mapper[tb_name])
        self.name_df_mapper = name_df_mapper
        self.name_df_row_counts = name_df_row_counts
        if verbose: print(name_df_row_counts)
    
    def process_abs_timestamp(self, df: pd.DataFrame, name: str):
        r""" Add the hospitaladmitoffset column into all tables after filtering negative values.
        After filtering strange examples, we get 95,618 patients.
        """
        if name == 'patient':
            df = df[df['hospitaladmitoffset'] < 0]
            df['unitoffset'] = -df['hospitaladmitoffset']
            df['hospitaladmitoffset'] -= df['hospitaladmitoffset']
        elif name == 'medication':
            df = df[df['drugorderoffset'] >= 0]
        elif name == 'diagnosis':
            df = df[df['diagnosisoffset'] >= 0]
        elif name == 'lab':
            df = df[df['labresultoffset'] >= 0]
        return df
    
    def translate_df_to_json(self):
        # If possible, read dataframes from csv files.
        if self.name_df_mapper is None or self.name_df_row_counts is None:
            self.read_from_database()
        # Build patient containers.
        self._build_patient_containers()
        # Extract important information.
        self._extract_diag_info()
        self._extract_med_info()
        self._extract_lab_info()
    
    def save(self, target_path: str):
        print("We save data to path ", target_path)
        json.dump(self.all_data, open(target_path, 'w', encoding='utf-8'))
    
    def show_example(self):
        host_ids = list(self.all_data.keys())
        host_id = random.randint(0, len(host_ids)-1)
        print(self.all_data[host_ids[host_id]])
    
    def _build_patient_containers(self):
        # Initialize the data structure from patient_time_select. Got 88,231 patients.
        all_data, patient_df = dict(), self.name_df_mapper['patient']
        num_patient_more1 = 0
        for _, row in tqdm(patient_df.iterrows(), total=len(patient_df)):
            host_id, unit_id = int(row['patienthealthsystemstayid']), int(row['patientunitstayid'])
            if host_id not in all_data:
                all_data[host_id] = {'unit_records':dict(), 'demographic':dict()}
            if unit_id not in all_data[host_id]['unit_records']:
                all_data[host_id]['unit_records'][unit_id] = {
                    'diag_seq': [],
                    'med_seq': [],
                    'lab_test_seq': [],
                    'unit_time': int(row['unitoffset']),
                    'tocharge': int(row['hospitaldischargeoffset']),
                    'dischargestatus': row['unitdischargestatus']
                }
            all_data[host_id]['demographic'] = {
                'gender': row['gender'],
                'age': row['age'],
                'height': row['admissionheight'],
                'admissionweight': row['admissionweight'],
            }
        for host_id in all_data:
            if len(all_data[host_id]['unit_records']) > 1:
                num_patient_more1 += 1
        self.all_data = all_data
        self.unit2host = {x:k for k, v in all_data.items() for x in v['unit_records'] if type(x) == int}

    def _extract_diag_info(self):
        # Extract diagnosis information from diagnosis_time_select. 
        all_data, diag_df, unit2host = self.all_data, self.name_df_mapper['diagnosis'], self.unit2host
        diag_df = diag_df.groupby('patientunitstayid', as_index=False).agg(lambda x: list(x))
        print("Extract diagnosis information from diagnosis_time_select. ")
        for _, row in tqdm(diag_df.iterrows(), total=len(diag_df)):
            unit_id = int(row['patientunitstayid'])
            host_id = unit2host.get(unit_id)
            if host_id in all_data and unit_id in all_data[host_id]['unit_records']:
                icdcode, icd_time = row['icd9code'], row['diagnosisoffset']
                unit_time = all_data[host_id]['unit_records'][unit_id]['unit_time']
                icd_time = [int(x)+unit_time for x in icd_time]
                all_data[host_id]['unit_records'][unit_id]['diag_seq'].extend(list(zip(icdcode, icd_time)))
                # Sort the icd9codes by time.
                all_data[host_id]['unit_records'][unit_id]['diag_seq'] = sorted(all_data[host_id]['unit_records'][unit_id]['diag_seq'],
                                                                key=lambda x: x[1])
        self.all_data = all_data
    
    def _extract_med_info(self):
        # Extract medication information from medication_time_select. 
        all_data, med_df, unit2host = self.all_data, self.name_df_mapper['medication'], self.unit2host
        med_df = med_df.groupby('patientunitstayid', as_index=False).agg(lambda x: list(x))
        print("Next, we extract medication information from medication_time_select.")
        for _, row in tqdm(med_df.iterrows(), total=len(med_df)):
            unit_id = int(row['patientunitstayid'])
            host_id = unit2host.get(unit_id)
            if host_id in all_data and unit_id in all_data[host_id]['unit_records']:
                drugname, dosage, timestamp = row['drugname'], row['dosage'], row['drugorderoffset']
                unit_time = all_data[host_id]['unit_records'][unit_id]['unit_time']
                timestamp = [int(x)+unit_time for x in timestamp]
                all_data[host_id]['unit_records'][unit_id]['med_seq'].extend(list(zip(drugname, dosage, timestamp)))
                all_data[host_id]['unit_records'][unit_id]['med_seq'] = sorted(all_data[host_id]['unit_records'][unit_id]['med_seq'],
                                                                key=lambda x: x[2])
        self.all_data = all_data
    
    def _extract_lab_info(self):
        # Extract lab test information from diagnosis_time_select.
        all_data, lab_df, unit2host = self.all_data, self.name_df_mapper['lab'], self.unit2host
        lab_df = lab_df.groupby('patientunitstayid', as_index=False).agg(lambda x: list(x))
        print("Finally, we extract lab test information from diagnosis_time_select.")
        for _, row in tqdm(lab_df.iterrows(), total=len(lab_df)):
            unit_id = int(row['patientunitstayid'])
            host_id = unit2host.get(unit_id)
            if host_id in all_data and unit_id in all_data[host_id]['unit_records']:
                labname, labresult, timestamp = row['labname'], row['labresult'], row['labresultoffset']
                unit_time = all_data[host_id]['unit_records'][unit_id]['unit_time']
                timestamp = [int(x)+unit_time for x in timestamp]
                all_data[host_id]['unit_records'][unit_id]['lab_test_seq'].extend(list(zip(labname, labresult, timestamp)))
                all_data[host_id]['unit_records'][unit_id]['lab_test_seq'] = sorted(all_data[host_id]['unit_records'][unit_id]['lab_test_seq'],
                                                                key=lambda x: x[2])
        self.all_data = all_data

        

