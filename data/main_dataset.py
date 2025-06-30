import sys
sys.path.append(".")
import json
import os
import argparse
from default_settings import EICUDefaultSettings
from database import EICUConfig, EICUDataReader
from feature import VocabBuilder, FeatureBuilder, LabelBuilder, DataSampler, DataSpliter

def get_environ_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default='eicu', choices=['eicu'], help="The database you selected.")
    parser.add_argument("--db_path", type=str, default='eicu', choices=['eicu'], help="The database path.")
    parser.add_argument("--target_dir", type=str, default=".", help="Target directory to save.")

    return parser.parse_args()



if __name__ == '__main__':
    args = get_environ_args()
    reader = None
    if args.db_name == 'eicu':
        # Basic dataconfig for generating dataset.
        eicu_config = EICUConfig.load_from_default_settings(EICUDefaultSettings(args.db_path))
        reader = EICUDataReader(eicu_config)
    if reader is not None:
        target_voc_dir = args.target_dir
        print("Dataset processing pipeline works...")
        debug_reader = False
        if not os.path.exists(os.path.join(target_voc_dir, 'before_feat_ext.json')) or debug_reader:
            reader.translate_df_to_json()
            reader.save(os.path.join(target_voc_dir, 'before_feat_ext.json'))
            reader.show_example()
            print("data saved %s" % os.path.join(target_voc_dir, 'before_feat_ext.json'))
        else:
            reader.all_data = json.load(open(os.path.join(target_voc_dir, 'before_feat_ext.json'),
                            'r', encoding='utf-8'))
        
        print("Building vocabulary...")
        voc_debug = False
        voc_builder = VocabBuilder(eicu_config)
        if not os.path.exists(os.path.join(target_voc_dir, 'voc.json')) or voc_debug:
            voc_builder.build_voc_from_data(reader.all_data)
            voc_builder.save(target_voc_dir)
            print("Vocabulary saved to %s" % target_voc_dir)
        else:
            voc_builder.vocabulary = json.load(open(os.path.join(target_voc_dir, 'voc.json'),
                            'r', encoding='utf-8'))

        
        print("Building features and index mapping..")
        feat_debug = False
        feat_builder = FeatureBuilder(eicu_config)
        tgt_path = os.path.join(target_voc_dir, 'id_json_data.json')
        if not os.path.exists(tgt_path) or feat_debug:
            feat_builder.translate_raw_to_ids(reader.all_data, voc_builder.vocabulary, voc_builder.code_merger)
            feat_builder.save(target_voc_dir)
        else:
            feat_builder.processed_data = json.load(open(tgt_path, 'r', encoding='utf-8'))

        print("Label building...")
        label_builder = LabelBuilder(eicu_config)
        label_debug = False
        tgt_path = os.path.join(target_voc_dir, 'label_add_json_data.json')
        if not os.path.exists(tgt_path) or label_debug:
            label_builder.label_processing_with_neg_samples(feat_builder.processed_data, voc_builder.vocabulary)
            label_builder.save(target_voc_dir)
        else:
            label_builder.processed_data = json.load(open(tgt_path, 'r', encoding='utf-8'))

        print("Data sampling...")
        data_sampler = DataSampler(eicu_config)
        data_sampler.sampling(label_builder.processed_data)

        print("Train/dev/test spliting...")
        data_spliter = DataSpliter(eicu_config)
        data_spliter.split_data(data_sampler.processed_data, target_voc_dir)
        print("Data saved to %s" % target_voc_dir)