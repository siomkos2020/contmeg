# Write the main logics for training, evaluating and testing a disease-progression model.
import sys
sys.path.append('.')
sys.path.append('..')
import json
import random
import os
import argparse
import dill
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from data_utils import *
from tpp_models import *
from eval_utils import get_seq_generation_metrics

def set_seed(seed):
    r"""Set the random seed at seed."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_environ_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Learning rate.")
    parser.add_argument("--train_path", type=str, default="xxx", help="Training path.")
    parser.add_argument("--eval_path", type=str, default="xxx", help="Eval path.")
    parser.add_argument("--test_path", type=str, default="xxx", help="Test path.")
    parser.add_argument("--vocab_path", type=str, default="xxx", help="Vocabulary path.")
    parser.add_argument("--device", type=int, default=0, help="Vocabulary path.")
    parser.add_argument("--num_labels", type=int, default=5, help="Number of predicted labels.")
    parser.add_argument("--model_save_dir", type=str, default="", help="Model saved path.")
    parser.add_argument("--model_name", type=str, default='BasicSeq2Seq', help="Model name.")
    parser.add_argument("--model_load_path", type=str, default='', help="Model loading path.")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--db_name",  type=str, default='', help="Dataset name.")
    parser.add_argument("--log_dir", type=str, default="./logger")
    parser.add_argument("--gen_preds", action='store_true')
    
    return parser.parse_args()

def map_category_to_times(time_cates: torch.tensor, dataset_name: str):
    if dataset_name == 'eicu':
        cont = time_cates.clone().float()
        cont[time_cates == 0] = 0.1
        cont[time_cates == 1] = 0.6
        cont[time_cates == 2] = 3.
        cont[time_cates == 3] = 5. 
        return cont


def eval(model, eval_dataloader, device, step, args, dataset_name='eicu', save_preds = False):
    model.eval()
    batch_type_preds, batch_time_preds, batch_type_gt, batch_time_gt, batch_type_mask = [], [], [], [], []
    batch_diag_seq, batch_diag_time, batch_med_seq, batch_med_time, batch_lab_seq, batch_lab_time = [], [], [], [], [], []
    batch_intensity_funcs, batch_delta_funcs = [], []
    batch_type_probs = []
    eval_loss_mean = []
    for _, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader.dataset)//args.batch_size):
        # Receive input tensors.
        diag_seq = batch['diag_seq'].long().to(device)
        diag_time = batch['diag_times'].float().to(device)
        diag_mask = batch['diag_mask'].float().to(device)

        med_seq = batch['med_seq'].long().to(device)
        med_time = batch['med_times'].float().to(device)
        med_mask = batch['med_mask'].float().to(device)

        lab_seq = batch['lab_seq'].long().to(device)
        lab_time = batch['lab_times'].float().to(device)
        lab_mask = batch['lab_mask'].float().to(device)
        lab_ts = batch['lab_ts'].float().to(device)

        batch_label_seq = batch['label_seq'].long().to(device)
        batch_label_times = batch['label_times'].long().to(device)
        batch_label_mask = batch['label_mask'].float().to(device)

        demo = batch['demo'].float().to(device)
        batch_final_diags = batch['final_diag'].long().to(device)
        batch_final_mask = batch['final_diag_mask'].long().to(device)
        # 收集eval的loss
        model.train()
        eval_loss = model(diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                    lab_mask, demo, batch_label_seq, batch_label_times, batch_label_mask, lab_ts,
                    batch_final_diags, batch_final_mask)
        eval_loss_mean.append(eval_loss.item())
        model.eval()
        # Model forward
        if save_preds and args.model_name in ['multi_tpp_tl_lab']:
            try:
                type_preds, type_probs, time_preds, inten_funcs, delta_attns = model(diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                    lab_mask, demo, batch_label_seq, batch_label_times, batch_label_mask, lab_ts, 
                    batch_final_diags, batch_final_mask, test_mode=True)
                batch_intensity_funcs.extend(inten_funcs)
                batch_delta_funcs.append(delta_attns)
            except Exception as e:
                print(e)
                continue
        else:
            type_preds, type_probs, time_preds = model(diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                    lab_mask, demo, batch_label_seq, batch_label_times, batch_label_mask, lab_ts, 
                    batch_final_diags, batch_final_mask)
        batch_label_times = map_category_to_times(batch_label_times, dataset_name)
        if time_preds.dtype != torch.float:
            time_preds = map_category_to_times(time_preds, dataset_name)
        batch_type_preds.extend(type_preds.detach().cpu().numpy().tolist())
        batch_time_preds.extend(time_preds.detach().cpu().numpy().tolist())
        batch_type_gt.extend(batch_label_seq.detach().cpu().numpy().tolist())
        batch_time_gt.extend(batch_label_times.detach().cpu().numpy().tolist())
        batch_type_mask.extend(batch_label_mask.detach().cpu().numpy().tolist())
        batch_type_probs.extend(type_probs.detach().cpu().numpy().tolist()) 
        # Collect examples
        if save_preds:
            batch_diag_seq.extend(diag_seq.detach().cpu().numpy().tolist())
            batch_diag_time.extend(diag_time.detach().cpu().numpy().tolist())
            batch_med_seq.extend(med_seq.detach().cpu().numpy().tolist())
            batch_med_time.extend(med_time.detach().cpu().numpy().tolist())
            batch_lab_seq.extend(lab_seq.detach().cpu().numpy().tolist())
            batch_lab_time.extend(lab_time.detach().cpu().numpy().tolist())

    # Achieve generative metrics.
    end_id = eval_dataloader.dataset.label_vocab['END']
    lls, rmses, accs, aucs = get_seq_generation_metrics(batch_type_preds, batch_time_preds, 
                                                 batch_type_gt, batch_time_gt, batch_type_probs,
                                                 END_ID=end_id)

    print("\nEval %d steps" % (step))
    print("lls@k:        ", lls)
    print("accs@k:       ", accs)
    print("rmses@k:      ", rmses)
    model.train()
    return lls, rmses, accs, aucs, np.mean(eval_loss_mean)

def build_model(args, **kwargs):
    model_name, vocab, device = kwargs['model_name'], kwargs['vocab'], kwargs['device']
    # Split time interval into ranges.
    if args.db_name == 'eicu':
        timebucket = 4
    common_args = {'num_diag': len(vocab['diag_voc'])+1,
                   'num_med': len(vocab['med_voc'])+1,
                   'num_lab': len(vocab['lab_voc'])+1,
                   'num_labels': len(vocab['label'])+1,
                   'time_buckets': timebucket}
    if model_name == 'multi_tpp_tl_lab':
        if args.db_name == 'eicu':
            time_discrete_map = {0:0.1, 1:0.6, 2:3., 3:5.}
            model = MultiTaskTimeTPPEnhanceSeq(**common_args, hidden_dim=256, time_discrete_fn=map_category_to_times,
                        time_discrete_map=time_discrete_map, dataset_name=args.db_name)
    model = model.to(device)
    return model

def train(model, datasets, device, args):
    batch_size = args.batch_size
    # Prepare dataloader.
    train_dataset, eval_dataset = datasets['train'], datasets['eval']
    train_dataloader = prepare_HED_dataloader(train_dataset, batch_size)
    eval_dataloader = prepare_HED_dataloader(eval_dataset, batch_size)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                args.lr, betas=(0.9, 0.999), eps=1e-5)
    global_step = 0
    model.train()
    best_perform = -1
    train_global_loss, train_mean_loss, eval_mean_loss = [], [], []
    # eval(model, eval_dataloader, device, global_step, args)
    for epoch in range(args.epochs):
        loss_mean = []
        for i, batch in enumerate(train_dataloader):
            # Receive input tensors.
            diag_seq = batch['diag_seq'].long().to(device)
            diag_time = batch['diag_times'].float().to(device)
            diag_mask = batch['diag_mask'].float().to(device)

            med_seq = batch['med_seq'].long().to(device)
            med_time = batch['med_times'].float().to(device)
            med_mask = batch['med_mask'].float().to(device)

            lab_seq = batch['lab_seq'].long().to(device)
            lab_time = batch['lab_times'].float().to(device)
            lab_mask = batch['lab_mask'].float().to(device)
            lab_ts = batch['lab_ts'].float().to(device)

            batch_label_seq = batch['label_seq'].long().to(device)
            batch_label_times = batch['label_times'].long().to(device)
            batch_label_mask = batch['label_mask'].float().to(device)

            demo = batch['demo'].float().to(device)
            batch_final_diags = batch['final_diag'].long().to(device)
            batch_final_mask = batch['final_diag_mask'].long().to(device)
            # Model forward
            loss = model(diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                    lab_mask, demo, batch_label_seq, batch_label_times, batch_label_mask, lab_ts,
                    batch_final_diags, batch_final_mask)
            # Compute classification loss and averaged on lens.
            loss_mean.append(loss.item())
            # 
            train_global_loss.append(loss.item())
            # Optimization.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            print("\r Step: %d, loss: %.4f" % (global_step, loss.item()), end="")
        # Evalution at each epoch.
        print("\nEPOCH: %d, Step: %d, loss: %.4f\n" % (epoch, global_step, np.mean(loss_mean)))
        if epoch % 1 == 0:
            _, rmses, accs, _, evloss = eval(model, eval_dataloader, device, global_step, args)
            eval_mean_loss.append(evloss)
            train_mean_loss.append(np.mean(loss_mean).item())
            print(len(eval_mean_loss), len(train_mean_loss))
            mean_acc = np.array(accs).mean().item()
            mean_rmses = np.array(rmses).mean().item()
            if -mean_rmses*0.2 + 0.8*mean_acc > best_perform:
                best_perform = -mean_rmses*0.2 + 0.8*mean_acc
                torch.save({'model': model.state_dict(),
                            'best_perform': best_perform, 'epoch':epoch}, 
                            os.path.join(args.model_save_dir, args.model_name+".pth"))
                print("Model saved to %s" % (os.path.join(args.model_save_dir, args.model_name+".pth")))
    json.dump({
        'train_global_loss': train_global_loss,
        'train_mean_loss': train_mean_loss,
        'eval_mean_loss': eval_mean_loss
    }, open('loss_records.json', 'w'))

def test(model, vocab, device, db_class, args):
    batch_size = args.batch_size
    lls_list, rmses_list, accs_list, macro_accs_list, aucs_list = [], [], [], [], []
    model.load_state_dict(torch.load(args.model_load_path)['model'], strict=True)
    rounds = 5 if not args.gen_preds else 1
    for _ in range(rounds):
        if not args.gen_preds:
            test_dataset = db_class(args.test_path, label_vocab=vocab['label'], is_train=False, time_level='day', multi_test=True)
        else:
            test_dataset = db_class(args.test_path, label_vocab=vocab['label'], is_train=False, time_level='day')
        test_dataloader = prepare_HED_dataloader(test_dataset, batch_size)
        global_step = 0
        lls, rmses, accs, aucs = eval(model, test_dataloader, device, global_step, args, save_preds=True)
        lls_list.append(lls)
        rmses_list.append(rmses)
        accs_list.append(accs)
        aucs_list.append(aucs)

    lls_list = np.array(lls_list)
    rmses_list = np.array(rmses_list)
    accs_list = np.array(accs_list)
    macro_accs_list = np.array(macro_accs_list)
    aucs_list = np.array(aucs_list)
    avg_lls_list, std_lls_list = lls_list.mean(0), lls_list.std(0)
    avg_rmses_list, std_rmses_list = rmses_list.mean(0), rmses_list.std(0)
    avg_accs_list, std_accs_list = accs_list.mean(0), accs_list.std(0)
    avg_macro_accs_list, std_macro_accs_list = macro_accs_list.mean(0), macro_accs_list.std(0)
    avg_aucs_list, std_aucs_list = aucs_list.mean(0), aucs_list.std(0)

    lls = [str("%.2f"%x)+"("+str("%.2f"%y)+")" for x, y in zip(avg_lls_list.tolist(), std_lls_list.tolist())]
    rmses = [str("%.2f"%x)+"("+str("%.2f"%y)+")" for x, y in zip(avg_rmses_list.tolist(), std_rmses_list.tolist())]
    accs = [str("%.2f"%(x*100))+"("+str("%.2f"%(y*100))+")" for x, y in zip(avg_accs_list.tolist(), std_accs_list.tolist())]
    aucs = [str("%.2f"%(x*100))+"("+str("%.2f"%(y*100))+")" for x, y in zip(avg_aucs_list.tolist(), std_aucs_list.tolist())]
    for i in range(len(avg_lls_list)):
        print("ll  (L=%d) %s, acc (L=%d) %s, rmse(L=%d) %s, auc (L=%d) %s" % 
              (i+1, lls[i], i+1, accs[i], i+1, rmses[i], i+1, aucs[i]))


if __name__ == '__main__':
    # Parse environment arguments.
    args = get_environ_arguments()
    # Set the random seed.
    set_seed(args.seed)
    # Load vocab.
    vocab = json.load(open(args.vocab_path, 'r', encoding='utf-8'))
    # Initialize the model
    device = torch.device("cuda:%d" % args.device)
    model = build_model(args, **{'model_name': args.model_name, 'vocab': vocab, 'device': device})
    if args.db_name == 'eicu':
        db_class = HealthEventData
    elif args.db_name == 'youyidata':
        db_class = PrivateHealthEventData
    # Main process.
    if args.test:
        print("\n=============test process==============")
        test(model, vocab, device, db_class, args)
    else:
        datasets = {
            'train': db_class(args.train_path, label_vocab=vocab['label'], is_train=True, time_level='day'),
            'eval': db_class(args.eval_path, label_vocab=vocab['label'], is_train=False, time_level='day'),
        }
        train(model, datasets, device, args)


