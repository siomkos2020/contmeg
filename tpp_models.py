import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from patient_rep import *

def softplus(x, beta = None):
    # hard thresholding at 20
    # temp = beta * x
    temp = x
    # temp[temp > 20] = 20
    # return 1.0 / beta * torch.log(1 + torch.exp(temp))
    return torch.log(1 + torch.exp(temp))



class ExpertNet(nn.Module):
    def __init__(self, feature_dim, output_dim, dropout_p = 0.):
        super().__init__()
        # Input dim & output dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        # Backbone
        self.dnn_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(feature_dim, output_dim),
        )

    def forward(self, input_hiddens):
        r"""
        input_hiddens: (b, h)
        """
        assert input_hiddens.size()[-1] == self.feature_dim
        out = self.dnn_layer(input_hiddens)
        return out


class CGCLayer(nn.Module):
    def __init__(self, feature_dim,
                 task_dim=64, n_experts=2,
                 n_share_experst=1):
        super().__init__()
        self.n_expert = n_experts
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 1+n_share_experst),
                nn.Softmax(dim=-1),
            )
        ]*n_experts)
        self.share_tasks = ExpertNet(feature_dim, task_dim)
        self.tasks = nn.ModuleList([ExpertNet(feature_dim, task_dim),
                                    ExpertNet(feature_dim, task_dim)])
    
    def forward(self, inputs):
        # Outputs from shared network.
        shared_outputs = self.share_tasks(inputs).unsqueeze(1)            # (b, feat_dim)
        task_outputs = []
        for i, expert in enumerate(self.tasks):
            task_out = expert(inputs).unsqueeze(1)          # (b, 1, feat_dim)
            gate = self.gates[i](inputs).unsqueeze(-1)       # (b, 2, 1)
            task_out = (gate * torch.cat([shared_outputs, task_out], dim=1)).sum(1)
            task_outputs.append(task_out)
        
        return task_outputs[0], task_outputs[1]



class MultiTaskTimeTPPEnhanceSeq(nn.Module):
    def __init__(self,
                 num_diag,
                 num_med,
                 num_lab,
                 num_labels,
                 max_output_points = 10,
                 pat_rep_dim = 128,
                 mse_loss_weight = 1,
                 hidden_dim = 512,
                 time_discrete_fn = None,
                 time_discrete_map = None,
                 time_buckets = 4,
                 dataset_name = 'eicu',
                 rep_model = 'timeline'
                 ):
        super().__init__()
        self.num_labels = num_labels                        # Number of considered events.
        self.max_output_points = max_output_points          # Maximum points to generate.
        if rep_model == 'timeline':
            self.patient_rep = TimeLine(num_diag, num_med, num_lab, pat_rep_dim)
        elif rep_model == 'hitanet':
            self.patient_rep = BaseMLP(num_diag, num_med, num_lab, pat_rep_dim)
        self.emb_drop = nn.Dropout(0.5)
        # Label embedding, add PAD and END.
        self.label_embed = nn.Embedding(num_labels+2, pat_rep_dim)
        self.time_embed = nn.Embedding(time_buckets, pat_rep_dim)
        # Decoder for generating future events.
        self.hidden_dim = hidden_dim
        self.type_decoder = nn.GRU(pat_rep_dim*2, self.hidden_dim, batch_first=True, dropout=0.2)
        self.dec_token1 = nn.Parameter(torch.randn(1, pat_rep_dim, requires_grad=True))
        self.dec_token2 = nn.Parameter(torch.randn(1, pat_rep_dim, requires_grad=True))
        self.time_decoder = nn.GRU(pat_rep_dim*2, self.hidden_dim,  batch_first=True, dropout=0.2)
        self.lambda_decoder = nn.GRU(pat_rep_dim*2, self.hidden_dim, batch_first=True, dropout=0.2)
        self.pat2base = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.time_coef = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        # Multi-task modules
        self.cgc_layer = CGCLayer(pat_rep_dim, task_dim=pat_rep_dim)
        self.type_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels+2),
        )
        self.time_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, time_buckets)
        )

        self.type_loss = nn.CrossEntropyLoss(reduce=False)
        self.time_discrete_fn = time_discrete_fn
        self.time_discrete_map = time_discrete_map
        self.dataset_name = dataset_name
    
    def get_intensity_function(self, encs, t_interval):
        r"""Poisson process assumes historical dependence.
        encs:       (b, ls, enc_dim)
        t_inerval:  (b, T/ls, 1)
        """
        seq_len, time_len = encs.size()[1], t_interval.size()[1]
        if seq_len == time_len:
            intensity = softplus(self.pat2base(encs) + self.time_coef(t_interval)) # (b, ls, 1)
            return intensity
        else:
            base_feat = self.pat2base(encs).unsqueeze(2)                  # (b, ls, 1, num_label+2)
            time_effect = self.time_coef(t_interval).unsqueeze(1)         # (b, 1, T, num_label+2)
            intensity = softplus(base_feat + time_effect)                 # (b, ls, T, 1)
            return intensity

    def _compute_event_ll(self, encs, label_seq, label_intervals, obs_mask):
        intensity_obs = self.get_intensity_function(encs, label_intervals.unsqueeze(-1))        # (b, ls, 1)
        intensity_obs = intensity_obs.squeeze(-1)   # (b, ls,)
        # Maximize intensities of observed events.
        event_ll = ((torch.log(intensity_obs + 1e-6)*obs_mask).sum(1) / obs_mask.sum(1)).mean()

        return event_ll
    
    def _compute_non_event_ll(self, encs, label_seq, obs_mask):
        r"""Importance sampling for estimating integral."""
        n_samp, T = 100, 10
        sampled_times = torch.rand(size=(1, n_samp), device=label_seq.device)*T            # (1, n_samp)
        intensity_unobs = self.get_intensity_function(encs, sampled_times.unsqueeze(-1))                   # (b, ls, n_samp, num_label+2)
        non_event_ll = (((intensity_unobs.sum(-1).mean(2))*obs_mask).sum(1) / obs_mask.sum(1)).mean()
        return non_event_ll

    def _compute_multi_task_loss(self, time_rnn_encs, type_rnn_encs, label_seq, label_times, obs_mask):
        b_size, seq_len, _ = time_rnn_encs.size()
        type_rnn_encs = type_rnn_encs.view(b_size*seq_len, -1)
        time_rnn_encs = time_rnn_encs.reshape(b_size*seq_len, -1)
        label_seq = label_seq.view(b_size*seq_len,)
        label_times = label_times.view(b_size*seq_len,)
        # type_preds, time_preds = self.cgc_layer(encs)
        type_preds, time_preds = type_rnn_encs, time_rnn_encs
        type_preds = self.type_predictor(type_preds)
        time_preds = self.time_predictor(time_preds)
        # Compute time prediction loss.
        time_loss = self.type_loss(time_preds, label_times).view(b_size, seq_len)
        time_loss = ((time_loss * obs_mask).sum(1) / obs_mask.sum(1)).mean()
        # Compute type prediction loss.
        type_loss = self.type_loss(type_preds, label_seq).view(b_size, seq_len)
        type_loss = ((type_loss * obs_mask).sum(1) / obs_mask.sum(1)).mean()
        return time_loss + type_loss

    def compute_loss(self, time_rnn_encs, type_rnn_encs, lambda_rnn_encs, label_seq, label_times, label_mask):
        r"""Compute MLE loss for inhomogeneous Poisson Process.
        encs: (b, ls, enc_dim)
        label_times: (b, ls)
        """
        # Intensity function at observed events.
        obs_mask = label_mask
        # Maximize intensities of observed events.
        # Make label times continuous.
        label_times_cont = self.time_discrete_fn(label_times, self.dataset_name)
        event_ll = self._compute_event_ll(lambda_rnn_encs, label_seq, label_times_cont, obs_mask)                             # (b, ls,)
        # Minimize intensities of unobserved events (calculated by the numerical integral method).
        non_event_ll = self._compute_non_event_ll(lambda_rnn_encs, label_seq, obs_mask)
        event_loss = -(event_ll - non_event_ll)
        # Predictive loss for event types and time intervals.
        multi_task_loss = self._compute_multi_task_loss(time_rnn_encs, type_rnn_encs, label_seq, label_times, obs_mask)
        # Compute loss for predicting first (event, time).
        total_loss = 0.1*event_loss + multi_task_loss
        return total_loss
   
    def get_patient_rep(self, diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                lab_mask, demo, label_seq, label_times, label_mask, lab_ts = None, final_diags = None, final_diags_mask = None):
        return self.patient_rep(diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                lab_mask, demo)

    def forward(self, diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                lab_mask, demo, label_seq, label_times, label_mask, lab_ts = None, final_diags = None, final_diags_mask = None,
                test_mode = False):
        rep = self.get_patient_rep(diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                lab_mask, demo, label_seq, label_times, label_mask, lab_ts = None, final_diags = None, final_diags_mask = None)
        
        if self.training:
            # Label embedding.
            batch_size, ls = label_seq.size()
            label_embs = self.emb_drop(self.label_embed(label_seq))                          # (b, ls, rnn_in)
            # Add time embeddings into decoding.
            time_embs = self.emb_drop(self.time_embed(label_times)) 
            # Concat patient representations and rnn_inputs
            # Multi-task aggregation layer.
            time_rep, type_rep = self.cgc_layer(rep)
            time_rep = time_rep + rep
            type_rep = type_rep + rep
            type_rnn_inputs = torch.cat([self.dec_token1.repeat((batch_size, 1)).unsqueeze(1), label_embs[:, :-1, :]], dim=1)            # (b, ls, rnn_in)
            type_rnn_inputs = torch.cat([type_rep.unsqueeze(1).repeat((1, ls, 1)), type_rnn_inputs], dim=-1)
            time_rnn_inputs = torch.cat([self.dec_token2.repeat((batch_size, 1)).unsqueeze(1), time_embs[:, :-1, :]], dim=1)            # (b, ls, rnn_in)
            time_rnn_inputs = torch.cat([time_rep.unsqueeze(1).repeat((1, ls, 1)), time_rnn_inputs], dim=-1)
            lambda_rnn_inputs = type_rnn_inputs.detach() + time_rnn_inputs.detach()
            # First hidden vector.
            h0 = torch.zeros((1, 1, self.hidden_dim), device=type_rnn_inputs.device).repeat((1, batch_size, 1))   # (1, b, h)
            # Decoding process.
            type_rnn_encs, _ = self.type_decoder(type_rnn_inputs, h0)
            time_rnn_encs, _ = self.time_decoder(time_rnn_inputs, h0)
            lambda_encs, _ = self.lambda_decoder(lambda_rnn_inputs, h0)

            label_times_cont = self.time_discrete_fn(label_times, self.dataset_name)
            intensity_obs = self.get_intensity_function(lambda_encs, label_times_cont.unsqueeze(-1))        # (b, ls, 1)
            intensity_obs = intensity_obs.detach()
            intensity_mask = torch.tril(label_mask.unsqueeze(-1).repeat((1, 1, ls)))                # (b, ls, ls)
            intensity_mask = torch.where(intensity_mask > 0.5, 0., -1e9)
            intensity_mask = torch.softmax(intensity_obs+intensity_mask, dim=-1)
            type_rnn_encs = torch.bmm(intensity_mask, type_rnn_encs)                                  # (b, ls, rnn_in)
            # Calculate loss function based on TPP assumption.
            loss = self.compute_loss(time_rnn_encs, type_rnn_encs, lambda_encs, label_seq, label_times, label_mask)
            return loss
        else:
            if test_mode:
                return self.generate_future_points(rep, label_times, label_seq, return_int_func=True)
            return self.generate_future_points(rep, label_times, label_seq)
    

    def generate_future_points(self, patient_rep, gt_label_time = None, gt_label_seq = None, return_int_func = False):
        # Returned results.
        type_preds, type_probs, time_preds = [], [], []
        returned_intensity_funcs = []
        # Compute initial patient representations from RNN.
        b_size, _ = patient_rep.size()
        h0 = torch.zeros((1, 1, self.hidden_dim), device=patient_rep.device).repeat((1, b_size, 1))                             # (1, b, h)
        time_rep, type_rep = self.cgc_layer(patient_rep)
        time_rep = time_rep + patient_rep
        type_rep = type_rep + patient_rep
        # Initial RNN inputs.
        type_rnn_inputs = torch.cat([type_rep.unsqueeze(1), self.dec_token1.repeat((b_size, 1)).unsqueeze(1)], dim=-1)        # (b, 1, 2h)
        time_rnn_inputs = torch.cat([time_rep.unsqueeze(1), self.dec_token2.repeat((b_size, 1)).unsqueeze(1)], dim=-1)        # (b, 1, 2h)
        time_rnn_encs, _ = self.time_decoder(time_rnn_inputs, h0)                                        # (b, 1, 2*h)
        type_rnn_encs, _ = self.type_decoder(type_rnn_inputs, h0)
        lambda_rnn_inputs = type_rnn_inputs+time_rnn_inputs
        lambda_rnn_encs, _ = self.type_decoder(type_rnn_inputs+time_rnn_inputs, h0)

        sampled_times = [0.1*x for x in range(10*10)]
        sampled_times = torch.tensor(sampled_times, device=lambda_rnn_encs.device).unsqueeze(0).unsqueeze(-1)
        sampled_intensity_func = self.get_intensity_function(lambda_rnn_encs, sampled_times)
        # Traverse all test samples.
        for i in range(b_size):
            # To collect predictions of single sample.
            type_preds_i, type_probs_i, time_preds_i = [0], [0], [0]
            intens_func_i = []
            intens_func_i.append(sampled_intensity_func[i, -1,:,:].sum(-1).detach().cpu().tolist())
            # RNN inputs of i-th sample.
            time_rnn_inputs_i = time_rnn_inputs[i].unsqueeze(0)
            type_rnn_inputs_i = type_rnn_inputs[i].unsqueeze(0)
            lambda_rnn_inputs_i = lambda_rnn_inputs[i].unsqueeze(0)
            time_rnn_enc_i = time_rnn_encs[i].unsqueeze(0)
            type_rnn_enc_i = type_rnn_encs[i].unsqueeze(0)
            lambda_rnn_enc_i = lambda_rnn_encs[i].unsqueeze(0)
            intensity_masks = [torch.tensor([[[0.]]], device=lambda_rnn_enc_i.device)]
            # Initialize the first event and happening time.
            # Simulation for a single sample.
            while len(type_preds_i) < self.max_output_points+1:
                type_pred_prob_i, time_pred_i = type_rnn_enc_i[:,-1,:], time_rnn_enc_i[:,-1,:]
                time_pred_i = torch.softmax(self.time_predictor(time_pred_i), dim=-1)
                time_pred_i = torch.argmax(time_pred_i, dim=-1).item()
                type_pred_prob_i = torch.softmax(self.type_predictor(type_pred_prob_i), dim=-1).squeeze(0)
                type_pred_i = torch.argmax(type_pred_prob_i, dim=-1)
                time_preds_i.append(time_pred_i)
                type_preds_i.append(type_pred_i)
                type_probs_i.append(type_pred_prob_i.unsqueeze(0))
                pred_label_emb = self.label_embed(torch.tensor([type_preds_i[-1]], device=patient_rep.device).long())        # (1, h)
                # Add tim embedding.
                pred_time_emb = self.time_embed(torch.tensor([time_preds_i[-1]], device=patient_rep.device).long())
                next_type_input = torch.cat([patient_rep[i].unsqueeze(0).unsqueeze(1), 
                                                        pred_label_emb.unsqueeze(1)], 
                                                        dim=-1)
                next_time_input = torch.cat([patient_rep[i].unsqueeze(0).unsqueeze(1), 
                                                        pred_time_emb.unsqueeze(1)], 
                                                        dim=-1)
                time_rnn_inputs_i = torch.cat([time_rnn_inputs_i, next_time_input], dim=1)                     # (1, ls, 2h)
                type_rnn_inputs_i = torch.cat([type_rnn_inputs_i, next_type_input], dim=1)                     # (1, ls, 2h)
                lambda_rnn_inputs_i = time_rnn_inputs_i + type_rnn_inputs_i                                 # (1, ls, 2h)
                type_rnn_enc_i, _ = self.type_decoder(type_rnn_inputs_i, h0[:,i,:].unsqueeze(1))
                time_rnn_enc_i, _ = self.time_decoder(time_rnn_inputs_i, h0[:,i,:].unsqueeze(1))
                lambda_rnn_enc_i, _ = self.type_decoder(lambda_rnn_inputs_i, h0[:,i,:].unsqueeze(1))

                time_pred_tensor = torch.tensor([self.time_discrete_map[time_pred_i]], dtype=torch.float, device=time_rnn_inputs_i.device)                         # (1, ls, r_h)
                intensity_func_i = self.get_intensity_function(lambda_rnn_enc_i, time_pred_tensor.unsqueeze(1)) 
                intensity_masks.append(intensity_func_i[:,-1,:].squeeze(-1).unsqueeze(1))
                intensity_func_i_mask = torch.softmax(torch.cat(intensity_masks, dim=1), dim=1)
                type_rnn_enc_i = (type_rnn_enc_i*intensity_func_i_mask).sum(1, keepdim=True) 
                sampled_intensity_func_i = self.get_intensity_function(lambda_rnn_enc_i, sampled_times)  
                intens_func_i.append(sampled_intensity_func_i[0, -1,:,:].sum(-1).detach().cpu().tolist())
            returned_intensity_funcs.append(intens_func_i)
            type_preds.append(torch.tensor(type_preds_i[1:], dtype=torch.long, device=patient_rep.device).unsqueeze(0))
            type_probs.append(torch.cat(type_probs_i[1:], dim=0).unsqueeze(0))
            time_preds.append(torch.tensor(time_preds_i[1:], device=patient_rep.device).unsqueeze(0))

        type_preds = torch.cat(type_preds, dim=0)
        type_probs = torch.cat(type_probs, dim=0)
        time_preds = torch.cat(time_preds, dim=0)
        if return_int_func:
            return type_preds, type_probs, time_preds, returned_intensity_funcs
        return type_preds, type_probs, time_preds
