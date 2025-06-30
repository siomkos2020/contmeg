import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BaseMLP(nn.Module):
    def __init__(self,
                 num_diag,
                 num_med,
                 num_lab,
                 pat_emb_dim = 128,
                 lab_ts_dim = 150
                 ):
        super().__init__()
        self.diag_emb = nn.Embedding(num_diag, 64)
        self.diag_time_layer = nn.Linear(1, 16)
        self.med_emb = nn.Embedding(num_med, 64)
        self.med_time_layer = nn.Linear(1, 16)
        self.lab_type_emb = nn.Embedding(num_lab, 64)
        self.lab_time_layer = nn.Linear(1, 16)
        self.dense_layer = nn.Linear(4, 64)

        # Time series.
        self.ts_hidden_dim = 256
        self.batch_norm = nn.BatchNorm1d(num_features=lab_ts_dim)
        self.time_enc = nn.GRU(lab_ts_dim, self.ts_hidden_dim, batch_first=True)
        # self.lab_val_enc = nn.Sequential(
        #     nn.Linear(lab_ts_dim, self.ts_hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.ts_hidden_dim, self.ts_hidden_dim)
        # )
        self.final_diag_mlp = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, pat_emb_dim)
        )

        self.hidden_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64*4+16*3, pat_emb_dim),
            # nn.Dropout(0.8),
            # nn.ReLU(),
            # nn.Linear(256, pat_emb_dim),
            # nn.Dropout(0.8)
        )
        self.emb_drop = nn.Dropout(0.5)
    
    def forward(self, diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                lab_mask, demo):
        r"""
        Do patient representation for observed multi-channel time-aware features.
        diag_seq:    (b, l, num_diag), the past diagnosis ids sequence.  
        diag_time:   (b, l, ),         the past diagnosis time sequence.
        diag_mask:   (b, l, num_diag), the past diagnosis mask sequence.
        med_seq:     (b, l, num_med),  the past medication ids sequence.
        med_time:    (b, l, ),         the past medication time sequence.
        med_mask:    (b, l, num_med),  the past medication mask sequence.
        lab_seq:     (b, l, num_lab),  the past lab test ids sequence.
        lab_time:    (b, l, ),         the past lab test time sequence.
        lab_mask:    (b, l, num_lab),  the past lab test mask sequence.
        demo:        (b, demo_dim),    the past demographic features.
        label_seq:   (b, l,),          the future bad events sequence.
        
        return rep,  (b, h)
        """
        diag_len_mask = diag_mask.sum(-1)
        diag_len_mask[diag_len_mask > 0] = 1.
        diag_embs = self.emb_drop((self.diag_emb(diag_seq) * diag_mask.unsqueeze(-1)).mean(2))     # (b, l, emb)
        diag_embs = (diag_embs * diag_len_mask.unsqueeze(-1)).mean(1)         # (b, emb)

        med_len_mask = med_mask.sum(-1)
        med_len_mask[med_len_mask > 0] = 1.
        med_embs = self.emb_drop((self.med_emb(med_seq) * med_mask.unsqueeze(-1)).mean(2))     # (b, l, emb)
        med_embs = (med_embs * med_len_mask.unsqueeze(-1)).mean(1)              # (b, emb)

        lab_len_mask = lab_mask.sum(-1)
        lab_len_mask[lab_len_mask > 0] = 1.
        lab_embs = self.emb_drop((self.lab_type_emb(lab_seq[:,:,:,0]) * lab_mask.unsqueeze(-1)).mean(2))     # (b, l, emb)
        lab_embs = (lab_embs * lab_len_mask.unsqueeze(-1)).mean(1)                            # (b, emb)

        diag_time_vec = self.diag_time_layer(diag_time.unsqueeze(-1))           # (b, l, 16)
        diag_time_vec = (diag_time_vec * diag_len_mask.unsqueeze(-1)).mean(1)  # (b,  16)

        med_time_vec = self.med_time_layer(med_time.unsqueeze(-1))           # (b, l, 16)
        med_time_vec = (med_time_vec * med_len_mask.unsqueeze(-1)).mean(1)   # (b,  16)

        lab_time_vec = self.lab_time_layer(lab_time.unsqueeze(-1))           # (b, l, 16)
        lab_time_vec = (lab_time_vec * lab_len_mask.unsqueeze(-1)).mean(1)   # (b,  16)

        demo_vec = self.dense_layer(demo)
        # Time series encoding.
        # bsize, ls, num_lab = lab_time_series.size()
        # lab_ts_mask = torch.where(lab_time_series == -1, 0., 1.)
        # lab_time_series = lab_time_series.transpose(1, 2).view(bsize*num_lab, ls, -1)     # (b*num_lab, ls, 1)

        # Concat all feature embeddings.
        rep = torch.cat([diag_embs, diag_time_vec,
                        med_embs, med_time_vec,
                        lab_embs, lab_time_vec, demo_vec], dim=-1)    # (b, 64*4+16*3)
        rep = self.hidden_mlp(rep)  # (b, h)
        
        return rep


class ScaledDotAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.qnet = nn.Linear(input_dim, hidden_dim)
        self.knet = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, input_q, input_k, k_mask):
        r"""
        k_mask: (b, l,), positive value: 1., negative calue: 0.
        """
        q = self.qnet(input_q)  # (b, l, h)
        k = self.knet(input_k)  # (b, l, h)
        assert len(k_mask.size()) == 2
        k_mask = torch.where(k_mask.unsqueeze(-1) > 0.5, 0., -1e9)                      # (b, l, l)
        k_mask = torch.bmm(k_mask, k_mask.transpose(-1, -2))                            # (b, l, l)
        attn = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_dim)           # (b, l, l)
        attn = torch.softmax(attn+k_mask, dim=-1)
        out = torch.bmm(attn, input_k)  # (b, l, h)
        return out


class ConceptEmb(nn.Module):
    def __init__(self, num_emb, emb_dim, with_time = True):
        super().__init__()
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.emb_drop = nn.Dropout(0.5)
        if with_time:
            self.theta = nn.Embedding(num_emb, 1)
            self.mu = nn.Embedding(num_emb, 1)
        self.attn = ScaledDotAttn(emb_dim, emb_dim//2)
    
    def time_discount(self, input_ids, input_x, input_times, input_mask):
        r"""
        input_ids:   (b, n,)
        input_x:     (b, n, h)
        input_times: (b, )
        input_mask:  (b, n)
        """
        theta = self.theta(input_ids).squeeze(-1)                                   # (b, n,)
        mu = self.mu(input_ids).squeeze(-1)                                         # (b, n,)
        sx = torch.sigmoid(theta - mu * input_times.unsqueeze(1)).unsqueeze(-1)
        out = ((input_x * sx) * input_mask.unsqueeze(-1)).sum(1)                      # (b, h)
        return out

    def forward(self, input_ids, id_mask, input_times = None):
        r"""
        input_ids:  (b, ls, num_emb)
        id_mask:    (b, ls, num_emb)
        input_times:(b, ls)
        """
        bsize, ls, num_id = input_ids.size()
        id_embs = self.emb_drop(self.emb(input_ids))        # (b, ls, num_emb, h)
        id_embs = id_embs.view(bsize*ls, num_id, -1)        # (b*ls, num_emb, h)
        id_mask = id_mask.view(bsize*ls, num_id)            # (b*ls, num_emb, )
        id_embs = self.attn(id_embs, id_embs, id_mask)      # (b*ls, num_emb, h)
        if input_times is not None:
            id_embs = self.time_discount(input_ids.view(bsize*ls, num_id),
                                      id_embs,
                                      input_times.view(bsize*ls, ),
                                      id_mask)
        else:
            id_embs = id_embs.sum(1)
        embs = id_embs.reshape((bsize, ls, -1))
        return embs


class TimeLine(nn.Module):
    def __init__(self,
                 num_diag,
                 num_med,
                 num_lab,
                 pat_emb_dim 
                 ):
        super().__init__()
        self.pat_emb_dim = pat_emb_dim
        # Basic embedding layers.
        self.emb_modules = nn.ModuleList([ConceptEmb(num_diag, 64), 
                                          ConceptEmb(num_med, 64),
                                          ConceptEmb(num_lab, 64)])
        self.demo_layer = nn.Linear(4, pat_emb_dim)
        # GRUs for each feature sequence.
        self.rnns = nn.ModuleList([nn.GRU(64, pat_emb_dim, 
                                          batch_first=True,
                                          bidirectional=True) for _ in range(3)])
        # Outpur layer.
        self.out_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(7*pat_emb_dim, pat_emb_dim)
        )

    
    def forward(self, diag_seq, diag_time, diag_mask, med_seq, med_time, med_mask, lab_seq, lab_time,
                lab_mask, demo):
        r"""
        Do patient representation for observed multi-channel time-aware features.
        diag_seq:    (b, l, num_diag), the past diagnosis ids sequence.  
        diag_time:   (b, l, ),         the past diagnosis time sequence.
        diag_mask:   (b, l, num_diag), the past diagnosis mask sequence.
        med_seq:     (b, l, num_med),  the past medication ids sequence.
        med_time:    (b, l, ),         the past medication time sequence.
        med_mask:    (b, l, num_med),  the past medication mask sequence.
        lab_seq:     (b, l, num_lab),  the past lab test ids sequence.
        lab_time:    (b, l, ),         the past lab test time sequence.
        lab_mask:    (b, l, num_lab),  the past lab test mask sequence.
        demo:        (b, demo_dim),    the past demographic features.
        label_seq:   (b, l,),          the future bad events sequence.
        
        return rep,  (b, h)
        """
        lab_seq = lab_seq[:,:,:,0]      # Don't consider the lab test results.
        # Embed all clinical events into vectors.
        feat_hiddens = []
        b_size = diag_seq.size()[0]
        h0 = torch.zeros((2, b_size, self.pat_emb_dim), device=diag_seq.device)
        for i, (feat_ids, feat_mask, feat_time) in enumerate(zip([diag_seq, med_seq, lab_seq], 
                                                      [diag_mask, med_mask, lab_mask],
                                                      [diag_time, med_time, lab_time])):
            feat_len_mask = feat_mask.sum(-1)
            feat_len_mask[feat_len_mask > 0] = 1.
            feat_intervals = feat_time[:, -1].unsqueeze(-1) - feat_time
            feat_embs = self.emb_modules[i](feat_ids, feat_mask, feat_intervals)
            hiddens, _ = self.rnns[i](feat_embs, h0)
            hiddens = hiddens * feat_len_mask.unsqueeze(-1)
            feat_hiddens.append(hiddens.sum(1))

        # Static feature embeddings.
        demo_vec = self.demo_layer(demo)
        # Forward MHA.
        feat_hiddens = torch.cat([demo_vec] + feat_hiddens, dim=1)   # (b, 4*h)
        rep = self.out_layer(feat_hiddens)
        
        return rep
    

