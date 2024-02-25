import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps) 

def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps) 

def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean() 

def intra_class_relation(y_s, y_t):
    return inter_class_relation((y_s.T).softmax(dim=1), (y_t.T).softmax(dim=1))

def dist(z_s, z_t, args):
    inter_loss = inter_class_relation((z_s/args.s_temp).softmax(dim=1), (z_t/args.t_temp).softmax(dim=-1))
    intra_loss = intra_class_relation(z_s/args.s_temp, z_t/args.t_temp)
    kd_loss = args.l_0 * inter_loss + args.l_1 * intra_loss 
    return kd_loss

def dino(t_feat, s_feat, args, reduction='mean'):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        s_feat = s_feat / args.s_temp
        t_feat = t_feat / args.t_temp

        targets = F.softmax(t_feat, dim=-1)
        logprobs = F.log_softmax(s_feat.view(s_feat.shape[0], -1), dim=1)
        batchloss = - torch.sum(targets.view(targets.shape[0], -1) * logprobs, dim=1)
        if reduction == 'none':
            return batchloss
        elif reduction == 'mean':
            return torch.mean(batchloss)
        elif reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')

def dinoss(t_feat, s_feat, args, reduction='mean'):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        def ce(tf, sf, args, reduction):
            sf = sf / args.s_temp
            tf = tf / args.t_temp

            targets = F.softmax(tf, dim=-1)
            logprobs = F.log_softmax(sf.view(s_feat.shape[0], -1), dim=1)
            batchloss = - torch.sum(targets.view(targets.shape[0], -1) * logprobs, dim=1)
            if reduction == 'none':
                return batchloss
            elif reduction == 'mean':
                return torch.mean(batchloss)
            elif reduction == 'sum':
                return torch.sum(batchloss)
            else:
                raise NotImplementedError('Unsupported reduction mode.')
        return ce(t_feat, s_feat, args, reduction) + 0.5*ce(t_feat.t(), s_feat.t(), args, reduction)


def coss(t_feat, s_feat, args, **kwargs):
    def dot_p(t_feat, s_feat, args):
        tf = F.normalize(t_feat, dim=-1, p=2)/args.t_temp
        sf = F.normalize(s_feat, dim=-1, p=2)/args.s_temp
        #check random nans in vit training
        batchloss = -(tf * sf).sum(dim=-1)
        return batchloss
    
    return args.l_0 * torch.mean(dot_p(t_feat, s_feat, args)) + args.l_1 * torch.mean(dot_p(t_feat.T, s_feat.T, args))
