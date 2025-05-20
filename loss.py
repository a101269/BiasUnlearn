# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random
import json
import numpy as np
import pandas as pd
import torch


def compute_weight():
    import numpy as np
    typecount = {'gender': 1522, 'profession': 4833, 'race': 5923, 'religion': 488}
    total = sum(typecount.values())
    sqrt_weights = {k: np.sqrt(total) / np.sqrt(v) for k, v in typecount.items()}
    sum_sqrt = sum(sqrt_weights.values())
    normalized_sqrt = {k: v / sum_sqrt for k, v in sqrt_weights.items()}
    print(sqrt_weights)     #{'gender': 2.90, 'profession': 1.62, 'race': 1.47, 'religion': 5.11}
    print( normalized_sqrt)    #{'gender': 0.26, 'profession': 0.15, 'race': 0.13, 'religion': 0.46}


class DynamicWeightAdapter:
    def __init__(self, initial_weights, bia_type, momentum=0.9):
        self.weights=[]
        for k in bia_type:
            self.weights.append(initial_weights[k])
        self.momentum = momentum
        self.loss_history = {k: 0 for k in initial_weights}
        self.bia_type = ['gender', 'profession', 'race', 'religion']

    def update(self, current_losses):
        for k in current_losses:
            self.loss_history[k].append(current_losses[k])
        if len(self.loss_history[k])>2:
            self.loss_history[k].pop(0)

        ema_losses = {}
        for k in self.loss_history:
            if len(self.loss_history[k]) == 0:
                ema_losses[k] = 0
                continue
            ema = (1 - self.momentum) * current_losses[k]
            if len(self.loss_history[k]) > 1:
                ema += self.momentum * self.loss_history[k][-2]
            ema_losses[k] = ema

        avg_loss = sum(ema_losses.values()) / len(ema_losses)
        for k in ema_losses:
            self.weights[k] *= (ema_losses[k] / (avg_loss + 1e-6))

        return self.weights



def compute_kl(pretrained_model, current_model, batch, device):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss


def lm_loss(operation, batch, model, pad_id=-100, device="cuda:0", weight=None):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"].to(device),
        batch["labels"].to(device),
    )
    seq_len = labels.size(1)
    col_indices = torch.arange(seq_len, device=labels.device).expand_as(labels)
    mask = col_indices < start_locs.unsqueeze(1)  # -> (batch_size, 1)
    modified_label = labels.clone()
    modified_label[mask] = -100

    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    # loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = modified_label[:, 1:] # labels[:, 1:]
    # shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):


        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        # one_inp, one_st = input_ids[bid], start_locs[bid]
        # position_weight = torch.zeros_like(one_inp)
        # assert len(position_weight) == len(position_loss) + 1
        # position_weight[one_st:] = 1  # only focus on answer part
        # print(position_weight)
        # print(position_weight[one_st:].shape)
        # Ignore the padding part.
        # print(one_inp)
        # position_weight[one_inp == pad_id] = 0
        # print(pad_id)
        # print(position_weight)
        # print('\n\n')
        # if position_weight.sum() > 0:
        #     position_weight = position_weight / position_weight.sum()
        # one_loss = (position_weight[:-1] * position_loss).sum()
        # losses.append(one_loss)
        if weight:
            position_loss*=weight[batch["bias_type"][bid]]
        losses.append(position_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss

