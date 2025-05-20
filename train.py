# encoding:utf-8
#    Date  :  2025/2/28

import argparse
import logging
import random
import time
import torch.nn.functional as F
import numpy as np
import torch
import os
import json
from itertools import cycle
from tqdm import tqdm
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from dataloader import get_stereoset_answers_plaintext
from Evaluator import BiasEvaluator,ScoreEvaluator
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingLR,LinearLR
from loss import lm_loss, compute_kl, DynamicWeightAdapter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
accelerator = Accelerator()
device = accelerator.device

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

target_map = {"gpt2": ["attn.c_proj", "attn.c_attn"],
              "qwen": ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
              "meta": ['up_proj', 'k_proj', 'o_proj', 'gate_proj', 'q_proj', 'v_proj', 'down_proj']}
bia_type = ['gender', 'profession', 'race', 'religion']
type2id = {'gender': 0, 'profession': 1, 'race': 2, 'religion': 3}

def eval(evaluator,model):
    results = evaluator.evaluate(model)
    score_evaluator = ScoreEvaluator(results)

    overall = score_evaluator.get_overall_results()
    res=score_evaluator.pretty_print(overall)
    return res

def train(first_turn,e1poch,idx,ster,anti,unrelate,model,tokenizer,device,pretrained_model,evaluator, optimizer, lr_scheduler,pre_res):
    stop_FLAG=False
    # while epoch < args.epoch and idx < args.max_unlearn_steps:
    #     epoch += 1
    while idx < args.max_unlearn_steps:
        for star_batch, anti_batch, unrelate_batch in zip(ster, cycle(anti),cycle(unrelate)):  # 12766/4/4=797 epoch
            # ster_loss = lm_loss("ga", star_batch, model, device=device, pad_id=tokenizer.pad_token_id)
            ster_loss = lm_loss("gd", star_batch, model, device=device, pad_id=tokenizer.pad_token_id,
                                weight=None)#weight_adapter.weights
            ster_loss_ref = lm_loss("gd", star_batch, pretrained_model, device=device, pad_id=tokenizer.pad_token_id)
            neg_log_ratio = ster_loss - ster_loss_ref
            loss_npo = -F.logsigmoid(args.beta * neg_log_ratio).mean() * 2 / args.beta

            # 在数据加载阶段添加遗忘标记
            # forget_mask = (ster_loss.detach() > 2.0).float()  # [batch_size]
            # loss_npo = (-F.logsigmoid(args.beta * neg_log_ratio) * forget_mask).mean() * 2 / args.beta


            # kl_loss = compute_kl(pretrained_model, model, unrelate_batch, device)
            # loss = loss_npo * args.ster_weight + args.kl_weight * kl_loss

            anti_loss = lm_loss("gd", anti_batch, model, device=device, pad_id=tokenizer.pad_token_id,
                                weight=None) #weight_adapter.weights
            kl_loss = compute_kl(pretrained_model, model, unrelate_batch, device)
            # loss2 = args.anti_weight * anti_loss + args.kl_weight * kl_loss

            loss = args.ster_weight * loss_npo + args.anti_weight * anti_loss + args.kl_weight * kl_loss

            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            current_lr = lr_scheduler.get_last_lr()[0]

            # print("current_lr",current_lr)
            stats = (
                f"batch: {idx}, "
                f"lr: {current_lr:.8f} "
                f"ster_loss: {args.ster_weight * ster_loss:.3f}, "
                f"anti_loss: {args.anti_weight * anti_loss:.3f}, "
                f"current_div_loss: {args.kl_weight * kl_loss:.3f}, "
            )
            logging.info(stats)
            print(stats)
            idx += 1

            # Save model.

            if first_turn :
                if idx % args.save_every == 0 and idx >= 100:
                    res = eval(evaluator, model)
                    if idx >= 100:
                        return res, idx
            else:
                if idx % args.eval_every == 0:
                    try:
                        model.save_pretrained("save/" + args.model_save_dir + "/" + args.model_save_dir + "_" + str(idx),safe_serialization=False)
                    except:
                        model.module.save_pretrained("save/" + args.model_save_dir + "/" + args.model_save_dir + "_" + str(idx),safe_serialization=False)
                    logging.info("pre_res")
                    print("pre_res")
                    logging.info(pre_res)
                    print(pre_res)
                    res = eval(evaluator, model)
                    return res,idx
            # if idx % args.save_every == 0:
            #     try:
            #         model.save_pretrained("save/"+args.model_save_dir + "/"+args.model_save_dir + "_" + str(idx), safe_serialization=False)
            #     except:
            #         model.module.save_pretrained("save/"+args.model_save_dir + "/"+args.model_save_dir + "_" + str(idx), safe_serialization=False)
            #     logging.info(f"saved model at step {idx}")
            #     print(f"saved model at step {idx}")
    res = eval(evaluator, model)
    return res, idx

def main(args) -> None:
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # use LoRA.
    name_head = args.model_name.split("/")[-1].split('-')[0].lower()
    logging.info("name_head:")
    print("name_head:", name_head)
    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,  # 训练模式
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_map[name_head]
        )
        model = get_peft_model(model, peft_config)

    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    has_pad = tokenizer.pad_token is not None
    if not has_pad:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    evaluator = BiasEvaluator(tokenizer=tokenizer)


    anti_dataloader, ster_dataloader, unrelate_dataloader, ster_race_dataloader,ster_gender_dataloader,ster_profession_dataloader,ster_religion_dataloader,anti_race_dataloader,anti_gender_dataloader,anti_profession_dataloader,anti_religion_dataloader = get_stereoset_answers_plaintext(tokenizer,
                                                                                            mix_anti=args.mix_anti,
                                                                                            ster_batch_size=args.ster_batch_size,
                                                                                            batch_size=args.batch_size,
                                                                                            type2id=type2id)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    device_count = torch.cuda.device_count()
    num_training_steps = min(args.max_unlearn_steps * device_count, len(ster_dataloader) * args.epoch)
    logging.info("num_training_steps,"+str(num_training_steps))
    print("num_training_steps,"+str(num_training_steps))
    logging.info("len(ster_dataloader) * args.epoch"+str(len(ster_dataloader) * args.epoch))
    print("len(ster_dataloader) * args.epoch" + str(len(ster_dataloader) * args.epoch)) # 12912


    lr_scheduler = get_scheduler( #cosine
        name="linear",
        optimizer=optimizer,
        num_warmup_steps= 10 ,
        num_training_steps = num_training_steps*1.2,
    )

    typecount = {'gender': 1522, 'profession': 4833, 'race': 5923, 'religion': 488}
    init_weight = {'gender': 0.26, 'profession': 0.15, 'race': 0.13, 'religion': 0.46}
    weight_adapter = DynamicWeightAdapter(init_weight, bia_type)

    print("anti_dataloader get_stereoset_answers_plaintext")
    print(anti_dataloader)


    model, optimizer,  anti_dataloader, ster_dataloader, unrelate_dataloader, ster_race_dataloader,ster_gender_dataloader,ster_profession_dataloader,ster_religion_dataloader,anti_race_dataloader,anti_gender_dataloader,anti_profession_dataloader,anti_religion_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer,  anti_dataloader, ster_dataloader, unrelate_dataloader, ster_race_dataloader,ster_gender_dataloader,ster_profession_dataloader,ster_religion_dataloader,anti_race_dataloader,anti_gender_dataloader,anti_profession_dataloader,anti_religion_dataloader, lr_scheduler
    )

    category_map = {
        "ster": {
            "race": ster_race_dataloader,
            "gender": ster_gender_dataloader,
            "profession": ster_profession_dataloader,
            "religion": ster_religion_dataloader
        },
        "anti": {
            "race": anti_race_dataloader,
            "gender": anti_gender_dataloader,
            "profession": anti_profession_dataloader,
            "religion": anti_religion_dataloader
        }
    }

    print("anti_dataloader accelerator.prepare")
    print(anti_dataloader)
    model.train()

    # Reference model for computing KL.

    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    pretrained_model.to(device)
    pretrained_model.eval()
    # Start unlearning.
    ster_loss = 0.0
    idx = 0
    epoch = 0
    min_anti_loss = 0.1
    stop_FLAG = False
    start_time = time.time()
    # Stop if bad loss is big enough or reaching max step.

    print("cycle(anti_dataloader)")
    print(anti_dataloader)
    print(cycle(anti_dataloader))
    logging.info("len ster_dataloader")
    logging.info(len(ster_dataloader))
    print(ster_dataloader)
    current_ster = ster_dataloader
    current_anti = anti_dataloader
    reverse=False
    first_turn=True
    pre_res=None
    while idx < args.max_unlearn_steps :
        if reverse:
            scores, idx = train(first_turn, epoch, idx, current_anti, current_ster, unrelate_dataloader, model, tokenizer, device, pretrained_model, evaluator, optimizer, lr_scheduler,pre_res)
        else:
            scores, idx = train(first_turn, epoch, idx, current_ster, current_anti, unrelate_dataloader, model,tokenizer, device, pretrained_model, evaluator, optimizer, lr_scheduler, pre_res)

        pre_res=scores
        bases = [49,49,49,48]

        gaps = [abs(float(score) - bases[i]) for i, score in enumerate(scores[:4])]
        if all(gap < 2 for gap in gaps):
            try:
                model.save_pretrained("save/" + args.model_save_dir + "/" + args.model_save_dir + "_" + str(idx), safe_serialization=False)
            except:
                model.module.save_pretrained("save/" + args.model_save_dir + "/" + args.model_save_dir + "_" + str(idx), safe_serialization=False)
            logging.info(f"saved model at step {idx}")
            end_time = time.time()
            logging.info("Total time: %d sec" % (end_time - start_time))

            # if args.use_lora:
            #     model = model.merge_and_unload()
            logging.info("Unlearning finished")
            return

        max_gap_idx = gaps.index(max(gaps))
        selected_category = bia_type[max_gap_idx]
        if float(scores[max_gap_idx])< bases[max_gap_idx]:
            reverse = True
        else:
            reverse = False
        if max(gaps)>5:
            fineturn_lr=args.lr
            fineturn_steps=3000 if idx<400 else 1000
            args.eval_every=80
        elif 3<max(gaps)<=5:
            fineturn_lr = args.lr*4/5
            fineturn_steps=200
            args.eval_every = 50
        elif max(gaps)<=3:
            fineturn_lr = args.lr*4/5
            fineturn_steps = 200
            args.eval_every = 30

        current_ster = category_map['ster'][selected_category]
        current_anti = category_map['anti'][selected_category]
        first_turn=False

        accelerator.free_memory(optimizer)
        # args.save_every=10
        del optimizer
        optimizer = AdamW(model.parameters(), lr=fineturn_lr)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=5,
            num_training_steps=fineturn_steps,
        )
        (optimizer, lr_scheduler) = accelerator.prepare(optimizer, lr_scheduler)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--reinit", action="store_true")
    parser.add_argument("--mix_anti", action="store_true")
    parser.add_argument("--epoch", type=int, default=3)

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument("--beta", type=float, default=0.1, help="Weight on the bad loss.")
    parser.add_argument(
        "--ster_weight",
        type=float,
        default=1,
        help="Weight on the bad loss."
    )
    parser.add_argument(
        "--anti_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.2,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--ster_batch_size", type=int, default=16, help="Batch size of unlearning."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Unlearning LR.")
    parser.add_argument(
        "--max_ster_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=100, help="How many steps to save model."
    )
    parser.add_argument(
        "--eval_every", type=int, default=30, help="How many steps to save model."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)

