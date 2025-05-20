# encoding:utf-8
#    Date  :  2025/2/28
import copy
import argparse
import logging
import random
import time
import torch.nn.functional as F
import numpy as np
import torch
from itertools import cycle
from tqdm import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from dataloader import get_stereoset_answers_plaintext
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingLR, LinearLR
from loss import lm_loss, compute_kl, DynamicWeightAdapter
import yaml

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

target_map = {"gpt2": ["attn.c_proj", "attn.c_attn"],
              "qwen": ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
              "meta": ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
              "llama": ['up_proj', 'k_proj', 'o_proj', 'gate_proj', 'q_proj', 'v_proj', 'down_proj']
              }
bia_type = ['gender', 'profession', 'race', 'religion']
type2id = {'gender': 0, 'profession': 1, 'race': 2, 'religion': 3}


def main(args) -> None:

    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=3)#,offload_optimizer_device="cpu")
    # deepspeed_plugin = DeepSpeedPlugin(
    #     zero_stage=3,
    #     zero3_param_persistence_threshold= {"params": ["*.embedding.weight"], "no_shard": True}
    # )
    # ds_config = {
    #     "zero_optimization": {
    #         "stage": 3,
    #         "exclude_embedding": True,  # 关键配置：排除 Embedding 层
    #     },
        # 其他配置...
    # }
    deepspeed_plugin1 = DeepSpeedPlugin()
    deepspeed_plugin2 = DeepSpeedPlugin()
    plugins = {"student": deepspeed_plugin1 , "teacher": deepspeed_plugin2}

    accelerator = Accelerator(deepspeed_plugin=plugins)

    device = accelerator.device
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

    # logging.info("tokenizer.pad_token_id")
    # print("tokenizer.pad_token_id")
    # logging.info(tokenizer.pad_token_id)
    # print(tokenizer.pad_token_id)
    # Load  data.
    anti_dataloader, ster_dataloader, unrelate_dataloader = get_stereoset_answers_plaintext(tokenizer,
                                                                                            mix_anti=args.mix_anti,
                                                                                            ster_batch_size=args.ster_batch_size,
                                                                                            batch_size=args.batch_size,
                                                                                            type2id=type2id)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    # optimizer2 = AdamW(model.parameters(), lr=args.lr)
    # from torch.optim.lr_scheduler import LambdaLR
    # lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    # from torch.optim.lr_scheduler import CosineAnnealingLR
    # min_lr = 1e-6  # 最小学习率
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=3000, eta_min=min_lr)
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
        num_training_steps = num_training_steps* args.epoch, #llama3
    )

    # lr_scheduler = get_scheduler( #cosine
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps= 10 ,
    #     num_training_steps = num_training_steps*1.2, #GPT-large
    # )

    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps= 10 ,
    #     num_training_steps = len(ster_dataloader) * 3, #GPT-mid
    # )

    typecount = {'gender': 1522, 'profession': 4833, 'race': 5923, 'religion': 488}
    init_weight = {'gender': 0.26, 'profession': 0.15, 'race': 0.13, 'religion': 0.46}
    weight_adapter = DynamicWeightAdapter(init_weight, bia_type)

    print("anti_dataloader get_stereoset_answers_plaintext")
    print(anti_dataloader)

    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    model, optimizer,  anti_dataloader, ster_dataloader, unrelate_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer,  anti_dataloader, ster_dataloader, unrelate_dataloader, lr_scheduler,
    )

    ref_model= accelerator.prepare(ref_model)

    print("anti_dataloader accelerator.prepare")
    print(anti_dataloader)


    # deepspeed_plugin1.enable()

    # Reference model for computing KL.
    model.train()

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
    tmp = args.save_every
    # while ster_loss < args.max_ster_loss and idx < args.max_unlearn_steps:
    while epoch < args.epoch and idx < args.max_unlearn_steps:
        epoch += 1
        for star_batch, anti_batch, unrelate_batch in zip(ster_dataloader, cycle(anti_dataloader),
                                                          cycle(unrelate_dataloader)):  # 12766/4/4=797 epoch
            # if idx > 800 and idx < 1600:
            #     args.lr = 3e-6
            # elif idx > 1600:
            #     args.lr = 2e-6

            if idx % 100 == 0 and args.reinit:
                accelerator.free_memory(optimizer)
                del optimizer
                optimizer = AdamW(model.parameters(), lr=args.lr)
                num_training_steps = args.max_unlearn_steps
                lr_scheduler = get_scheduler(
                    name="linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps,
                )
                (optimizer, lr_scheduler) = accelerator.prepare(optimizer, lr_scheduler)

            # ster_loss = lm_loss("ga", star_batch, model, device=device, pad_id=tokenizer.pad_token_id)
            ster_loss = lm_loss("gd", star_batch, model, device=device, pad_id=tokenizer.pad_token_id,
                                weight=None)#weight_adapter.weights
            ster_loss_ref = lm_loss("gd", star_batch, ref_model, device=device, pad_id=tokenizer.pad_token_id)
            neg_log_ratio = ster_loss - ster_loss_ref
            loss_npo = -F.logsigmoid(args.beta * neg_log_ratio).mean() * 2 / args.beta

            # 在数据加载阶段添加遗忘标记
            # forget_mask = (ster_loss.detach() > 2.0).float()  # [batch_size]
            # loss_npo = (-F.logsigmoid(args.beta * neg_log_ratio) * forget_mask).mean() * 2 / args.beta


            # kl_loss = compute_kl(ref_model, model, unrelate_batch, device)
            # loss = loss_npo * args.ster_weight + args.kl_weight * kl_loss

            anti_loss = lm_loss("gd", anti_batch, model, device=device, pad_id=tokenizer.pad_token_id,
                                weight=None)  # weight_adapter.weights
            kl_loss = compute_kl(ref_model, model, unrelate_batch, device)
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

            if idx>400 and idx<500:

                args.save_every=20
            else:
                args.save_every=tmp
            if idx % args.save_every == 0 or idx == 100:
                try:
                    model.save_pretrained("save/"+args.model_save_dir + "/"+args.model_save_dir + "_" + str(idx), safe_serialization=False)
                except:
                    model.module.save_pretrained("save/"+args.model_save_dir + "/"+args.model_save_dir + "_" + str(idx), safe_serialization=False)
                logging.info(f"saved model at step {idx}")
                print(f"saved model at step {idx}")
            if args.anti_weight * anti_loss < min_anti_loss:
                stop_FLAG = True
                break
        if stop_FLAG:
            break
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    # if args.use_lora:
    #     model = model.merge_and_unload()

    # Save final model.

    # try:
    #     model.save_pretrained(args.model_save_dir, safe_serialization=False)
    # except:
    #     model.module.save_pretrained(args.model_save_dir, safe_serialization=False)
    logging.info("Unlearning finished")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_file", type=str, help="config file",)
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
        "--save_every", type=int, default=800, help="How many steps to save model."
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


