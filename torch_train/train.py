import os
import argparse
import datetime

import tensorflow.compat.v1 as tf
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from transformers import T5ForConditionalGeneration, AutoConfig
from transformers import AutoTokenizer

import sys
sys.path.append('../torch_datasets')
from splinter_dataset import SplinterDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train ArgParser.')
    parser.add_argument('--train_dataset', type=str, required=True,
                        help='train dataset')
    parser.add_argument('--eval_dataset', type=str, required=True,
                        help='development dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--checkpoint', type=str, default=None, required=False,
                        help='pytorch checkpoint')
    parser.add_argument('--from_pretrained', type=bool, default=False, required=False,
                        help='huggingface checkpoint')
    parser.add_argument('--cache_dir', type=str, default=None, required=False,
                        help='cache_dir fo huggingface data')
    parser.add_argument('--config', type=str, default='t5-small', required=False,
                        help='huggingface config')
    parser.add_argument('--train_batch_size', type=int, default=2, required=False,
                        help='train dataset batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, required=False,
                        help='number of steps between optimizer steps')
    parser.add_argument('--eval_batch_size', type=int, default=1, required=False,
                        help='evaluation dataset batch size')
    parser.add_argument('--skip_eval', type=int, default=0, required=False,
                        help='skip evaluation on checkpoint')
    parser.add_argument('--save_best', type=int, default=1, required=False,
                        help='skip evaluation on checkpoint')
    parser.add_argument('--save_latest', type=int, default=1, required=False,
                        help='skip evaluation on checkpoint')
    parser.add_argument('--save_step', type=int, default=0, required=False,
                        help='save a checkpoint for each step')
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False,
                        help='initial learning rate')
    parser.add_argument('--num_train_steps', type=int, default=200000, required=False,
                        help='number of training steps')
    parser.add_argument('--warmup_steps', type=int, default=10000, required=False,
                        help='number of warmup steps from training steps')
    parser.add_argument('--save_checkpoints_steps', type=int, default=1000, required=False,
                        help='number of steps between saving and evaluating checkpoints')
    parser.add_argument('--gamma', type=float, default=0.9, required=False,
                        help='gamma factor for ExponentialLR')
    parser.add_argument('--num_workers', type=int, default=0, required=False,
                        help='number of workers for dataloaders')
    parser.add_argument('--schedule_steps', type=int, default=1000, required=False,
                        help='number of steps in epoch')
    parser.add_argument('--print_loss_steps', type=int, default=0, required=False,
                        help='number of steps for printing the train loss')
    args = parser.parse_args()
    return args


def create_dataloaders(args):
    train_dataset = SplinterDataset(args.train_dataset)
    eval_dataset = SplinterDataset(args.eval_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    tf.print(f'Created train dataset ({len(train_dataset)} examples)')

    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    tf.print(f'Created eval dataset ({len(eval_dataset)} examples)')

    return train_dataloader, eval_dataloader

# TODO add F1 score
def eval(model, eval_dataloader, device):
    tf.print('Evaluating')
    total_loss, count = 0.0, 0.0
    for i, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        total_loss += loss.item()
        count += 1

    assert(count > 0)
    avg_loss = total_loss / count
    return avg_loss


def checkpoint(output_dir, writer, step, train_loss, model,
               device, eval_dataloader, scheduler, optimizer,
               best, args):
    eval_loss = float('Inf')
    if args.skip_eval == False:
        model.eval()
        eval_loss = eval(model, eval_dataloader, device)
        model.train()
        tf.print(f"step #{step} eval loss: {eval_loss}")
        writer.add_scalar("Loss/eval", eval_loss, step)

    tf.print(f"step #{step} train loss: {train_loss}")
    writer.add_scalar("Loss/train", train_loss, step)

    ckpt = {
        'step': step,
        'best': best,
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'model': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if args.save_best == 1 and eval_loss <= best:
        tf.print('Saving best')
        best = eval_loss
        checkpoint_path = os.path.join(checkpoint_dir, f'best.pt')
        torch.save(ckpt, checkpoint_path)

    if args.save_latest == 1:
        tf.print('Saving latest')
        checkpoint_path = os.path.join(checkpoint_dir, f'latest.pt')
        torch.save(ckpt, checkpoint_path)

    if args.save_step == 1:
        tf.print('Saving checkpoint')
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{step}.pt')
        torch.save(ckpt, checkpoint_path)

    return best

def load_checkpoint(checkpoint_path, model, scheduler, optimizer):
    tf.print('Loading from checkpoint')
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model'])
    scheduler.load_state_dict(ckpt['scheduler']) # TODO - not sure this is okay! test it.
    optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt['step'], ckpt['best']


def create_output_dirs(output_dir):
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


def main(args):
    # TODO add graph of loss?
    create_output_dirs(args.output_dir)
    writer = SummaryWriter(log_dir=args.output_dir, max_queue=1)

    train_dataloader, eval_dataloader = create_dataloaders(args)

    if args.from_pretrained:
        tf.print(f'Loading pretrained: {args.config}')
        model = T5ForConditionalGeneration.from_pretrained(args.config, cache_dir=args.cache_dir)
    else:
        tf.print(f'Initializing random: {args.config}')
        t5config = AutoConfig.from_pretrained(args.config, cache_dir=args.cache_dir)
        model = T5ForConditionalGeneration(t5config)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    total_epoch = int(args.warmup_steps / args.schedule_steps) + 1 # cannot be 0
    scheduler_wrap = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=total_epoch, after_scheduler=scheduler)

    best = float('Inf')
    steps = 0
    if args.checkpoint:
        steps, best = load_checkpoint(args.checkpoint, model, scheduler_wrap, optimizer)
    virtual_epoch = int(steps / args.schedule_steps) + 1
    scheduler_wrap.step(virtual_epoch) # not always correct..

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()

    tf.print('Starting train loop')
    while steps <= args.num_train_steps:
        for batch in train_dataloader:
            if steps >= args.num_train_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0] / args.gradient_accumulation_steps
            loss.backward()

            if (steps + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if steps % args.save_checkpoints_steps == 0:
                best = checkpoint(
                    args.output_dir, writer, steps, loss.item(),
                    model, device, eval_dataloader,
                    scheduler_wrap, optimizer,
                    best, args
                )
            if int(steps / args.schedule_steps) + 1 > virtual_epoch:
                virtual_epoch = int(steps / args.schedule_steps) + 1
                scheduler_wrap.step(virtual_epoch)
            if args.print_loss_steps and (steps + 1) % args.print_loss_steps == 0:
                timestamp = datetime.datetime.now()
                tf.print(f"{timestamp} step: #{steps}, virtual_epoch: {virtual_epoch}, "
                      f"train_loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}")
            steps += 1

    tf.print('Done Training')
    best = checkpoint(
        args.output_dir, steps, 0,
        model, device, eval_dataloader,
        scheduler_wrap, optimizer,
        best, args
    )
    tf.print(f'Best Evalusation loss {best}')
    writer.flush()


if __name__ == '__main__':
    tf.print('Started')
    args = parse_args()
    tf.print(args)
    main(args)
