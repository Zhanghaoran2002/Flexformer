import os
import sys
import numpy as np
import random
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from listops.input_torch import get_datasets
from models.transformer import TransformerEncoder
from models.flexformer import FlexformerEncoder
from utils.train_utils import compute_weighted_cross_entropy, compute_weighted_accuracy, create_learning_rate_scheduler
from torch.optim.lr_scheduler import LambdaLR


def train_step(model, optimizer, batch, device):
    model.train()
    inputs = batch['inputs'].to(device)
    targets = batch['targets'].to(device)
    optimizer.zero_grad()
    logits = model(inputs)
    loss = compute_weighted_cross_entropy(logits, targets, num_classes=10)
    loss.backward()
    optimizer.step()
    acc = compute_weighted_accuracy(logits, targets)
    return loss.item(), acc.item()

@torch.no_grad()
def eval_step(model, batch, device):
    model.eval()
    inputs = batch['inputs'].to(device)
    targets = batch['targets'].to(device)
    logits = model(inputs)
    loss = compute_weighted_cross_entropy(logits, targets, num_classes=10)
    acc = compute_weighted_accuracy(logits, targets)
    return loss.item(), acc.item()

def main(args):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_train_steps': args.num_train_steps,
        'num_eval_steps': args.num_eval_steps,
        'eval_frequency': args.eval_frequency,
        'model_dir': args.model_dir,
        'data_dir': args.data_dir,
        'task_name': args.task_name,
        'max_length': args.max_length,
        'emb_dim': args.emb_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'qkv_dim': args.qkv_dim,
        'mlp_dim': args.mlp_dim,
        'classifier': args.classifier,
        'num_classes': args.num_classes,
        'dropout_rate': args.dropout_rate,
        'attention_dropout_rate': args.attention_dropout_rate,
        'factors': 'constant * linear_warmup * rsqrt_decay', 
        'base_learning_rate': args.learning_rate,
        'warmup_steps': args.warmup,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, vocab = get_datasets(
        n_devices=1,
        task_name=config['task_name'],
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )
    vocab_size = len(vocab)
    print("Size of vocabulary:",vocab_size)

    if args.model_name.lower()=='transformer':
        model = TransformerEncoder(
            vocab_size=vocab_size,
            emb_dim=config['emb_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            qkv_dim=config['qkv_dim'],
            mlp_dim=config['mlp_dim'],
            max_len=config['max_length'],
            classifier=config['classifier'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            attention_dropout_rate=config['attention_dropout_rate']
        ).to(device)
    elif args.model_name.lower()=='flexformer':
        model = FlexformerEncoder(
            vocab_size=vocab_size,
            emb_dim=config['emb_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            qkv_dim=config['qkv_dim'],
            mlp_dim=config['mlp_dim'],
            num_freq=args.num_freq,
            max_len=config['max_length'],
            classifier=config['classifier'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate'],
            attention_dropout_rate=config['attention_dropout_rate']
        ).to(device)
    print("Number of trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam(
        model.parameters(), 
        lr=1.0,  
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay
    )
    lr_lambda = create_learning_rate_scheduler(
        factors=config['factors'],
        base_learning_rate=config['base_learning_rate'],
        warmup_steps=config['warmup_steps'],
    )
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    check_path = 'checkpoints/'+f'{args.model_dir}/'
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    save_path = 'results/'+f'{args.model_dir}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    istest = 1 if args.mode=='test' else 0
        
    if args.mode=='train':
        min_eval_loss = float('inf')
        step = 0
        earlystop = 0
        starttime = time.time() 
        train_loss = []
        for epoch in range(args.epochs): 
            print("\nTraining Epoch: {}/{}".format(epoch + 1, args.epochs))
            for batch in train_loader:
                loss, acc = train_step(model, optimizer, batch, device)
                train_loss.append(loss)
                step += 1
                scheduler.step()
    
                if step % config['eval_frequency'] == 0:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("\n" + current_time)
                    timecost=time.time()-starttime
                    speed=args.eval_frequency/timecost
                    print(f"Training {args.eval_frequency} iters spends time {timecost:.4f}s, speed {speed:.4f}iters/s")
                    mean_train_loss = sum(train_loss)/len(train_loss)
                    print(f"Training loss:{loss:.4f}")
                    
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Adjusting learning rate to:{current_lr:.6f}")
                    print("Model evaluating")
                    eval_losses, eval_accs = [], []
                    for i, batch in enumerate(val_loader):
                        if i >= config['num_eval_steps']:
                            break
                        eloss, eacc = eval_step(model, batch, device)
                        eval_losses.append(eloss)
                        eval_accs.append(eacc)
                    mean_loss = sum(eval_losses) / len(eval_losses)
                    mean_acc = sum(eval_accs) / len(eval_accs)
                    # writer.add_scalar('eval/loss', mean_loss, step)
                    # writer.add_scalar('eval/accuracy', mean_acc, step)
                    print(f"Step {step}: Eval Loss {mean_loss:.4f}, Eval Acc {mean_acc:.4f}")
                    train_loss = []
                    if mean_loss<min_eval_loss:
                        # torch.save(model.state_dict(), os.path.join(check_path, f"model_step_{step}.pt"))
                        torch.save(model.state_dict(), os.path.join(check_path, "checkpoint.pt"))
                        min_eval_loss=mean_loss
                        earlystop=0
                    else:
                        earlystop+=1
                        print("Early stopping: ", f"{earlystop}/{args.patience}")
        
                    if step >= config['num_train_steps'] or earlystop>=args.patience:
                        break
                    starttime=time.time()
            if step >= config['num_train_steps'] or earlystop>=args.patience:
                break
        istest = 1
    
    if istest:
        print("Testing")
        model.train()
        dummy_input = torch.randint(0, vocab_size, (1, args.max_length-1)).to(device) 
        model(dummy_input) 
        model.load_state_dict(torch.load(os.path.join(check_path, "checkpoint.pt")))
        model.eval()
        test_losses, test_accs = [], []
        for batch in test_loader:
            tloss, tacc = eval_step(model, batch, device)
            test_losses.append(tloss)
            test_accs.append(tacc)
        mean_test_loss = sum(test_losses) / len(test_losses)
        mean_test_acc = sum(test_accs) / len(test_accs)
        
        # y_true = torch.tensor([], dtype=torch.long).to(device)
        # y_pred = torch.tensor([], dtype=torch.float32).to(device)
        
        # for batch in test_loader:
        #     # 获取真实标签
        #     inputs = batch['inputs'].to(device)
        #     targets = batch['targets'].to(device)
        #     logits = model(inputs)
        #     y_true = torch.cat([y_true, targets])
        #     y_pred = torch.cat([y_pred, logits])
        #     torch.cuda.empty_cache()
        #     print(y_true)
            
        # mean_test_loss = compute_weighted_cross_entropy(y_pred, y_true)
        # mean_test_acc = compute_weighted_accuracy(y_pred, y_true)
        print(f"Test Loss {mean_test_loss:.4f}, Test Acc {mean_test_acc:.4f}")
        with open(os.path.join(save_path, 'results.json'), 'w') as f:
            json.dump({'loss': mean_test_loss, 'accuracy': mean_test_acc}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 添加所有必要的参数
    parser.add_argument('--mode', type=str, default='train')
    
    parser.add_argument('--model_name', type=str, default='transformer')
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs',type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--warmup',type=int,default=5000)
    parser.add_argument('--num_train_steps', type=int, default=20000)
    parser.add_argument('--num_eval_steps', type=int, default=999999)
    parser.add_argument('--eval_frequency', type=int, default=500)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=0)
    
    parser.add_argument('--model_dir', type=str, default='./model_ckpt')
    parser.add_argument('--data_dir', type=str, default='./dataset/')
    parser.add_argument('--task_name', type=str, default='basic')
    
    
    parser.add_argument('--max_length', type=int, default=2000)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--qkv_dim', type=int, default=128)
    parser.add_argument('--mlp_dim', type=int, default=256)
    parser.add_argument('--classifier', action='store_true', default=True)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.1)
    parser.add_argument('--num_freq', type=int, default=32)

    parser.add_argument('--seed', type=int, default=2025)
    
    args = parser.parse_args()
    print(args)
    main(args)