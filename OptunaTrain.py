import os
from collections import deque
from collections import OrderedDict 

import random
import numpy as np

#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch.optim as optim

from torch.utils import data

from torch.utils.data import Dataset, DataLoader, Subset

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from sklearn.metrics import roc_auc_score

import optuna

import argparse

#Check if the input is a valid directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Train a Model model')

parser.add_argument('--epochs', type=int, default=20, help='How many training epochs to perform')
parser.add_argument('--non-neg', type=bool, default=False, help='Should non-negative training be used')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
parser.add_argument('--trials', type=int, default=50, help='Number of hyper-parameter tuning trials to perform')
#Default is set ot 16 MB! 
parser.add_argument('--max_len', type=int, default=16000000, help='Maximum length of input file in bytes, at which point files will be truncated')
parser.add_argument('--val-split', type=float, default=0.1, help='Batch size during training')

parser.add_argument('--model', type=str, default="MalConv", choices=["MalConv", "Avast", "MalConvML", "MalConvGCT"], help='Type of model to train')

parser.add_argument('--gpus', nargs='+', type=int)


parser.add_argument('mal_train', type=dir_path, help='Path to directory containing malware files for training')
parser.add_argument('ben_train', type=dir_path, help='Path to directory containing benign files for training')
parser.add_argument('mal_test', type=dir_path, help='Path to directory containing malware files for testing')
parser.add_argument('ben_test', type=dir_path, help='Path to directory containing benign files for testing')

args = parser.parse_args()

GPUS = args.gpus

NON_NEG = args.non_neg
EPOCHS = args.epochs
MAX_FILE_LEN = args.max_len
MODEL_NAME = args.model
TRIALS = args.trials

BATCH_SIZE = args.batch_size

if MODEL_NAME.lower() == "MalConv".lower():
    from MalConv import getParams, initModel
elif MODEL_NAME.lower() == "Avast".lower():
    from AvastStyleConv import getParams, initModel
elif MODEL_NAME.lower() == "MalConvML".lower():
    from MalConvML import getParams, initModel
elif MODEL_NAME.lower() == "MalConvGCT".lower():
    from MalConvGCT import getParams, initModel



whole_dataset = BinaryDataset(args.ben_train, args.mal_train, sort_by_size=True, max_len=MAX_FILE_LEN )
test_dataset = BinaryDataset(args.ben_test, args.mal_test, sort_by_size=True, max_len=MAX_FILE_LEN )

loader_threads = max(multiprocessing.cpu_count()-4, multiprocessing.cpu_count()//2+1)

#Create train & validation split

#First we define our own random split, b/c we want to keep data shuffle order in 
#tact b/c it will make trainin faster. This is because we kept things orded by size, so batches 
#can be as small as possible. 
def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    #if sum(lengths) != len(dataset):
    #    raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths)).tolist()
    to_ret = []
    for offset, length in zip(torch._utils._accumulate(lengths), lengths):
        selected = indices[offset - length:offset]
        selected.sort()
        to_ret.append( Subset(dataset, selected) )
    return to_ret


whole_len = len(whole_dataset)
train_size = int(whole_len*(1-args.val_split))
validation_size = whole_len-train_size
dataset_train_split, dataset_val_split = random_split(whole_dataset, (train_size, validation_size))

train_loader = DataLoader(dataset_train_split, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
                        sampler=RandomChunkSampler(dataset_train_split,BATCH_SIZE))

val_loader = DataLoader(dataset_val_split, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
                        sampler=RandomChunkSampler(dataset_val_split,BATCH_SIZE))

# train_loader = DataLoader(whole_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
#                         sampler=RandomChunkSampler(whole_dataset,BATCH_SIZE))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
                        sampler=RandomChunkSampler(test_dataset,BATCH_SIZE))

if GPUS is None:#use ALL of them! (Default) 
    device_str = "cuda:0"
else:
    if GPUS[0] < 0:
        device_str = "cpu"
    else:
        device_str = "cuda:{}".format(GPUS[0])
    

device = torch.device(device_str if torch.cuda.is_available() else "cpu")

def objective(trial):

    args_to_use = {
        'lr':0.001,
    }
    
    if not trial is None:
        for param, (sample_func, sample_args) in getParams().items():
            args_to_use[param] = getattr(trial, sample_func)(**sample_args)
        args_to_use['lr'] = trial.suggest_loguniform('lr', low=1e-4, high=1e-2)
    
    args_to_use = OrderedDict(sorted(args_to_use.items(), key=lambda t: t[0]))
    
    model = initModel(**args_to_use).to(device)
    

    base_name = MODEL_NAME + "_".join([a + "_" + str(b) for (a, b) in args_to_use.items()])

    if NON_NEG:
        base_name = "NonNeg_" + base_name

    if GPUS is None or len(GPUS) > 1:
        model = nn.DataParallel(model, device_ids=GPUS)

    if not os.path.exists(base_name):
        os.makedirs(base_name)
    file_name = os.path.join(base_name, base_name)
    

    headers = ['epoch', 'train_acc', 'train_auc', 'test_acc', 'test_auc', 'val_acc', 'val_auc']

#     csv_log_out = open(file_name + ".csv", 'w')
    with open(file_name + ".csv", 'w') as csv_log_out:
        csv_log_out.write(",".join(headers) + "\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args_to_use['lr'])

        for epoch in tqdm(range(EPOCHS)):

            preds = []
            truths = []
            running_loss = 0.0


            train_correct = 0
            train_total = 0

            epoch_stats = {'epoch':epoch}

            model.train()
            for inputs, labels in tqdm(train_loader):

                #inputs, labels = inputs.to(device), labels.to(device)
                #Keep inputs on CPU, the model will load chunks of input onto device as needed
                labels = labels.to(device)

                optimizer.zero_grad()

            #     outputs, penultimate_activ, conv_active = model.forward_extra(inputs)
                outputs, penult, post_conv = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss #+ decov_lambda*(decov_penalty(penultimate_activ) + decov_penalty(conv_active))
            #     loss = loss + decov_lambda*(decov_penalty(conv_active))
                loss.backward()
                optimizer.step()
                if NON_NEG:
                    for p in model.parameters():
                        p.data.clamp_(0)


                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                with torch.no_grad():
                    preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
                    truths.extend(labels.detach().cpu().numpy().ravel())

                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            #end train loop

            #print("Training Accuracy: {}".format(train_correct*100.0/train_total))

            epoch_stats['train_acc'] = train_correct*1.0/train_total
            epoch_stats['train_auc'] = roc_auc_score(truths, preds)
            #epoch_stats['train_loss'] = roc_auc_score(truths, preds)

            #Save the model and current state!
            model_path = os.path.join(base_name, "epoch_{}.checkpoint".format(epoch))


            #Have to handle model state special if multi-gpu was used
            if type(model).__name__ is "DataParallel":
                mstd = model.module.state_dict()
            else:
                mstd = model.state_dict()

            #Copy dict, and add extra info to save off 
            check_dict = args_to_use.copy()
            check_dict['epoch'] = epoch
            check_dict['model_state_dict'] = mstd
            check_dict['optimizer_state_dict'] = optimizer.state_dict()
            check_dict['non_neg'] = NON_NEG
            torch.save(check_dict, model_path)


            #Test Set Eval
            model.eval()
            eval_train_correct = 0
            eval_train_total = 0

            preds = []
            truths = []
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader):

                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs, _, _ = model(inputs)

                    _, predicted = torch.max(outputs.data, 1)

                    preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
                    truths.extend(labels.detach().cpu().numpy().ravel())

                    eval_train_total += labels.size(0)
                    eval_train_correct += (predicted == labels).sum().item()

            epoch_stats['test_acc'] = eval_train_correct*1.0/eval_train_total
            epoch_stats['test_auc'] = roc_auc_score(truths, preds)

            #We've now done an epoch of training. Lets do a validation run to see what our current reuslts look like & report to Optuna
            eval_train_correct = 0
            eval_train_total = 0
            preds = []
            truths = []
            with torch.no_grad():
                for inputs, labels in val_loader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs, _, _ = model(inputs)

                    _, predicted = torch.max(outputs.data, 1)
                    
                    preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
                    truths.extend(labels.detach().cpu().numpy().ravel())
                    
                    eval_train_total += labels.size(0)
                    eval_train_correct += (predicted == labels).sum().item()

            validation_error = 1.0 - eval_train_correct/eval_train_total
            trial.report(validation_error, epoch)
            
            epoch_stats['val_acc'] = eval_train_correct*1.0/eval_train_total
            epoch_stats['val_auc'] = roc_auc_score(truths, preds)
            
            csv_log_out.write(",".join([str(epoch_stats[h]) for h in headers]) + "\n")
            csv_log_out.flush()

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.structs.TrialPruned()

        #end for epoch loop
        for att in ['val_acc', 'val_auc', 'test_acc', 'test_auc']:
            trial.set_user_attr(att, epoch_stats[att])
    #end 'with csv' log
    return validation_error
    

study_name = MODEL_NAME
if NON_NEG:
    study_name = "NonNeg_" + study_name
study = optuna.create_study(study_name=study_name, storage='sqlite:///{}.db'.format(study_name), pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, n_trials=TRIALS)



study.trials_dataframe().to_pickle(out_name + "_pd.pkl")



