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

from torch.optim.lr_scheduler import StepLR

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
        
def is_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)
        
parser = argparse.ArgumentParser(description='Train a Model model')

parser.add_argument('--epochs', type=int, default=300, help='How many training epochs to perform')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
#Default is set ot 16 MB! 
parser.add_argument('--max_len', type=int, default=16000000, help='Maximum length of input file in bytes, at which point files will be truncated')

parser.add_argument('--save-every', type=int, default=25, help='Batch size during training')

parser.add_argument('--gpus', nargs='+', type=int)

parser.add_argument('--checkpoint', type=is_file, help='File to load and use')
parser.add_argument('--log', default="long_train", type=str, help='Log file location')


names_in_check_order=  ["Avast", "MalConvML", "MalConvGCT", "MalConv"]

parser.add_argument('--model', type=str, default=None, choices=names_in_check_order, help='Type of model to train')

parser.add_argument('mal_train', type=dir_path, help='Path to directory containing malware files for training')
parser.add_argument('ben_train', type=dir_path, help='Path to directory containing benign files for training')
parser.add_argument('mal_test', type=dir_path, help='Path to directory containing malware files for testing')
parser.add_argument('ben_test', type=dir_path, help='Path to directory containing benign files for testing')

args = parser.parse_args()

GPUS = args.gpus

torch.backends.cudnn.enabled = False

EPOCHS = args.epochs
MAX_FILE_LEN = args.max_len

BATCH_SIZE = args.batch_size

if args.model is not None:
    MODEL_NAME = args.model
else:
    #Noe model name type was specified. Can we infer it from the file path of the checkpoint?
    for option in names_in_check_order:
        if option in args.checkpoint:
            MODEL_NAME = option
            break
        
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



if MODEL_NAME.lower() == "MalConv".lower():
    from MalConv import getParams, initModel
elif MODEL_NAME.lower() == "Avast".lower():
    from AvastStyleConv import getParams, initModel
elif MODEL_NAME.lower() == "MalConvML".lower():
    from MalConvML import getParams, initModel
elif MODEL_NAME.lower() == "MalConvGCT".lower():
    from MalConvGCT import getParams, initModel
    print("CORRECT GCT")


if GPUS is None:#use ALL of them! (Default) 
    device_str = "cuda:0"
else:
    if GPUS[0] < 0:
        device_str = "cpu"
    else:
        device_str = "cuda:{}".format(GPUS[0])
    

device = torch.device(device_str if torch.cuda.is_available() else "cpu")


checkpoint = torch.load(args.checkpoint, map_location=device)
print([key for key in checkpoint.keys()])

NON_NEG = checkpoint['non_neg']

#Create model of same type
model = initModel(**checkpoint).to(device)
#optimizer = optim.AdamW(model.parameters(), lr=checkpoint['lr'])
optimizer = optim.AdamW(model.parameters())

#Restore weights and parameters
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


del checkpoint['model_state_dict']
del checkpoint['optimizer_state_dict']
args_to_use = checkpoint

whole_dataset = BinaryDataset(args.ben_train, args.mal_train, sort_by_size=True, max_len=MAX_FILE_LEN )
test_dataset = BinaryDataset(args.ben_test, args.mal_test, sort_by_size=True, max_len=MAX_FILE_LEN )

#Sub sample for testing purposes, not use when you want to do real work
#whole_dataset = random_split(whole_dataset, [1000])[0]
#test_dataset = random_split(test_dataset, [1000])[0]

loader_threads = max(multiprocessing.cpu_count()-4, multiprocessing.cpu_count()//2+1)

train_loader = DataLoader(whole_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
                        sampler=RandomChunkSampler(whole_dataset,BATCH_SIZE))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
                        sampler=RandomChunkSampler(test_dataset,BATCH_SIZE))

headers = ['epoch', 'train_acc', 'train_auc', 'test_acc', 'test_auc']

base_name = args.log

if not os.path.exists(base_name):
    os.makedirs(base_name)
file_name = os.path.join(base_name, base_name)

with open(base_name + ".csv", 'w') as csv_log_out:
    csv_log_out.write(",".join(headers) + "\n")

    criterion = nn.CrossEntropyLoss()
    
    scheduler = StepLR(optimizer, step_size=EPOCHS//10, gamma=0.5)
    
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
        if epoch % args.save_every == 0 or epoch == EPOCHS-1:
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

        csv_log_out.write(",".join([str(epoch_stats[h]) for h in headers]) + "\n")
        csv_log_out.flush()
        
        scheduler.step()
