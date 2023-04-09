# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
from distutils.log import log
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

from yaml import parse
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from sklearn.metrics import recall_score,precision_score,f1_score
from tqdm import tqdm, trange
import multiprocessing
from model import LexicalProbeModel,LexicalWeightedProbeModel
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)

import sys
sys.path.append("../../../utils")
from utils import save_json_data
logger = logging.getLogger(__name__)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

label_vocab={"<pad>":0, "KEYWORD":1, "NAME":2, "NUMBER":3, "STRING":4, "OPERATOR":5}
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 aligned_tagging, 
                 tagging_ids

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.aligned_tagging = aligned_tagging
        self.tagging_ids = tagging_ids

def align_tokenizations(data,tokenizer,args):
    code=[item[0]  for item in data['token_type']][:args.block_size-4]
    tagging = [item[1]  for item in data['token_type']][:args.block_size-4]


    tokenized_code = tokenizer.tokenize(" ".join(code))

    aligned_tagging = []
    current_word = ''
    index = 0 # index of current word in sentence and tagging
    for token in tokenized_code:
        if tagging[index] == "STRING" and len(code[index].split()) > 1 and len(current_word)>0:
            current_word += re.sub(r'Ġ', ' ', token) # recompose word with subtoken
        else:
            current_word += re.sub(r'Ġ', '', token) # recompose word with subtoken
        # note that some word factors correspond to unknown words in BERT
        try:
            assert (token == tokenizer.unk_token or code[index].startswith(current_word))
        except:
            print(code[index],token, current_word )

        if token == tokenizer.unk_token or code[index] == current_word: # if we completed a word
            current_word = ''
            aligned_tagging.append(tagging[index])
            index += 1
        else: # otherwise insert padding
            aligned_tagging.append(tokenizer.pad_token)

    assert len(tokenized_code) == len(aligned_tagging)

    return tokenized_code, aligned_tagging
        
def convert_examples_to_features(item):
    js,tokenizer,args = item[0],item[1],item[2]

    code_tokens, aligned_tagging = align_tokenizations(js,tokenizer,args) 
    code_tokens, aligned_tagging = code_tokens[:args.block_size-4], aligned_tagging[:args.block_size-4]
    source_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length

    aligned_tagging= ["<pad>"]*3 +  aligned_tagging + ["<pad>"]
    tagging_ids =  [label_vocab[tag] for tag in aligned_tagging] 
    tagging_ids += [label_vocab["<pad>"]]*padding_length

    # logger.debug("source_ids: %s"%(str(source_ids)))
    # logger.debug("tagging_ids: %s"%(str(tagging_ids)))
    return InputFeatures(source_tokens,source_ids,aligned_tagging, tagging_ids)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        data = []
        n_debug_samples = args.n_debug_samples
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                data.append((js,tokenizer,args))
                if args.debug  and len(data) >= n_debug_samples:
                    break
        pool = multiprocessing.Pool(cpu_cont)
        self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:1]):
                    logger.info("*** Example ***")
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("aligned_taggings: {}".format([x.replace('\u0120','_') for x in example.aligned_tagging]))
                    logger.info("tagging_ids: {}".format(' '.join(map(str, example.tagging_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].tagging_ids)
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """ 
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    logger_frquence = int(len(train_dataloader)/10)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    criterion = nn.CrossEntropyLoss()
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)        
            labels=batch[1].to(args.device) 
            # model.train()
            if args.probe_type in [ "layer-wise"]:
                y_scores = model(inputs)
            elif args.probe_type in [ "weighted-layer-wise"]:
                y_scores = model(inputs)[0]
            else:
                raise UndefinedError ("Undefine probe_type")   
            loss = criterion(y_scores.view(-1, len(label_vocab)), labels.view(-1))
            # loss,logits = model(inputs,labels)
              #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%logger_frquence  == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
        for key, value in results.items():
            if type(value) is not list:
                logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['acc']>best_mrr:
            best_mrr = results['acc']
            logger.info("  "+"*"*20)  
            logger.info("  Best acc:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-acc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)





def evaluate(args, model, tokenizer,file_name,eval_when_training=False):
    evaluated_dataset = TextDataset(tokenizer, args, file_name)
    evaluated_sampler = SequentialSampler(evaluated_dataset)
    evaluated_dataloader = DataLoader(evaluated_dataset, sampler=evaluated_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(evaluated_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    total_loss = correct = num_loss = num_perf = 0
    criterion = nn.CrossEntropyLoss()
    all_layer_weights = []
    avg_layer_weights = []
    for batch in evaluated_dataloader: 
        inputs = batch[0].to(args.device)        
        labels=batch[1].to(args.device)  
        with torch.no_grad(): # no need to store computation graph for gradients
        # perform inference and compute loss
            if args.probe_type in [ "layer-wise"]:
                y_scores = model(inputs)
            elif args.probe_type in [ "weighted-layer-wise"]:
                y_scores, layer_weights = model(inputs) # layer_weights [bs, seq, layer_number]
                all_layer_weights.append(layer_weights.view(-1,layer_weights.shape[-1]).cpu().numpy())
            else:
                raise UndefinedError ("Undefine probe_type")   
            
            loss = criterion(y_scores.view(-1, len(label_vocab)), labels.view(-1)) #{bs,seq_len, vocab_size]}

            # gather loss statistics
            total_loss += loss.item()
            num_loss += 1

            # gather accuracy statistics
            y_pred = torch.max(y_scores, 2)[1] # compute highest-scoring tag
            mask = (labels != 0) # ignore <pad> tags
            correct += torch.sum((y_pred == labels) * mask) # compute number of correct predictions
            num_perf += torch.sum(mask).item()
        # return total_loss / num_loss, correct.item() / num_perf
    if len(all_layer_weights) > 0 and "test" in file_name:
        all_layer_weights=np.concatenate(all_layer_weights,0)
        avg_layer_weights = all_layer_weights.mean(0).tolist()
        avg_layer_weights = [ round(item,4) for item in avg_layer_weights]
        result = {
            "loss":round(total_loss / num_loss, 2),
            "acc":round(correct.item() / num_perf *100, 2),
            "layer_weights":avg_layer_weights
        }
    else:
        result = {
                "loss":round(total_loss / num_loss, 2),
                "acc":round(correct.item() / num_perf *100, 2),
            }

    return result

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)                      


def parse_arg():
    parser = argparse.ArgumentParser()

    ## interpretability
    parser.add_argument("--truncation_k_layer_index", default=0, type=int,help="")    
    parser.add_argument("--probe_type", default="layer-wise", choices=["layer-wise","weighted-layer-wise"],help="")    

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--do_zero_shot', action='store_true', help='debug mode', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # parser.add_argument('--epoch', type=int, default=42,
    #                     help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    args = parser.parse_args()    
    return args

def main():
    
    args = parse_arg()
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    if args.model_name_or_path in ["random"]:
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        config = RobertaConfig.from_pretrained("microsoft/unixcoder-base")
        # Initializing a model (with random weights) from the configuration
        config. output_hidden_states=True
        model = RobertaModel(config)  
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        model = RobertaModel.from_pretrained(args.model_name_or_path, output_hidden_states=True)
        
    logger.info("Probing type : %s"%args.probe_type)
    if args.probe_type in [ "layer-wise"]:
        config.num_hidden_layers = args.truncation_k_layer_index
        dict_model_params = dict(model.named_parameters())
        truncated_model = RobertaModel(config)
        for name, param in truncated_model.named_parameters():
            param.data.copy_( dict_model_params[name])
            param.requires_grad = False
            if args.debug:
                logger.info("copied the paramter")
        model = LexicalProbeModel(truncated_model,config,len(label_vocab))
    elif args.probe_type in [ "weighted-layer-wise"]:
        model = LexicalWeightedProbeModel(model,config,len(label_vocab))
    else:
        raise UndefinedError ("Undefine probe_type")


    logger.info("Training/evaluation parameters %s", args)
    logger.info(model.model_parameters())
    logger.info('The model has %s trainable parameters' % str(count_parameters(model)))

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training
    if args.do_train:
        train(args, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("** Valid results **")
        for key in sorted(result.keys()):
            if type(result[key]) is not list:
                logger.info("  %s = %s", key, str(round(result[key],3)))
            
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.test_data_file)
        logger.info("** Test results **")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        save_json_data(args.output_dir, "result.jsonl", result)



if __name__ == "__main__":
    main()


