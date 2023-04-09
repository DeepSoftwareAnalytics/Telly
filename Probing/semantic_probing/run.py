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
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model import SemanticProbeModel,SemanticWeightedProbeModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)
import sys
sys.path.append("../../../utils")
from utils import save_json_data, save_pickle_data, count_parameters


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 index,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label

        
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    code = ' '.join(js['code'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
    source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['index'],int(js['label']))

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
                if args.debug  and len(data) >= args.n_debug_samples:
                    break    
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:1]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label]=[]
            self.label_examples[e.label].append(e)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        label = self.examples[i].label
        index = self.examples[i].index
        labels = list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example = random.sample(self.label_examples[label],1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example
                break
        n_example = random.sample(self.label_examples[random.sample(labels,1)[0]],1)[0]
        
        return (torch.tensor(self.examples[i].input_ids),torch.tensor(p_example.input_ids),
                torch.tensor(n_example.input_ids),torch.tensor(label))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=2,pin_memory=True)
    
    args.max_steps = args.num_train_epochs*len( train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_map = [], 0
    
    model.zero_grad()
    for idx in range(args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)    
            p_inputs = batch[1].to(args.device)
            n_inputs = batch[2].to(args.device)
            labels = batch[3].to(args.device)
            model.train()
            if args.probe_type in [ "layer-wise"]:
                loss,vec = model(inputs,p_inputs,n_inputs,labels)
            elif args.probe_type in [ "weighted-layer-wise"]:
                loss,vec,_ = model(inputs,p_inputs,n_inputs,labels)
            else:
                raise UndefinedError ("Undefine probe_type")   

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())

            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),4)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  

        results = evaluate(args, model, tokenizer, args.eval_data_file)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))                    

        if results['eval_map'] > best_map:
            best_map = results['eval_map']
            logger.info("  "+"*"*20)  
            logger.info("  Best map:%s",round(best_map,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-map'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))   
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, data_file):
    """ Evaluate the model """
    eval_dataset = TextDataset(tokenizer, args, data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size, num_workers=4)
    
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs=[] 
    labels=[]
    all_layer_weights = []
    avg_layer_weights = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)    
        p_inputs = batch[1].to(args.device)
        n_inputs = batch[2].to(args.device)
        label = batch[3].to(args.device)
        with torch.no_grad():
            if args.probe_type in [ "layer-wise"]:
                lm_loss,vec = model(inputs,p_inputs,n_inputs,label)
            elif args.probe_type in [ "weighted-layer-wise"]:
                lm_loss, vec, layer_weights = model(inputs,p_inputs,n_inputs,label)
                all_layer_weights.append(layer_weights.view(-1,layer_weights.shape[-1]).cpu().numpy())
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
            

        nb_eval_steps += 1
    vecs = np.concatenate(vecs,0)
    labels = np.concatenate(labels,0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores=np.matmul(vecs,vecs.T)
    dic={}
    for i in range(scores.shape[0]):
        scores[i,i] = -1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])] = -1
        dic[int(labels[i])] += 1
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index = sort_ids[i,j]
            if int(labels[index]) == label:
                Avep.append((len(Avep)+1)/(j+1))
        MAP.append(sum(Avep)/dic[label])
          
          
    result = {
        "eval_loss": float(perplexity),
        "eval_map":float(np.mean(MAP))
    }
    if len(all_layer_weights) > 0 and "test" in data_file:
        all_layer_weights=np.concatenate(all_layer_weights,0)
        avg_layer_weights = all_layer_weights.mean(0).tolist()
        avg_layer_weights = [ round(item,4) for item in avg_layer_weights]
        result[ "layer_weights"] = avg_layer_weights

    return result

                        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncation_k_layer_index", default=0, type=int,help="")    
    parser.add_argument("--probe_type", default="layer-wise", choices=["layer-wise","weighted-layer-wise"],help="")    

    ## Required parameters
    parser.add_argument("--output_dir", default="saved_models", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=1000, help="must more than 500",required=False)
    
    ## Other parameters
    parser.add_argument("--train_data_file", default="dataset/train.jsonl", type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--eval_data_file", default="dataset/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/valid.jsonl", type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
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
        model = RobertaModel.from_pretrained(args.model_name_or_path,output_hidden_states=True) 
        
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
        model = SemanticProbeModel(truncated_model,config)
    elif args.probe_type in [ "weighted-layer-wise"]:
        model = SemanticWeightedProbeModel(model,config)
    else:
        raise UndefinedError ("Undefine probe_type")
    # model = Model(model,config,tokenizer,args)
    logger.info("Training/evaluation parameters %s", args)
    logger.info(model.model_parameters())
    logger.info('The model has %s trainable parameters' % str(count_parameters(model)))


    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training     
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        train(args, train_dataset, model, tokenizer)
        
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],2)))
            
    if args.do_test:
        # checkpoint_prefix = 'checkpoint-best-map/model.bin'
        # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        # model_to_load = model.module if hasattr(model, 'module') else model  
        # model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            if key != "layer_weights":
                result[key] = round(result[key]*100 if "map" in key else result[key],3)
            logger.info("  %s = %s", key, str(result[key]))
        save_json_data(args.output_dir, "result.jsonl",  result)    


if __name__ == "__main__":
    main()


