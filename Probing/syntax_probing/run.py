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
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import torch.nn as nn
import json
import numpy as np
from model import SyntaxProbModel,SyntaxWeightedProbeModel
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
from prettytable import PrettyTable
import multiprocessing
cpu_cont = multiprocessing.cpu_count()
logger = logging.getLogger(__name__)
import sys
# sys.path.append("dataset")
sys.path.append("../../../utils")
from utils import save_json_data, save_pickle_data, count_parameters

# import torch   
class BaseModel(nn.Module): 
    def __init__(self, ):
        super().__init__()
        
    def model_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table

class Model(BaseModel):   
    def __init__(self, encoder,args):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)    
        
def cal_r1_r5_r10(ranks):
    r1,r5,r10= 0,0,0
    data_len= len(ranks)
    for item in ranks:
        if item >=1:
            r1 +=1
            r5 += 1 
            r10 += 1
        elif item >=0.2:
            r5+= 1
            r10+=1
        elif item >=0.1:
            r10 +=1
    result = {"R@1":round(r1/data_len,3), "R@5": round(r5/data_len,3),  "R@10": round(r10/data_len,3)}
    return result


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

        
def convert_examples_to_features(item):
    """convert examples to token ids"""
    js,tokenizer,args = item[0], item[1], item[2]
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    sbt_ao = ' '.join(js['sbt_ao']).lower() 
    sbt_tokens = tokenizer.tokenize(sbt_ao)[:args.sbt_length-4]
    sbt_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+sbt_tokens+[tokenizer.sep_token]
    sbt_ids = tokenizer.convert_tokens_to_ids(sbt_tokens)
    padding_length = args.sbt_length - len(sbt_ids)
    sbt_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,sbt_tokens,sbt_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        n_debug_samples = args.n_debug_samples
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    if args.debug  and len(data) >= n_debug_samples:
                            break
                    data.append((js,tokenizer,args))
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append((temp,tokenizer,args))
                    if  args.debug  and len(data) >= n_debug_samples:
                            break
            elif "json" in file_path:
                for js in json.load(f):
                    data.append((js,tokenizer,args))
                    if args.debug and len(data) >= n_debug_samples:
                            break  

        # for js in data:
        #     self.examples.append(convert_examples_to_features(js,tokenizer,args))
        pool = multiprocessing.Pool(cpu_cont)
        
        self.examples=pool.map(convert_examples_to_features, data)
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:1]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("sbt_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("sbt_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
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
    tr_num,tr_loss,best_mrr = 0,0,-1 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            sbt_inputs = batch[1].to(args.device)
            #get code and nl vectors
            if args.probe_type in [ "layer-wise"]:
                code_vec = model(code_inputs)
                sbt_vec = model(sbt_inputs)
            elif args.probe_type in [ "weighted-layer-wise"]:
                code_vec = model(code_inputs)[0]
                sbt_vec = model(sbt_inputs)[0]
            else:
                raise UndefinedError ("Undefine probe_type")   
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",sbt_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%logger_frquence== 0:
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
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,file_name,eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    ranks = [] 
    all_layer_weights = []
    avg_layer_weights = []
    for batch in query_dataloader:
        code_inputs = batch[0].to(args.device)    
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            if args.probe_type in [ "layer-wise"]:
                code_vecs = model(code_inputs).cpu().numpy()
                nl_vecs = model(nl_inputs).cpu().numpy()
            elif args.probe_type in [ "weighted-layer-wise"]:
                code_vecs, layer_weights = model(code_inputs)
                code_vecs = code_vecs.cpu().numpy()
                nl_vecs = model(nl_inputs)[0].cpu().numpy()
                all_layer_weights.append(layer_weights.view(-1,layer_weights.shape[-1]).cpu().numpy())

            scores = np.matmul(nl_vecs,code_vecs.T)
            sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1] 
            rank=(1/(np.diagonal(sort_ids)+1)).tolist()
            ranks.extend(rank)
    result = cal_r1_r5_r10(ranks)
    if len(all_layer_weights) > 0 and "test" in file_name:
        all_layer_weights=np.concatenate(all_layer_weights,0)
        avg_layer_weights = all_layer_weights.mean(0).tolist()
        avg_layer_weights = [ round(item,4) for item in avg_layer_weights]
        result[ "layer_weights"] = avg_layer_weights
    
    result["eval_mrr"]  = round(float(np.mean(ranks)),3)

    return result

# def evaluate(args, model, tokenizer,file_name,eval_when_training=False):
#     query_dataset = TextDataset(tokenizer, args, file_name)
#     query_sampler = SequentialSampler(query_dataset)
#     query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
#     code_dataset = TextDataset(tokenizer, args, args.codebase_file)
#     code_sampler = SequentialSampler(code_dataset)
#     code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    
#     # Eval!
#     logger.info("***** Running evaluation *****")
#     logger.info("  Num queries = %d", len(query_dataset))
#     logger.info("  Num codes = %d", len(code_dataset))
#     logger.info("  Batch size = %d", args.eval_batch_size)

    
#     model.eval()
#     code_vecs = [] 
#     nl_vecs = []
#     for batch in query_dataloader:  
#         nl_inputs = batch[1].to(args.device)
#         with torch.no_grad():
#             nl_vec = model(nl_inputs=nl_inputs) 
#             nl_vecs.append(nl_vec.cpu().numpy()) 

#     for batch in code_dataloader:
#         code_inputs = batch[0].to(args.device)    
#         with torch.no_grad():
#             code_vec = model(code_inputs=code_inputs)
#             code_vecs.append(code_vec.cpu().numpy())  
#     model.train()    
#     code_vecs = np.concatenate(code_vecs,0)
#     nl_vecs = np.concatenate(nl_vecs,0)
    
#     scores = np.matmul(nl_vecs,code_vecs.T)
    
#     sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
#     nl_urls = []
#     code_urls = []
#     for example in query_dataset.examples:
#         nl_urls.append(example.url)
        
#     for example in code_dataset.examples:
#         code_urls.append(example.url)

#     ranks = []
#     for url, sort_id in zip(nl_urls,sort_ids):
#         rank = 0
#         find = False
#         for idx in sort_id[:1000]:
#             if find is False:
#                 rank += 1
#             if code_urls[idx] == url:
#                 find = True
#         if find:
#             ranks.append(1/rank)
#         else:
#             ranks.append(0)
#     if args.save_evaluation_reuslt:
#         evaluation_result = {"nl_urls":nl_urls, "code_urls":code_urls,"sort_ids":sort_ids[:,:100],"ranks":ranks}
#         save_pickle_data(args.save_evaluation_reuslt_dir, "evaluation_result.pkl",evaluation_result)
#     result = cal_r1_r5_r10(ranks)
#     result["eval_mrr"]  = round(float(np.mean(ranks)),3)

#     return result


def parse_args():
    parser = argparse.ArgumentParser()
    # 
    # parser.add_argument("--code_index", default=0, type=int,help="")    
    # parser.add_argument("--nl_index", default=0, type=int,help="")    
    # parser.add_argument("--layer_index", default=0, type=int,help="")    
    parser.add_argument("--truncation_k_layer_index", default=0, type=int,help="")    
    parser.add_argument("--probe_type", default="layer-wise", choices=["layer-wise","weighted-layer-wise"],help="")    

    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--save_evaluation_reuslt', action='store_true', help='save_evaluation_reuslt', required=False)
    parser.add_argument('--save_evaluation_reuslt_dir', type=str, required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--sbt_length", default=256, type=int,
                        help="Optional sbt input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")     
    args = parser.parse_args()
    return args                  
                        
def main():

    
    #print arguments
    args = parse_args()
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
        
    # config.num_hidden_layers = args.truncation_k_layer_index
    # dict_model_params = dict(model.named_parameters())
    # truncated_model = RobertaModel(config)
    # for name, param in truncated_model.named_parameters():
    #     param.data.copy_( dict_model_params[name])
    #     param.requires_grad = False
    #     if args.debug:
    #         logger.info("copied the paramter")
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
        model = SyntaxProbModel(truncated_model,config)
    elif args.probe_type in [ "weighted-layer-wise"]:
        model = SyntaxWeightedProbeModel(model,config)
    else:
        raise UndefinedError ("Undefine probe_type")

    # model = SyntaxProbModel(truncated_model,config)
    logger.info("Training/evaluation parameters %s", args)
    logger.info(model.model_parameters())
    logger.info('The model has %s trainable parameters' % str(count_parameters(model)))

    # exit()
    
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
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.test_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        save_json_data(args.output_dir, "result.jsonl", result)

if __name__ == "__main__":
    main()


