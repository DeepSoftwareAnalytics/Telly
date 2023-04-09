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
# from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
import sys
sys.path.append("../utils")
from utils import save_json_data

class Model(nn.Module):   
    def __init__(self, encoder,args):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None): 
        pass
 

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

        
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
                    if args.debug and len(data) > args.n_debug_samples:
                        break
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
                    if args.debug and len(data) > args.n_debug_samples:
                        break
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)
                    if args.debug and len(data) > args.n_debug_samples:
                        break 

        for js in data[:5000]:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))

def get_triup_score_of_hidden_layer_i(code_vecs,i):
    hidden_layer_i = [item[i] for item in code_vecs]
    hidden_layer_i = np.concatenate(hidden_layer_i,0)
    # hidden_layer_i = hidden_layer_i.mean(1)
    scores = np.matmul(hidden_layer_i,hidden_layer_i.T)
    triup_score = []
    for j in range(scores .shape[0]):
        triup_score.extend(scores[j,j:].tolist())
    return triup_score

# class ARGS(object):
#     def __init__(self,):
#         # self.train_data_file="../code/dataset/CSN/python/train.jsonl"
#         # self.code_length=256
#         # self.nl_length=128
#         self.model_name_or_path="microsoft/unixcoder-base"

        
def parse_arg():
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    # params
    parser.add_argument("--nl_length", default=10, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    # model type
    parser.add_argument("--model_type", default=None, type=str,choices=["pre-trained", "fine-tuned"],
                        help="pre-train or fine tuned")
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--loaded_model_path", default=None, type=str,
                        help="loaded model path")
    # task
    parser.add_argument("--task", default=None, type=str,
    choices=["pre-trained", "code-search", "clone-detection","code-sum","code-gen","code-completion"],
                        help="evaluated tasks")
    # data
    parser.add_argument("--train_data_file", default="../code/dataset/CSN/python/train.jsonl", type=str, 
                        help="The input training data file (a text file).")
    # save 
    parser.add_argument("--output_dir", default="saved_layer_wised_representations_matrix", type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()    
    return args

if __name__=="__main__":
    args = parse_arg()
    logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # load model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path,output_hidden_states=True) 
    logger.info("loaded model")
    # load dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=50,num_workers=4)
    logger.info("loaded dataset")

    logger.info("The model type is %s"%args.model_type)
    if args.model_type == "fine-tuned":
        model = Model(encoder=model, args=args)
        # model_to_load = model.module if hasattr(model, 'module') else model  
        model.load_state_dict(torch.load(args.loaded_model_path), strict=False)
         

    model.to(args.device)
    # Obtain the vectoe
    model.eval()
    code_vecs = [] 
    for batch in tqdm(train_dataloader, total=len(train_dataloader)) :
        code_inputs = batch[0] .to(args.device) 
        with torch.no_grad():
            if args.model_type == "fine-tuned":
                output =  model.encoder(code_inputs,attention_mask=code_inputs.ne(1)).hidden_states
            else:
                output =  model(code_inputs,attention_mask=code_inputs.ne(1)).hidden_states
        output = [torch.nn.functional.normalize(item.mean(1), p=2, dim=-1)  for item in output]  
        output = [item.cpu().numpy().tolist() for item in output]
        # output = output .tolist()
        code_vecs.append( output) 
    logger.info("obtained the representation")        
    pre_train_data_sample_scores= {}
    for i in range(13):
        pre_train_data_sample_scores[i] = get_triup_score_of_hidden_layer_i(code_vecs,i)
    logger.info("obtained the matrix")  
    # output_dir = os.path.join(args.output_dir, args.task)
    save_json_data( args.output_dir, "result.jsonl", pre_train_data_sample_scores)
    

                
