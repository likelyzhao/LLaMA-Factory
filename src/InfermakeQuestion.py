import inspect
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file

from llmtuner.data import get_template_and_fix_tokenizer
from llmtuner.data.loader import load_single_dataset
from llmtuner.data.parser import get_dataset_list
from llmtuner.extras.constants import CHOICES, SUBJECTS
from llmtuner.hparams import get_infer_args
from llmtuner.model import load_model, load_tokenizer
from llmtuner.eval.template import get_eval_template
from llmtuner import ChatModel


def formatcontent(prompt:str, query: str) -> str:
    qa_num = 3
    p = prompt.format(count=qa_num, text=query)
    return p

# load message from file and save the result with json format
class FileInfer:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:        
        self.model_args, self.data_args, self.eval_args, self.generation_args = get_infer_args(args)
        self.chat_model = ChatModel()
        outpath = self.generation_args.filepath.replace(os.path.splitext(self.generation_args.filepath)[-1], 
                                                         "_out.jsonl")
        self.fout = open(outpath, "w")
        with open(self.generation_args.prompt_path, 'r') as f:
            self.template = ''.join(f.readlines())

    def eval(self) -> None:
        from tqdm import tqdm
        with open(self.generation_args.filepath) as f:

            for line in tqdm(f.readlines()):
                if len(line) < 100:
                    continue
                #print(dataset[i])
                #import pdb
                #pdb.set_trace()
                content = formatcontent(self.template, line)
                message = {"role": "user", "content": content}
            
                if len(message['content']) > 8192/3:
                    print(message['content']+ "is too long ")
                    continue
                
                #print(dataset[i])
                #print(message)
                response = ""
                
                for new_text in self.chat_model.stream_chat([message]):
                    response += new_text
   
                    # print(message)
                    # print(response)
                    # continue
                
                
                newsample = {}
                newsample['input'] = message['content']
                newsample['model'] = self.model_args.model_name_or_path
                newsample['adapter'] = self.model_args.adapter_name_or_path
                newsample['response'] = response
                #print(newsample)
                import json 
                json.dump(newsample, self.fout, ensure_ascii=False)
                self.fout.write("\n")
                self.fout.flush()



def main():
    FileInfer().eval()

if __name__ == "__main__":
    main()
