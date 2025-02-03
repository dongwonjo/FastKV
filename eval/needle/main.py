"""
Adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack
"""
import sys
sys.path.append(".")

import tiktoken
import os 
import glob
import json
import logging
import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
# from anthropic import Anthropic
# from dotenv import load_dotenv
import numpy as np
import argparse
from pathlib import Path
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# from openai import OpenAI
from datetime import datetime, timezone
import time
import torch
import tqdm

from utils import utils

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 model_name,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="eval/needle/PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 128000,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "LLaMA",
                 openai_api_key = None,
                 anthropic_api_key = None,
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 save_path = None,
                 args = None):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        :param modified: Whether or not modify the model. Choose from [None, 'select', 'snapkv', 'h2o'].
        :param topk: Top k selection based on KV cache.
        :param select_layer_idx: For select mode.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.model_name = model_name
        
        self.args = args
        self.save_path = save_path

        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: 
            self.model_version = model_name
        
        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        # KV compression
        if args.mode == 'fullkv':
            from baseline.fullkv.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.mode == 'fastkv':
            from baseline.fastkv.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.mode == 'snapkv':
            from baseline.snapkv.monkeypatch import replace_llama, replace_mistral, replace_phi3
            replace_llama()
            replace_mistral()
            replace_phi3()
        elif args.mode == 'gemfilter':
            from baseline.gemfilter.monkeypatch import replace_llama, replace_mistral
            replace_llama()
            replace_mistral()
        elif args.mode == 'adakv':
            from baseline.adakv.adaptive_snapkv.monkeypatch import replace_llama_adaptive, replace_mistral_adaptive
            replace_llama_adaptive()
            replace_mistral_adaptive()
        elif args.mode == 'headkv':
            from baseline.headkv.headkv.monkeypatch import replace_llama, replace_mistral
            replace_llama(args.method)
            replace_mistral(args.method)
        else:
            raise ValueError(f"We does not support {args.mode} mode") 

        if(self.model_provider not in ["OpenAI", "Anthropic"]):  
            # Load Model & Tokenizer
            logging.info(f'Load Model & Tokenizer...')
            self.enc = AutoTokenizer.from_pretrained(self.model_name, device_map='auto', trust_remote_code=True)
            self.model_to_test = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', attn_implementation='flash_attention_2', torch_dtype=torch.float16)
            self.model_to_test.eval()
        else: 
            self.model_to_test = OpenAI(api_key=openai_api_key)
            if(self.model_provider == "OpenAI"):
                self.enc = tiktoken.encoding_for_model(self.model_name)
            elif(self.model_provider == "Anthropic"):
                self.enc = Anthropic().get_tokenizer()

        self.model_to_test_description = model_name
        
        self.evaluation_model = None
        self.debug='debug'
        model_name = model_name.split('/')[-1]

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in tqdm.tqdm(self.context_lengths, desc=f"Processing the each context length..."):
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                logging.info(f"Context Length: {context_length}, Depth Percent: {depth_percent}")
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        else: 
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },
                
            ]

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                logging.info("Result exists, skipping")
                return
            else:
                logging.info("Result does not exist, testing")

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        test_start_time = time.time()
        if(self.model_provider in ["OpenAI", "Anthropic"]):
            response = self.model_to_test.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=300,
                temperature=0
            )
            response = response.choices[0].message.content
        else:
            if self.args.mode == 'fastkv':
                from baseline.fastkv.fastkv_utils import compress
                compress(self.model_to_test, self.args)
            elif self.args.mode == 'snapkv':
                from baseline.snapkv.snapkv_utils import compress
                compress(self.model_to_test, self.args)
            elif self.args.mode == 'gemfilter':
                from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection, set_topk
                set_topk(self.model_to_test, self.args.max_capacity_prompt, mode='gemfilter') 
            elif self.args.mode == 'adakv':
                from baseline.adakv.adaptive_snapkv.snapkv_utils import compress
                compress(self.model_to_test, self.args) 
            elif self.args.mode == 'headkv':
                from baseline.headkv.headkv.snapkv_utils import compress
                compress(self.model_to_test, self.args)  
                    
            input = self.enc(prompt, return_tensors="pt").to(self.model_to_test.device)
            context_length = input.input_ids.shape[-1]
            with torch.no_grad():
                if self.args.mode == 'gemfilter':
                    response = gemfilter_generate_selection(input['input_ids'], input['attention_mask'], 
                        self.model_to_test, self.enc, max_gen_len=50, select_layer_idx=self.args.filter_idx)
                else:
                    output = self.model_to_test.generate(
                                                        **input,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        temperature=1.0,
                                                        top_p=1.0,
                                                        max_new_tokens=50,
                                                        )[0]
                    response = self.enc.decode(output[context_length:], skip_special_tokens=True).strip()
                
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        score = scorer.score(self.needle, response)['rouge1'].fmeasure * 10

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            logging.info(f"-- Test Summary -- ")
            logging.info(f"Duration: {test_elapsed_time:.1f} seconds")
            logging.info(f"Context: {context_length} tokens")
            logging.info(f"Depth: {depth_percent}%")
            logging.info(f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            context_path = os.path.join(self.save_path, 'context')
            Path(context_path).mkdir(parents=True, exist_ok=True)  
            context_path = os.path.join(context_path, f'{context_file_location}_context.txt')
            with open(context_path, 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the result to file for retesting
            result_path = os.path.join(self.save_path, 'result')
            Path(result_path).mkdir(parents=True, exist_ok=True)  
            result_path = os.path.join(result_path, f'{context_file_location}_results.json')
            with open(result_path, 'w') as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = os.path.join(self.save_path, 'result')
        logging.info("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "Phi3", "GLM"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is 
            period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            logging.info("Insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "Phi3", "GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "Phi3", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "Phi3", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        logging.info("\n")
        logging.info("Starting Needle In A Haystack Testing...")
        logging.info(f"- Model: {self.model_name}")
        logging.info(f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        logging.info(f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        logging.info(f"- Needle: {self.needle.strip()}")
        logging.info("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


def main(args):
    set_seed(args.seed)
 
    if args.save_path:
        args.save_path = os.path.join(f"outputs/{args.model}/needle", args.save_path)
        Path(args.save_path).mkdir(parents=True, exist_ok=True)  
    else:
        tm = time.localtime(time.time())
        f_name = f"{tm.tm_year}_{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{tm.tm_sec}"
        args.save_path = os.path.join(f"outputs/{args.model}/needle", f_name)
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    utils.config_logging(os.path.join(args.save_path, f'process.log'))
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)

    if args.context_length is not None:
        args.context_length = np.array(args.context_length)
    else:
        args.context_length = None
    
    ht = LLMNeedleHaystackTester(model_name=args.model,
                                 model_provider=args.model_provider,
                                 needle=args.needle,
                                 retrieval_question=args.retrieval_question,
                                 context_lengths=args.context_length,
                                 save_contexts=False,
                                 save_results=True,
                                 save_path=args.save_path,
                                 args=args)

    ht.start_test(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--save_path", default="", type=str, help="Path to save the output")

    # KV Compression
    parser.add_argument("--mode", type=str, default="fastkv", choices=["fullkv", "fastkv", "snapkv", "gemfilter", "adakv", "headkv"])
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", type=str, default="avgpool")
    
    # FastKV
    parser.add_argument("--tsp_idx", type=int, default=15)
    parser.add_argument("--tsp_len", type=int, default=2048)

    # GemFilter
    parser.add_argument("--filter_idx", type=int, default=13)
    
    # AdaKV
    parser.add_argument("--skip", type=int, default=-1)
    parser.add_argument('--floor_alpha', type=float, default=0.2)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--pyram', action='store_true')
    parser.add_argument('--pyram_beta', default=20,type=int)
    parser.add_argument('--gqa_support', action='store_true')

    # HeadKV
    parser.add_argument("--method", type=str, default='ReasonKV', choices=['ReasonKV'])
    parser.add_argument("--head_choice", type=str, default='reason', choices=['copy', 'reason'])
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--temp', type=float, default=1.0)
    
    # Evaluation
    parser.add_argument('-s', '--s_len', default=0, metavar='N', type=int)
    parser.add_argument('-e', '--e_len', default=128000, metavar='N', type=int)
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument("--context_length", nargs='+', type=int, 
                        default=[8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000, 72000, 80000, 88000, 96000, 104000, 112000, 120000, 128000])
    parser.add_argument("--needle", type=str, 
                        default="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n")
    parser.add_argument("--retrieval_question", type=str, 
                        default="What is the best thing to do in San Francisco?") 
    args = parser.parse_args()

    main(args)