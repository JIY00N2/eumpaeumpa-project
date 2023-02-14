import logging
import os

import argparse
import pickle
import random
import math
import pandas as pd
import numpy as np
from datetime import datetime
import json
import random
import re



from transformers import BartForConditionalGeneration, BartConfig
from transformers import PreTrainedTokenizerFast

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


class Ft_Processor():

    def get_train_examples(self, data_dir):
        # pass
        return self.preprocessing(data_dir)

    def get_val_examples(self, data_dir):
        # pass
        return self.preprocessing(data_dir)

    def get_labels(self,):
        pass

    def read_json(self, data_dir):
        with open(data_dir, encoding="UTF8") as f:
            JSON_DATA = json.loads(f.read())
        return JSON_DATA

    def replace2abbrev(self, m):
        m = str(m.group())
        m = re.findall('\((.*?)\)', m)[0]
        return m

    def preprocessing(self, data_dir):
        json_data = self.read_json(data_dir)
        data_df = pd.DataFrame(columns=['uid', 'type', 'topic','partici_num','utter_num', 'context', 'summary'])
        uid = 0

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)

        for data in json_data['data'][0:args.train_data_num]:
            data_df.loc[uid, 'uid'] = uid
            data_df.loc[uid, 'type'] = data['header']['dialogueInfo']['type']
            data_df.loc[uid, 'topic'] = data['header']['dialogueInfo']['topic']
            data_df.loc[uid, 'partici_num'] = data['header']['dialogueInfo']['numberOfParticipants']
            data_df.loc[uid, 'utter_num'] = data['header']['dialogueInfo']['numberOfUtterances']
            context = ''
            for utter in data['body']['dialogue']:
                context = context + utter['utterance'] + ' '
                # if utter['utteranceID'] == 'U'+str(data['header']['dialogueInfo']['numberOfUtterances']):
                #    context = context + utter['utterance']

                # else:
                #    context = context + utter['utterance'] + '/ '

            context = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ·!』\\‘’|\(\)\[\]\<\>`\'…》]', '', context)  # 특수문자 제거
            context = re.sub('(([ㄱ-힣])\\2{1,})', '', context)
            context = emoji_pattern.sub(r'', context)
            data_df.loc[uid, 'context'] = context

            data_df.loc[uid, 'summary'] = re.sub('[ㄱ-힣]+\([^)]*\)', self.replace2abbrev, data['body']['summary'])
            uid += 1

        return data_df


class SummaryDataset(Dataset):
    def __init__(self, dataframe, max_seq_len, tokenizer) -> None:
        self.dataframe = dataframe
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len  # special token 포함해서 고려한 길이가 max_seq_len임

        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataframe.shape[0]

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)  # <s> 와 </s> token 포함된 상태에서 진행하는것임
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            input_id = input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]  # 1028개 까지만 수용
        return input_id, attention_mask

    def __getitem__(self, index):
        target_row = self.dataframe.iloc[index]
        context, summary = target_row['context'], target_row['summary']
        context_tokens = [self.bos_token] + \
                         self.tokenizer.tokenize(context) + [self.eos_token]
        summary_tokens = [self.bos_token] + \
                         self.tokenizer.tokenize(summary) + [self.eos_token]  # 애초에 summary token은 max_seq_len 보다 훨씬 짧음
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            context_tokens, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            summary_tokens, index)  # decoder에 넣기 위해 딱 맞는 형태로 갖춤
        # labels = self.tokenizer.convert_tokens_to_ids(summary_tokens[1:self.max_seq_len])
        labels = self.tokenizer.convert_tokens_to_ids(
            summary_tokens[1:(
                        self.max_seq_len + 1)])  # 아니 애초에 summary token의 길이보다 더 긴 indexing을 하는데 말이되나, labels을 만드는 의도가 머지 -> decoder에서 예측된 놈들이 나오는 것이니까 맞네 이게
        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                # for cross entropy loss masking
                labels += [-100]

        return {'input_ids': torch.tensor(encoder_input_id, dtype=torch.long),
                'attention_mask': torch.tensor(encoder_attention_mask, dtype=torch.long),
                'decoder_input_ids': torch.tensor(decoder_input_id[:128], dtype=torch.long),
                'decoder_attention_mask': torch.tensor(decoder_attention_mask[:128], dtype=torch.long),
                'labels': torch.tensor(labels[:128], dtype=torch.long)}

        '''
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id[:128], dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask[:128], dtype=np.float_),
                'labels': np.array(labels[:128], dtype=np.int_)}
        '''



class tts_model():
    def __init__(self, pretrained_model, model_path, device):
        self.model = BartForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model)
        self.model.load_state_dict(torch.load(model_path, map_location = device))
        pass

    def preprocessing(self,text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)

        context = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ·!』\\‘’|\(\)\[\]\<\>`\'…》]', '', text)  # 특수문자 제거
        context = re.sub('(([ㄱ-힣])\\2{1,})', '', context)
        context = emoji_pattern.sub(r'', context)

        return context

    def get_result(self, input):
        print(input)
        input_ids = self.tokenizer.encode(input, return_tensors='pt')

        summary_text_ids = self.model.generate(
            input_ids=input_ids,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id= self.model.config.eos_token_id,
            length_penalty=2.0,
            max_length=128,
            min_length=5,
            num_beams=4
        )

        predicted_txt = self.tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

        return predicted_txt

