from dataclasses import dataclass
import os
from pyexpat.errors import messages
import re
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Sampler
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from metanetwork_family import Metanetwork

import random
from collections import defaultdict
from typing import Optional
import numpy as np
from copy import deepcopy
from datasets import Column
import json
from datasets import Dataset as HFDataset
from utils.myddp import is_main_process, barrier

# ---------------------------
# Mock dataset for demo
# ---------------------------
def create_mock_dataset() -> Tuple[List[str], List[str]]:
    texts = [
        "1231",
        "2342",
        "3453",
        "4564",
        "5675",
        "6786",
        "7897",
        "8908",
        "9019",
        "0120",
    ] * 50
    df = pd.DataFrame({'text': texts})
    train_texts, val_texts = train_test_split(df['text'], test_size=0.1, random_state=42)
    return train_texts.tolist(), val_texts.tolist()


# ---------------------------
# Dataset
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer = None):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"text": str(self.texts[idx])}

class GroupTextDataset(Dataset):
    """
    DDP-safe dataset:
      - __init__ only loads cache
      - preprocess() (instance) builds and saves group_idx cache using self.texts
    """

    def __init__(
        self,
        texts,
        tokenizer,
        conversation_max_len: int,
        cache_dir: str,
        cache_name: str,
        map_num_proc: int = 16,
        map_batch_size: int = 2048,
        num_cache: int = 100,
        preprocess_mode: bool = False,
        overwrite: bool = False,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.conversation_max_len = conversation_max_len
        self.cache_dir = cache_dir
        self.cache_name = cache_name

        # preprocess settings stored on instance
        self.map_num_proc = map_num_proc
        self.map_batch_size = map_batch_size
        self.num_cache = num_cache

        self.cache_path = os.path.join(
            cache_dir, f"{cache_name}_group_idx_{conversation_max_len}.json"
        )
        
        if preprocess_mode:
            self.preprocess(overwrite=overwrite)

        # ---- init ONLY loads cache ----
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"Cache not found: {self.cache_path}\n"
                f"Create it first by calling dataset.preprocess() "
                f"(single process / rank0 before DDP)."
            )

        with open(self.cache_path, "r") as f:
            self.group_idx = json.load(f)

        print(
            f"[GroupTextDataset] Loaded {len(self.group_idx)} groups "
            f"for {len(self.texts)} texts. max_len={conversation_max_len}"
        )

    # ------------------------------------------------------------------ #
    # Instance preprocess: compute group_idx + save to cache
    # ------------------------------------------------------------------ #
    def preprocess(self, overwrite: bool = False) -> List[List[int]]:
        """
        Build group_idx from self.texts and save to self.cache_path.

        Call ONCE before DDP training (single process / rank0).
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.exists(self.cache_path) and not overwrite:
            with open(self.cache_path, "r") as f:
                self.group_idx = json.load(f)
            print(f"[preprocess] Cache exists, loaded: {self.cache_path}")
            return self.group_idx

        print("[preprocess] Creating group_idx...")

        # # ----------------- base_len & chat_len (same logic as before) ----------------- #
        # test_q = "who is adam ?"
        # test_a = "I don't know"
        # message_1 = [
        #     {"role": "user", "content": f"{test_q}"},
        #     {"role": "assistant", "content": f"{test_a}"},
        # ]
        # input_enc_1 = self.tokenizer.apply_chat_template(
        #     message_1,
        #     add_generation_prompt=False,
        #     tokenize=True,
        #     return_tensors="pt",
        #     return_dict=True,
        #     enable_thinking=False,
        # )
        # len1 = len(input_enc_1["input_ids"][0])

        # message_2 = message_1 * 2
        # input_enc_2 = self.tokenizer.apply_chat_template(
        #     message_2,
        #     add_generation_prompt=False,
        #     tokenize=True,
        #     return_tensors="pt",
        #     return_dict=True,
        #     enable_thinking=False,
        # )
        # len2 = len(input_enc_2["input_ids"][0])

        # len3 = len(self.tokenizer.tokenize(test_q)) + len(
        #     self.tokenizer.tokenize(test_a)
        # )

        self.base_len = 0
        self.chat_len = 11

        # ----------------- FAST PART: compute token lengths with HF map ----------------- #
        print("[preprocess] Computing token lengths with HF Dataset.map...")
        token_lens = self._compute_token_lengths_with_hf_dataset()

        max_body_len = self.conversation_max_len - self.base_len

        self.group_idx: List[List[int]] = []
        cache_group_idx = [[] for _ in range(self.num_cache)]
        cache_left_len = [max_body_len for _ in range(self.num_cache)]

        for i, tok_len in enumerate(token_lens):
            if i % 10000 == 0:
                print(f"[preprocess] processing {i}/{len(token_lens)}")

            l = int(tok_len) + self.chat_len

            if l > max_body_len:
                self.group_idx.append([i])
                continue

            success = False
            for j, leftl in enumerate(cache_left_len):
                if l <= leftl:
                    cache_group_idx[j].append(i)
                    cache_left_len[j] -= l
                    success = True
                    break

            if not success:
                t = int(np.argmin(cache_left_len))
                if cache_group_idx[t]:
                    self.group_idx.append(cache_group_idx[t])
                cache_group_idx[t] = [i]
                cache_left_len[t] = max_body_len - l

        for j in range(self.num_cache):
            if cache_group_idx[j]:
                self.group_idx.append(cache_group_idx[j])

        with open(self.cache_path, "w") as f:
            json.dump(self.group_idx, f)

        print(f"[preprocess] Saved group_idx to {self.cache_path}")
        print(
            f"[preprocess] Total {len(self.group_idx)} groups including "
            f"{len(self.texts)} texts created for max_len={self.conversation_max_len}."
        )
        return self.group_idx

    # ------------------------------------------------------------------ #
    # Uses self.texts directly (NOT static)
    # ------------------------------------------------------------------ #
    def _compute_token_lengths_with_hf_dataset(self) -> np.ndarray:
        hf_dataset = HFDataset.from_dict(
            {"text": [str(t) for t in self.texts]}
        )

        def compute_len(batch):
            enc = self.tokenizer(
                batch["text"],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return {"tok_len": [len(ids) for ids in enc["input_ids"]]}

        hf_dataset = hf_dataset.map(
            compute_len,
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.map_num_proc,
            desc="Computing token lengths",
            writer_batch_size=10,
        )

        return np.array(hf_dataset["tok_len"], dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.group_idx)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"textlist": [str(self.texts[i]) for i in self.group_idx[idx]]}

class SquadDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        answer = [str(ans).strip() for ans in self.data[idx]['answers']['text']]
        for i in range(len(answer)):
            if answer[i][0].islower():
                answer[i] = answer[i][0].upper() + answer[i][1:]
        return {"evidence": str(self.data[idx]['context']).strip(), "question": str(self.data[idx]['question']).strip(), "answer": answer}

class GroupedSquadDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        context_len: Optional[int] = None,
        sep: str = '<|endoftext|>',
        name: str = "Test",
        seed: int = 42,
    ):
        self.name = f"[GroupedSquadDataset: {name}]"
        self.tokenizer = tokenizer
        self.sep = sep
        self.context_len = context_len
        self.data = data
        self.seed = seed
        
        self.shuffle()
    
    def shuffle(self):
        # Apply seed if provided (for determinism)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        text_to_idx = defaultdict(list)
        for i, ex in enumerate(self.data):
            ctx = str(ex["context"]).strip()
            text_to_idx[ctx].append(i)

        all_context_list = deepcopy(list(text_to_idx.keys()))
        self.text_to_idx = text_to_idx
        
        if self.context_len is None or self.context_len <= 0:
            self.groups = [[ctx] for ctx in all_context_list]
        else:
            num_tokens = len(self.tokenizer(self.sep.join(all_context_list))["input_ids"])
            num_groups = (num_tokens + self.context_len - 1) // self.context_len

            random.shuffle(all_context_list)
            context_list_per_group = np.array_split(all_context_list, num_groups)
            self.groups = [[str(s) for s in arr] for arr in context_list_per_group]
            
        self.idx_to_groupidx = {}
        for group_idx, ctx_list in enumerate(self.groups):
            for ctx in ctx_list:
                for ex_idx in text_to_idx[ctx]:
                    self.idx_to_groupidx[ex_idx] = group_idx
        
        self.group_token_num = []
        for group in self.groups:            
            token_num = len(self.tokenizer(self.sep.join(group))["input_ids"])
            self.group_token_num.append(token_num)
            
        print(f"{self.name}: {len(self.groups)} groups created from {len(self.data)} examples.")
        print(f"{self.name}: Average context token length: {np.mean(self.group_token_num):.2f}, "
              f"Max context token length: {np.max(self.group_token_num)}, "
              f"Min context token length: {np.min(self.group_token_num)}")
        print(f"{self.name}: Top 20 largest context token lengths: {sorted(self.group_token_num, reverse=True)[:20]}")
        print(f"{self.name}: Average contexts per group: {len(self.data) / len(self.groups):.2f}")
        
        for group in self.groups:
            for ctx in group:
                assert not ctx.startswith(" "), f"Context has leading space: '{ctx}'"
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        group = self.groups[self.idx_to_groupidx[idx]]
        evidence = self.sep.join(list(random.sample(group, len(group))))
        answer = [str(ans).strip() for ans in self.data[idx]['answers']['text']]
        for i in range(len(answer)):
            if answer[i][0].islower():
                answer[i] = answer[i][0].upper() + answer[i][1:]
        return {"evidence": str(evidence).strip(), "question": str(self.data[idx]['question']).strip(), "answer": answer}
    
class HotpotqaDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        contest_list = []
        item = self.data[index]
        for sentences in item['context']['sentences']:
            contest_list.append("".join(sentences))
        context = "\n\n".join(contest_list)
        return {"evidence": context, "question": item['question'].strip(), "answer": [item['answer'].strip()]}

class MusiqueDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        paragraphs = item['paragraphs']
        qd = item['question_decomposition']
        context_list = []
        for p in paragraphs:
            context_list.append(p['paragraph_text'])
        context = "\n\n".join(context_list)
        question = item['question'].strip()
        answer_aliases = [t.strip() for t in item['answer_aliases']]
        answer = [item['answer'].strip()] + answer_aliases
        return {"evidence": context, "question": question, "answer": answer}

class MsmarcoDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        data = list(data)
        new_data = []
        for item in data:
            passages = item['passages']
            if sum(passages['is_selected']) == 0 or len(item['answers']) == 0:
                continue
            new_data.append(item)
        self.data = new_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        passages = item['passages']
        context = "\n\n".join([passages['passage_text'][i] for i in range(len(passages['passage_text']))])
        question = item['query']
        answer = item['answers']
        return {"evidence": context, "question": question.strip(), "answer": answer}
    
class IFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        group_idx_path: str,
        use_exceed: bool = True,
    ):
        self.item_list = json.load(open(data_path, "r"))
        group_idx_file = json.load(open(group_idx_path, "r"))
        self.group_idx = group_idx_file["group_idx"]
        self.exceed_idx_list = set(group_idx_file["exceed_idx_list"])
        if not use_exceed:
            for i, list in enumerate(self.group_idx):
                new_list = [idx for idx in list if idx not in self.exceed_idx_list]
                self.group_idx[i] = new_list
        self.group_idx = [list for list in self.group_idx if len(list) > 0]
        if is_main_process():
            print(f"[IFTDataset] Loaded {len(self.group_idx)} groups from {data_path}, use_exceed={use_exceed}")
         
    def __len__(self):
        return len(self.group_idx)

    def __getitem__(self, idx) -> Dict[str, Any]:
        idx_list = self.group_idx[idx]
        contexts = [self.item_list[i]['context'] for i in idx_list]
        conversations = [self.item_list[i]['conversations'] for i in idx_list]
        random.shuffle(contexts)
        random.shuffle(conversations)
        final_context = "<|endoftext|>".join(contexts)
        final_conversation = []
        for conv in conversations:
            final_conversation.extend(conv)
        return {"evidence": final_context, "conversations": final_conversation}
    

class IFTC1QADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_context_len: int = 3000,
        max_conversation_len: int = 256,
        use_exceed: bool = False,
    ):
        self.item_list = json.load(open(data_path, "r"))
        if not use_exceed:
            self.item_list = [item for item in self.item_list if item['contextlen'] <= max_context_len and item['conversationlen'] <= max_conversation_len]
        if is_main_process():
            print(f"[IFTC1QADataset] Loaded {len(self.item_list)} items from {data_path}, use_exceed={use_exceed}")
         
    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.item_list[idx]
        return {"evidence": item['context'], "conversations": item['conversations'], "contextlen": item['contextlen'], "conversationlen": item['conversationlen']}

class MQADataset(Dataset):
    def __init__(
        self,
        data,
    ):
        self.data = data
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        contexts = item['context']
        conversations = item['conversations']
        final_conversations = []
        for conv in conversations:
            final_conversations.extend([{'role': 'user', 'content': conv['question']}, {'role': 'assistant', 'content': conv['answer']}])
        questions = []
        answers = []
        for conv in conversations:
            questions.append(conv['question'])
            answers.append(conv['answer'])
        return {"evidence": contexts, "conversations": final_conversations, "questions": questions, "answers": answers}

class HumanDataset(Dataset):
    def __init__(
        self,
        data,
    ):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        contexts = item['context']
        questions = item['questions']
        return {"evidence": contexts, "questions": questions}
    
class SFTDataset(Dataset):
    def __init__(
        self,
        data,
    ):
        self.data = data  
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        results = {"evidence": self.data[idx]['context']}
        orig_conversation = []
        orig_questions = []
        orig_answers = []
        for conv in self.data[idx]['orig_conversation']:
            orig_conversation.extend([{'role': 'user', 'content': conv['question']}, {'role': 'assistant', 'content': conv['answer']}])
            orig_questions.append(conv['question'])
            orig_answers.append(conv['answer'])
        results["orig_conversation"] = orig_conversation
        results["orig_questions"] = orig_questions
        results["orig_answers"] = orig_answers
        for i in range(1, 11):
            conversation = []
            questions = []
            answers = []
            for conv in self.data[idx][f'conversation{i}']:
                conversation.extend([{'role': 'user', 'content': conv['question']}, {'role': 'assistant', 'content': conv['answer']}])
                questions.append(conv['question'])
                answers.append(conv['answer'])
            results[f'conversation{i}'] = conversation
            results[f'questions{i}'] = questions
            results[f'answers{i}'] = answers
        return results

# ---------------------------
# Collator with dynamic padding and label masking
# ---------------------------
@dataclass
class BaseCollator:
    def __post_init__(self):
        if "pretrain" in self.cfg:
            self.completion_freq = self.cfg.pretrain.completion_freq
            self.max_completion_ratio = self.cfg.pretrain.max_completion_ratio
            self.min_completion_ratio = self.cfg.pretrain.min_completion_ratio
        self.thinkend_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.eot = '<|endoftext|>'
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        self.imstart_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.imend_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.SYSTEM_PROMPT = "You are a concise assistant. Output only the final answer, in a few words, as short as possible. No explanations. Do not output anything else."

    def mask_label(self, labels):
        masks = torch.zeros_like(labels)
        for i, id in enumerate(labels):
            last_imend = self.conversation_max_length
            for j in range(len(id) - 1, 0, -1):
                if id[j].item() == self.imend_token_id:
                    last_imend = j
                elif id[j].item() == self.assistant_token_id and id[j - 1] == self.imstart_token_id:
                    masks[i, j+2: last_imend+2] = 1
        labels = labels.masked_fill(masks == 0, -100)
        return labels
        

@dataclass
class PretrainCollator(BaseCollator):
    tokenizer: Any
    cfg: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    metatrain: bool = False
    
    def __post_init__(self):
        super().__post_init__()
    
    def split_text(self, text):
        t = text.split()
        if len(t) < 2:
            return text, "Nothing to complete."

        ratio = 1.0 - random.uniform(self.min_completion_ratio, self.max_completion_ratio)
        split_index = round(len(t) * ratio)

        left = t[:split_index]
        right = t[split_index:]

        if not right:  # ensure right is not empty
            left, right = t[:-1], t[-1:]
        elif not left:
            left, right = t[:1], t[1:]
        
        return ' '.join(left), ' '.join(right)
    

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]
        
        if self.metatrain:            
            t = random.random()
            if t < self.completion_freq:
                splits = [self.split_text(text) for text in texts]
                evidence_texts = [split[0] for split in splits]
                answer_texts = texts
                # answer_texts = [split[1] for split in splits]
                messages = [[
                    {"role": "user", "content": f"<COMP>"},
                    {"role": "assistant", "content": f"{answer}"}
                ] for answer in answer_texts]
            else:
                evidence_texts = texts
                answer_texts = texts
                messages = [[
                    {"role": "user", "content": f"<RECON>"},
                    {"role": "assistant", "content": f"{answer}"}
                ] for answer in answer_texts]
        else:
            raise NotImplementedError("metatrain=False mode is not implemented in PretrainCollator.")

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        answer_enc = self.tokenizer(
            answer_texts,
            max_length=self.conversation_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_ids = answer_enc["input_ids"]
        answer_attention_mask = answer_enc["attention_mask"]


        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        labels = None
        if self.metatrain:
            labels = input_ids.clone()
            labels = self.mask_label(labels)
        
        # if is_main_process():
        #     res = "input"
        #     tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        #     for i, t in enumerate(tokens):
        #         res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        #     res = f"{res}\nevidence"
        #     tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        #     for i, t in enumerate(tokens):
        #         res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        #     print(res)
        # barrier()
        # exit()
        
        # # Debug print for the first item
        # first_input_ids = input_ids[0]
        # first_labels = labels[0]
        # first_evidence_ids = evidence_ids[0]
        # first_input_text = self.tokenizer.decode(first_input_ids, skip_special_tokens=False)
        # first_evidence_ids = [i for i in first_evidence_ids if i != self.tokenizer.pad_token_id]
        # first_evidence_text = self.tokenizer.decode(first_evidence_ids, skip_special_tokens=False)
        # print("\n=== First input sentence (meta-train mode) ===")
        # print(first_input_text)
        # print("\n=== First evidence sentence (meta-train mode) ===")
        # print(first_evidence_text)
        # # tokens = self.tokenizer.convert_ids_to_tokens(first_input_ids)
        # # print("\n=== Tokens, labels, and corresponding words ===")
        # # for i, (tok, lab) in enumerate(zip(tokens, first_labels.tolist())):
        # # # for i, tok in enumerate(tokens):
        # #     # decode the token alone to see its text segment
        # #     word_piece = self.tokenizer.decode(
        # #         [self.tokenizer.convert_tokens_to_ids(tok)],
        # #         skip_special_tokens=True,
        # #         clean_up_tokenization_spaces=False,
        # #     )
        # #     # show both raw token, decoded string, and label
        # #     # print(f"{tok:<20} | {word_piece:<15} | mask={input_attention_mask[0][i]}")
        # #     print(f"{tok:<20} | {word_piece:<15} | label={lab} | mask={input_attention_mask[0][i]}")
        # exit()
                
        return {
            "evidence": texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "answers": texts,
            "answer_ids": answer_ids,
            "answer_attention_mask": answer_attention_mask,
            "questions": ["Please repeat what you have read."] * len(texts),
        }
        
@dataclass
class TestPretrainCollator(BaseCollator):
    tokenizer: Any
    cfg: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    metatrain: bool = False
    mode: str = "recon"
    
    def __post_init__(self):
        super().__post_init__()
    
    def split_text(self, text):
        t = text.split()
        if len(t) < 2:
            return text, "Nothing to complete."

        ratio = 1.0 - random.uniform(self.min_completion_ratio, self.max_completion_ratio)
        split_index = round(len(t) * ratio)

        left = t[:split_index]
        right = t[split_index:]

        if not right:  # ensure right is not empty
            left, right = t[:-1], t[-1:]
        elif not left:
            left, right = t[:1], t[1:]
        
        return ' '.join(left), ' '.join(right)
    

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]
        
        if self.mode == "comp":
            splits = [self.split_text(text) for text in texts]
            evidence_texts = [split[0] for split in splits]
            answer_texts = texts
            # answer_texts = [split[1] for split in splits]
            messages = [[
                {"role": "user", "content": f"<COMP>"},
            ] for answer in answer_texts]
            label_messages = [[
                {"role": "user", "content": f"<COMP>"},
                {"role": "assistant", "content": f"{answer}"}
            ] for answer in answer_texts]
        elif self.mode == "recon":
            evidence_texts = texts
            answer_texts = texts
            messages = [[
                {"role": "user", "content": f"<RECON>"},
            ] for answer in answer_texts]
            label_messages = [[
                {"role": "user", "content": f"<RECON>"},
                {"role": "assistant", "content": f"{answer}"}
            ] for answer in answer_texts]
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented in TestPretrainCollator.")

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        
        answer_enc = self.tokenizer(
            answer_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_ids = answer_enc["input_ids"]
        answer_attention_mask = answer_enc["attention_mask"]

        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=9,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        
        label_enc = self.tokenizer.apply_chat_template(
                label_messages,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        labels = label_enc["input_ids"]
        full_input_ids = labels
        full_input_attention_mask = label_enc["attention_mask"]
        labels = self.mask_label(labels)
        
        # if is_main_process():
        #     res = "input"
        #     tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        #     for i, t in enumerate(tokens):
        #         res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        #     res = f"{res}\nevidence"
        #     tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        #     for i, t in enumerate(tokens):
        #         res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        #     print(res)
        # barrier()
        # exit()
        
        # # Debug print for the first item
        # first_input_ids = input_ids[0]
        # first_labels = labels[0]
        # first_evidence_ids = evidence_ids[0]
        # first_input_text = self.tokenizer.decode(first_input_ids, skip_special_tokens=False)
        # first_evidence_ids = [i for i in first_evidence_ids if i != self.tokenizer.pad_token_id]
        # first_evidence_text = self.tokenizer.decode(first_evidence_ids, skip_special_tokens=False)
        # print("\n=== First input sentence (meta-train mode) ===")
        # print(first_input_text)
        # print("\n=== First evidence sentence (meta-train mode) ===")
        # print(first_evidence_text)
        # # tokens = self.tokenizer.convert_ids_to_tokens(first_input_ids)
        # # print("\n=== Tokens, labels, and corresponding words ===")
        # # for i, (tok, lab) in enumerate(zip(tokens, first_labels.tolist())):
        # # # for i, tok in enumerate(tokens):
        # #     # decode the token alone to see its text segment
        # #     word_piece = self.tokenizer.decode(
        # #         [self.tokenizer.convert_tokens_to_ids(tok)],
        # #         skip_special_tokens=True,
        # #         clean_up_tokenization_spaces=False,
        # #     )
        # #     # show both raw token, decoded string, and label
        # #     # print(f"{tok:<20} | {word_piece:<15} | mask={input_attention_mask[0][i]}")
        # #     print(f"{tok:<20} | {word_piece:<15} | label={lab} | mask={input_attention_mask[0][i]}")
        # exit()
                
        return {
            "evidence": texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "full_input_ids": full_input_ids,
            "full_input_attention_mask": full_input_attention_mask,
            "labels": labels,
            "answers": texts,
            "answer_ids": answer_ids,
            "answer_attention_mask": answer_attention_mask,
            "questions": [f"{self.mode}"] * len(texts),
        }
        
@dataclass
class GroupPretrainCollator(BaseCollator):
    tokenizer: Any
    cfg: Any
    context_max_length: int = 1024 ############change###############
    conversation_max_length: int = 1024
    metatrain: bool = False
    
    def __post_init__(self):
        super().__post_init__()
    
    def split_text(self, text):
        t = text.split()
        if len(t) < 2:
            return text, "Nothing to complete."

        ratio = 1.0 - random.uniform(self.min_completion_ratio, self.max_completion_ratio)
        split_index = round(len(t) * ratio)

        left = t[:split_index]
        right = t[split_index:]

        if not right:  # ensure right is not empty
            left, right = t[:-1], t[-1:]
        elif not left:
            left, right = t[:1], t[1:]
        
        return ' '.join(left), ' '.join(right)
    

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        textlists = [ex["textlist"] for ex in batch]
        if self.metatrain:
            user_texts_list = []
            evidence_texts_list = []
            answer_texts_list = []
            for texts in textlists:            
                tlist = [random.random() for _ in range(len(texts))]
                evidence_texts = []
                answer_texts = []
                user_texts = []
                for i, t in enumerate(tlist):
                    if t < self.completion_freq:
                        split = self.split_text(texts[i])
                        evidence_texts.append(split[0])
                        answer_texts.append(texts[i])
                        user_texts.append("<COMP>")
                    else:
                        evidence_texts.append(texts[i])
                        answer_texts.append(texts[i])
                        user_texts.append("<RECON>")
                evidence_texts_list.append(evidence_texts)
                answer_texts_list.append(answer_texts)
                user_texts_list.append(user_texts)
        else:
            raise NotImplementedError("metatrain=False mode is not implemented in GroupPretrainCollator.")

        evidence_texts_all = [self.eot.join(random.sample(evidence_texts, len(evidence_texts))) for evidence_texts in evidence_texts_list]
        evidence_enc = self.tokenizer(
            evidence_texts_all,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]

        messages = []
        for i in range(len(textlists)):
            indices = list(range(len(textlists[i])))
            random.shuffle(indices)
            msg = []
            for id in indices:
                msg.append({"role": "user", "content": f"{user_texts_list[i][id]}"} )
                msg.append({"role": "assistant", "content": f"{answer_texts_list[i][id]}"} )
            messages.append(msg)
            
        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        labels = None
        if self.metatrain:
            labels = input_ids.clone()
            labels = self.mask_label(labels)
        
        # if is_main_process():
        #     res = "input"
        #     tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        #     for i, t in enumerate(tokens):
        #         res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        #     res = f"{res}\nevidence"
        #     tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        #     for i, t in enumerate(tokens):
        #         res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        #     print(res)
        #     exit()
        
        # # Debug print for the first item
        # first_input_ids = input_ids[0]
        # first_labels = labels[0]
        # first_evidence_ids = evidence_ids[0]
        # first_input_text = self.tokenizer.decode(first_input_ids, skip_special_tokens=False)
        # first_evidence_ids = [i for i in first_evidence_ids if i != self.tokenizer.pad_token_id]
        # first_evidence_text = self.tokenizer.decode(first_evidence_ids, skip_special_tokens=False)
        # print("\n=== First input sentence (meta-train mode) ===")
        # print(first_input_text)
        # print("\n=== First evidence sentence (meta-train mode) ===")
        # print(first_evidence_text)
        # # tokens = self.tokenizer.convert_ids_to_tokens(first_input_ids)
        # # print("\n=== Tokens, labels, and corresponding words ===")
        # # for i, (tok, lab) in enumerate(zip(tokens, first_labels.tolist())):
        # # # for i, tok in enumerate(tokens):
        # #     # decode the token alone to see its text segment
        # #     word_piece = self.tokenizer.decode(
        # #         [self.tokenizer.convert_tokens_to_ids(tok)],
        # #         skip_special_tokens=True,
        # #         clean_up_tokenization_spaces=False,
        # #     )
        # #     # show both raw token, decoded string, and label
        # #     # print(f"{tok:<20} | {word_piece:<15} | mask={input_attention_mask[0][i]}")
        # #     print(f"{tok:<20} | {word_piece:<15} | label={lab} | mask={input_attention_mask[0][i]}")
        # exit()
                
        return {
            "evidence": texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "questions": user_texts_list,
        }

@dataclass
class SquadCollator(BaseCollator):
    tokenizer: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    use_reference: bool = False
    metatrain: bool = False
    only_question: bool= False
    thinkend_token_id: int = None
    cfg: Any = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        questions = [ex["question"] for ex in batch]
        evidences = [ex["evidence"] for ex in batch]
        assert isinstance(batch[0]["answer"], list), "Answers should be a list of possible answers."
        answers = [str(random.choice(ex["answer"])) for ex in batch]
        full_answers = [ex["answer"] for ex in batch]

        evidence_enc = self.tokenizer(
            evidences,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        
        answer_enc = self.tokenizer(
            answers,
            max_length=self.conversation_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_ids = answer_enc["input_ids"]
        answer_attention_mask = answer_enc["attention_mask"]


        # if self.metatrain:            
        #     messages = [[
        #         {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
        #         {"role": "user", "content": f"Please answer the following question: {question}"},
        #         {"role": "assistant", "content": f"<think>I know the answer because I have read something about this.</think>\n{answer}"}
        #     ] for question, answer in zip(questions, answers)]
        # elif self.use_reference:
        #     messages = [[
        #         {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
        #         {"role": "user", "content": f"Please review the following reference materials.\n\nReference:\n{evidence}\n\nBased on the reference, answer this question:\n{question}"},
        #     ] for evidence, question in zip(evidences, questions)]
        # elif self.only_question:
        #     messages = [[
        #         {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
        #         {"role": "user", "content": f"Please answer the following question: {question}"},
        #     ] for question in questions]
        # else:
        #     messages = [[
        #         {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
        #         {"role": "user", "content": f"Please answer the following question: {question}"},
        #         {"role": "assistant", "content": f"<think>I know the answer because I have read something about this.</think>\n"}
        #     ] for question in questions]
        # SYSTEM_PROMPT = "You are a QA assistant. For every question, output ONLY the final answer. No explanation, no reasoning, no extra words, no punctuation unless necessary."
        if self.metatrain:            
            messages = [
                [
                    {"role": "user", "content": f"{question}"},
                    {"role": "assistant", "content": f"{answer}"}
                ]
                for question, answer in zip(questions, answers)
            ]
        elif self.use_reference:
            messages = [
                ([{"role": "system", "content": f"{self.SYSTEM_PROMPT}"}] if self.SYSTEM_PROMPT is not None else []) +
                [
                    {"role": "user", "content": f"Reference:\n{evidence}\n\nBased on the reference, answer this question:\n{question}"},
                ] 
                for evidence, question in zip(evidences, questions)]
        elif self.only_question:
            messages = [
                ([{"role": "system", "content": f"{self.SYSTEM_PROMPT}"}] if self.SYSTEM_PROMPT is not None else []) +
                [
                    {"role": "user", "content": f"{question}"},
                ] 
                for question in questions]
        else:
            messages = [
                [
                {"role": "user", "content": f"{question}"},
                ] 
                for question in questions]


        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True if not self.metatrain else False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length + 4 if (not self.metatrain and not self.use_reference and not self.only_question) else self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        labels = None
        if self.metatrain:
            labels = input_ids.clone()
            labels = self.mask_label(labels)
        elif not self.use_reference and not self.only_question:
            input_ids = input_ids[:, :-4]
            input_attention_mask = input_attention_mask[:, :-4]
            
        
        # res = "input"
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        # res = f"{res}\nevidence"
        # tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        # res = f"{res}\n\n"
        # print(res)
        # exit()
        
        # # Debug print for the first item
        # first_input_ids = input_ids[0]
        # first_labels = labels[0]
        # first_input_text = self.tokenizer.decode(first_input_ids, skip_special_tokens=False)
        # print("\n=== First input sentence (meta-train mode) ===")
        # print(first_input_text)
        # tokens = self.tokenizer.convert_ids_to_tokens(first_input_ids)
        # print("\n=== Tokens, labels, and corresponding words ===")
        # for i, (tok, lab) in enumerate(zip(tokens, first_labels.tolist())):
        # # for i, tok in enumerate(tokens):
        #     # decode the token alone to see its text segment
        #     word_piece = self.tokenizer.decode(
        #         [self.tokenizer.convert_tokens_to_ids(tok)],
        #         skip_special_tokens=True,
        #         clean_up_tokenization_spaces=False,
        #     )
        #     # show both raw token, decoded string, and label
        #     # print(f"{tok:<20} | {word_piece:<15} | mask={input_attention_mask[0][i]}")
        #     print(f"{tok:<20} | {word_piece:<15} | label={lab} | mask={input_attention_mask[0][i]}")
        # exit()
                
        return {
            "evidence": evidences,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "messages": messages,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "answers": answers,
            "full_answers": full_answers,
            "answer_ids": answer_ids,
            "answer_attention_mask": answer_attention_mask,
            "questions": questions,
        }

@dataclass
class IFTCollator(BaseCollator):
    tokenizer: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    cfg: Any = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        evidence_texts = [t['evidence'] for t in batch]
        conversation_texts = [t['conversations'] for t in batch]
        if isinstance(conversation_texts[0], Column):
            conversation_texts = list(conversation_texts[0])  # or conversation_texts[0][:]
        messages = [
            conversation for conversation in conversation_texts
        ]

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]

        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]

        labels = input_ids.clone()
        labels = self.mask_label(labels)
        
        # if (labels != -100).sum().item() == 0:        
        #     res = "input"
        #     tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        #     for i, t in enumerate(tokens):
        #         res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        #     res = f"{res}\nevidence"
        #     tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        #     for i, t in enumerate(tokens):
        #         res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        #     res = f"{res}\ncontext len:{batch[0]['contextlen']} conversation len:{batch[0]['conversationlen']}"
        #     res = f"{res}\n\n"
        #     print(res)
        #     exit()
                
        return {
            "evidence": evidence_texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
        }

@dataclass
class MQACollator(BaseCollator):
    tokenizer: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    cfg: Any = None
    sys_msg: bool = False
    no_evidence: bool = False

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        evidence_texts = [t['evidence'] for t in batch]
        conversation_texts = [t['conversations'] for t in batch]
        # if isinstance(conversation_texts[0], Column):
        #     conversation_texts = list(conversation_texts[0])  # or conversation_texts[0][:]
        if self.sys_msg:
            if self.no_evidence:
                PRMOPT = "You are a helpful assistant. Answer the question concisely with short words or phrases. Answer the question directly and output nothing else. Never say you don't know the answer. Never enter think mode."
                messages = [
                    [{"role": "system", "content": f"{PRMOPT}"}] + conversation
                    for conversation in conversation_texts
                ]
                initial_messages = [{"role": "system", "content": f"{PRMOPT}"} for _ in conversation_texts]
            else:
                PRMOPT = "You are a helpful assistant, answer the questions based on the given context. Each answer must be directly extractable from the context (i.e., an exact span or minor paraphrase for fluency). Do not invent information. Answer the question directly and output nothing else. Never enter think mode.\n\nContext: "
                messages = [
                    [{"role": "system", "content": f"{PRMOPT}{evidence}"}] + conversation
                    for evidence, conversation in zip(evidence_texts, conversation_texts)
                ]
                initial_messages = [{"role": "system", "content": f"{PRMOPT}{evidence}"} for evidence in evidence_texts]
        else:
            messages = [conversation for conversation in conversation_texts]
            initial_messages = [{} for _ in conversation_texts]

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]

        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        

        labels = input_ids.clone()
        labels = self.mask_label(labels)
            
        # res = "input"
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        # res = f"{res}\nevidence"
        # tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        # # res = f"{res}\ncontext len:{batch[0]['contextlen']} conversation len:{batch[0]['conversationlen']}"
        # res = f"{res}\n\n"
        # print(res, flush=True)
                
        return {
            "initial_messages": initial_messages,
            "evidence": evidence_texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "questions": [b['questions'] for b in batch],
            "answers": [b['answers'] for b in batch],
        }
        
@dataclass
class HumanCollator(BaseCollator):
    tokenizer: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    cfg: Any = None
    sys_msg: bool = False
    no_evidence: bool = False

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        evidence_texts = [t['evidence'] for t in batch]
        questions = [b['questions'] for b in batch]
        # if isinstance(conversation_texts[0], Column):
        #     conversation_texts = list(conversation_texts[0])  # or conversation_texts[0][:]
        if self.sys_msg:
            if self.no_evidence:
                PRMOPT = "You are a helpful assistant. Answer the question concisely with short words or phrases. Answer the question directly and output nothing else. Never say you don't know the answer. Never enter think mode."
                initial_messages = [{"role": "system", "content": f"{PRMOPT}"} for _ in questions]
            else:
                PRMOPT = "You are a helpful assistant, answer the questions based on the given context. Each answer must be directly extractable from the context (i.e., an exact span or minor paraphrase for fluency). Do not invent information. Answer the question directly and output nothing else. Never enter think mode.\n\nContext: "
                initial_messages = [{"role": "system", "content": f"{PRMOPT}{evidence}"} for evidence in evidence_texts]
        else:
            initial_messages = [{} for _ in questions]

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
            
        # res = "input"
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        # res = f"{res}\nevidence"
        # tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        # # res = f"{res}\ncontext len:{batch[0]['contextlen']} conversation len:{batch[0]['conversationlen']}"
        # res = f"{res}\n\n"
        # print(res, flush=True)
                
        return {
            "initial_messages": initial_messages,
            "evidence": evidence_texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "questions": [b['questions'] for b in batch],
        }

@dataclass
class SFTCollator(BaseCollator):
    tokenizer: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024
    cfg: Any = None
    sys_msg: bool = False
    no_evidence: bool = False

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        evidence_texts = [t['evidence'] for t in batch]
        
        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        
        conversation_texts = {}
        questions = {}
        answers = {}
        input_ids = {}
        labels = {}
        input_attention_mask = {}
        for i in range(1, 11):
            conversation_texts[i] = [t[f'conversation{i}'] for t in batch]
            questions[i] = [t[f'questions{i}'] for t in batch]
            answers[i] = [t[f'answers{i}'] for t in batch]
            
            messages = [conversation for conversation in conversation_texts[i]]

            input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
            input_ids[i] = input_enc["input_ids"]
            input_attention_mask[i] = input_enc["attention_mask"]
            labels[i] = input_ids[i].clone()
            labels[i] = self.mask_label(labels[i])
        
        orig_conversation_texts = [t['orig_conversation'] for t in batch]
        orig_questions = [t['orig_questions'] for t in batch]
        orig_answers = [t['orig_answers'] for t in batch]
        
        if self.sys_msg:
            if self.no_evidence:
                PRMOPT = "You are a helpful assistant. Answer the question concisely with short words or phrases. Answer the question directly and output nothing else. Never say you don't know the answer. Never enter think mode."
                messages = [
                    [{"role": "system", "content": f"{PRMOPT}"}] + conversation
                    for conversation in orig_conversation_texts
                ]
                initial_messages = [{"role": "system", "content": f"{PRMOPT}"} for _ in orig_conversation_texts]
            else:
                PRMOPT = "You are a helpful assistant, answer the questions based on the given context. Each answer must be directly extractable from the context (i.e., an exact span or minor paraphrase for fluency). Do not invent information. Answer the question directly and output nothing else. Never enter think mode.\n\nContext: "
                messages = [
                    [{"role": "system", "content": f"{PRMOPT}{evidence}"}] + conversation
                    for evidence, conversation in zip(evidence_texts, orig_conversation_texts)
                ]
                initial_messages = [{"role": "system", "content": f"{PRMOPT}{evidence}"} for evidence in evidence_texts]
        else:
            messages = [conversation for conversation in orig_conversation_texts]
            initial_messages = [{} for _ in orig_conversation_texts]
        
        orig_input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                max_length=self.conversation_max_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        orig_input_ids = orig_input_enc["input_ids"]
        orig_input_attention_mask = orig_input_enc["attention_mask"]
        orig_labels = orig_input_ids.clone()
        orig_labels = self.mask_label(orig_labels)
            
            
        # res = "input"
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {input_attention_mask[0][i]} label: {labels[0][i] if labels is not None else 'N/A'}"
        # res = f"{res}\nevidence"
        # tokens = self.tokenizer.convert_ids_to_tokens(evidence_ids[0])
        # for i, t in enumerate(tokens):
        #     res = f"{res}\n{i}: token_ids: {t} attention_mask: {evidence_attention_mask[0][i]}"
        # # res = f"{res}\ncontext len:{batch[0]['contextlen']} conversation len:{batch[0]['conversationlen']}"
        # res = f"{res}\n\n"
        # print(res, flush=True)
                
        return {
            "initial_messages": initial_messages,
            "evidence": evidence_texts,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "questions": {i: [b[f"questions{i}"] for b in batch] for i in range(1, 11)},
            "answers": {i: [b[f"answers{i}"] for b in batch] for i in range(1, 11)},
            "orig_input_ids": orig_input_ids,
            "orig_labels": orig_labels,
            "orig_input_attention_mask": orig_input_attention_mask,
            "orig_questions": orig_questions,
            "orig_answers": orig_answers,
        }