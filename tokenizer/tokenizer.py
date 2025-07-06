from typing import List, Union
import json

class Tokenizer:
    def __init__(self):
        self.merge_dict = None
        self.vocab = None

    def _get_stats(self, ids: List, stats_dict = None) -> dict:
        """
        Given a list of integers, return a dictionary to give the count of the pairs coming consecutively
        """
        stats_dict = {} if stats_dict is None else stats_dict
        for pair in zip(ids, ids[1:]):
            stats_dict[pair] = stats_dict.get(pair, 0) + 1
        return stats_dict
    
    def _merge(self, ids, pair, new_idx):
        """Replace the pair at all the places with the new index
        """
        if len(ids) == 1:
            return ids
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _build_vocab(self):
        """Function builds the vocab dict mapping each token to its raw bytes"""
        self.vocab = {id : bytes([id]) for id in range(256)}
        for idx in self.merge_dict:
            # print(self.merge_dict[idx])
            self.vocab[idx] = self.vocab[self.merge_dict[idx][0]] + self.vocab[self.merge_dict[idx][1]]
            # print(self.vocab)
        print(f"Vocabulary has been built internally of length {len(self.vocab)}, ready to encode and decode")
        
    def train(self, train_text: str, n_vocab : int, merge_dict_name = "merge_dict") -> dict:
        """Function will take a train_text on which the BPE tokenizer will get trained
        Parameters:
        train_text: single python string
        n_vocab: size of the vocabulary to be built
        """
        new_merges = n_vocab - 256
        train_raw_bytes = list(train_text.encode('utf-8'))
        print(f"Length of raw bytes: {len(train_raw_bytes)}")
        i = 0
        merge_dict = {}
        while i < new_merges:
            stats_dict = self._get_stats(train_raw_bytes)
            top_pair = max(stats_dict, key = stats_dict.get)
            new_token = 256 + i
            train_raw_bytes = self._merge(train_raw_bytes, top_pair, 256 + i)
            merge_dict[new_token] = top_pair
            i += 1
        print(f"Length of merged bytes: {len(train_raw_bytes)}")
        # need to save merge_dict
        merge_dict_path = f"{merge_dict_name}.json"
        with open(merge_dict_path, 'w') as file:
            json.dump(merge_dict, file)
        print(f"Merge dict has been save on the path: {merge_dict_path}")
        self.merge_dict = merge_dict
        self._build_vocab()

    def from_pretrained(self, merge_dict_path):
        with open(merge_dict_path, "r") as file:
            self.merge_dict = json.load(file)
        # When laoding a saved json, all the object's key gets converted into string
        self.merge_dict = {int(key) : val for key, val in self.merge_dict.items()}
        self._build_vocab()
    
    def encode(self, s : str) -> List[int]:
        """Function takes a single string and encodes it into bytes"""
        s_raw_bytes = list(s.encode('utf-8'))
        print(f"Length of raw bytes: {len(s_raw_bytes)}")
        for idx in self.merge_dict:
            s_raw_bytes = self._merge(s_raw_bytes, self.merge_dict[idx], idx)
        print(f"Length of final merged bytes: {len(s_raw_bytes)}")
        return s_raw_bytes
    
    def decode(self, ids : List[int]) -> Union[str, List[str]]:
        """Function will take a sequence of bytes and decode it back to unicode codepoints"""
        if self.vocab is not None:
            bts = b"".join([self.vocab[i] for i in ids])
            bts = bts.decode('utf-8', errors = "replace")
            return bts
        else:
            raise ValueError("Vocab has not been built, please built it first and then call decode")