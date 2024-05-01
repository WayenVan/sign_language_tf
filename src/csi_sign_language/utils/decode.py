from typing import Any
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F
import torchtext as tt
from typing import *

class CTCDecoder():
    
    def __init__(
        self, 
        vocab: tt.vocab.Vocab, 
        search_mode: Union[Literal['beam'], Literal['greedy']]='beam', 
        blank_id=0,
        batch_first=False,
        log_probs_input=True,
        beam_width=10
        ) -> None:
        self.vocab = vocab
        self.num_class = len(vocab)
        self.batch_first = batch_first
        self.log_probs_input= log_probs_input
        self.search_mode = search_mode
        self.blank_id = blank_id

        if search_mode == 'beam':
            self.beam_decoder = ctcdecode.CTCBeamDecoder(
                vocab.get_itos(), 
                beam_width=beam_width, 
                log_probs_input=self.log_probs_input,
                blank_id=blank_id,
                num_processes=10)
            
    def __call__(self, emission: torch.Tensor, seq_length = None) -> List[List[str]]:
        """
        :param probs: [n t c] if batch_first or [t n c]
        :param seq_length: [n] , defaults to None
        :return: batch of decoded tokens list[list[str]]
        """
        
        if not self.batch_first:
            emission = emission.permute(1, 0, 2) #force batch first
            
        if self.search_mode == 'beam':
            return self._beam_search_decode(emission, seq_length)
        else:
            return self._greedy_search(emission, seq_length)
        
    def _beam_search_decode(self, emission: torch.Tensor, seq_length=None):
        emission = emission.cpu()
        if seq_length is not None:
            seq_length = seq_length.cpu()
        
        ret = []
        beam_result, beam_scores, timesteps, out_seq_len = self.beam_decoder.decode(emission, seq_length)
        for batch_id in range(len(emission)):
            top_result = beam_result[batch_id][0][:out_seq_len[batch_id][0]] # [sequence_length]
            top_result = self.vocab.lookup_tokens(top_result.tolist())
            ret.append(top_result)
        return ret
    
    def _greedy_search(self, emission, seq_length=None):
        ret = []
        for batch_id in range(len(emission)):
            
            indices = torch.argmax(emission[batch_id], dim=-1)  # [num_seq, c]
            if seq_length is not None:
                indices = indices[:seq_length[batch_id]]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank_id]
            ret.append(self.vocab.lookup_tokens(indices))
        return ret
        
        
