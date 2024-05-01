from typing import Any
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F
import torchtext as tt
from typing import *
from fast_ctc_decode import beam_search

class CTCDecoder():
    '''
    Blank id is always zero
    '''
    
    def __init__(
        self, 
        vocab: tt.vocab.Vocab, 
        search_mode: Union[Literal['beam'], Literal['greedy']]='beam', 
        batch_first=False,
        beam_width=10
        ) -> None:
        self.vocab = vocab
        self.num_class = len(vocab)
        self.batch_first = batch_first
        self.search_mode = search_mode
        self.beam_width = beam_width

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
        # emmision [b t c]
        B, _, _ = emission.shape       
        emission = emission.cpu().numpy()
        if seq_length is not None:
            seq_length = seq_length.cpu().numpy()

        ret = []
        for b in range(B):
            v = self.vocab.get_itos()
            v = [f'){s}(' for s in v]
            
            if seq_length is not None:
                prob = emission[b][:seq_length[b]]
            else:
                prob = emission[b]
            
            output, _ = beam_search(prob, beam_size=self.beam_width, beam_cut_threshold=0., alphabet=v)
            ret.append(self.search_tokens(output))
        return ret
    
    @staticmethod
    def search_tokens(s: str):
        splited = s.split(')(')
        assert splited[0][0] == '('
        assert splited[-1][-1] == ')'
        
        splited[0] = splited[0][1:]
        splited[-1] = splited[-1][:-1]
        
        ret = [g[::-1] for g in splited]
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
        
        
