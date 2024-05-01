import torch
from fast_ctc_decode import beam_search
from paddlespeech_ctcdecoders import ctc_beam_search_decoding
import sys
sys.path.append('src')
import csi_sign_language.utils.ctc_decoder as ctc

def search_tokens(s: str):
    splited = s.split(')(')
    print(splited)
    assert splited[0][0] == '('
    assert splited[-1][-1] == ')'
    
    splited[0] = splited[0][1:]
    splited[-1] = splited[-1][:-1]
    
    ret = [g[::-1] for g in splited]
    print(ret)

torch.manual_seed(1)
posterios = torch.rand(10, 5)
poster = torch.nn.functional.softmax(posterios)

a = ['ab', 'bbb', 'ccc', 'd', 'e']
b = [f'){s}(' for s in a]
output1 ,_= beam_search(poster.numpy(), beam_size=50, beam_cut_threshold=0.01, alphabet=b)

print(output1)
search_tokens(output1)

ctc.CTCDecoder()

