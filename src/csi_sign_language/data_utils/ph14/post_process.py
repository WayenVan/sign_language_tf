from typing import List, Tuple
from itertools import groupby
import re
import os
import numpy as np
from ..interface_post_process import IPostProcess

class PostProcess(IPostProcess):
    
    def __init__(self) -> None:
        super().__init__()
        
    def process(self, hyp: List[List[str]], gt: List[List[str]]) -> Tuple:
        hyp = post_process(hyp)
        return hyp, gt

def post_process(output: List[List[str]], regex=True, merge=True):
    if regex:
        output = [apply_regex(item) for item in output] 
    if merge:
        output = [merge_same(item) for item in output]
    return output

def apply_regex(output: List[str]):
    """After investigation the shell file, we find that many of these scripts are useless, thus we comment them all
    """
    output_s = ' '.join(output)

    output_s = re.sub(r'loc-', r'', output_s)
    output_s = re.sub(r'cl-', r'', output_s)
    output_s = re.sub(r'qu-', r'', output_s)
    output_s = re.sub(r'poss-', r'', output_s)
    output_s = re.sub(r'lh-', r'', output_s)
    output_s = re.sub(r'S0NNE', r'SONNE', output_s)
    output_s = re.sub(r'HABEN2', r'HABEN', output_s)

    output_s = re.sub(r'__EMOTION__', r'', output_s)
    output_s = re.sub(r'__PU__', r'', output_s)
    output_s = re.sub(r'__LEFTHAND__', r'', output_s)
    
    output_s = re.sub(r'WIE AUSSEHEN', r'WIE-AUSSEHEN', output_s)
    output_s = re.sub(r'ZEIGEN ', r'ZEIGEN-BILDSCHIRM ', output_s)
    output_s = re.sub(r'ZEIGEN$', r'ZEIGEN-BILDSCHIRM', output_s)

    output_s = re.sub(r'^([A-Z]) ([A-Z][+ ])', r'\1+\2', output_s)
    output_s = re.sub(r'[ +]([A-Z]) ([A-Z]) ', r' \1+\2 ', output_s)
    output_s = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +]SCH) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +]NN) ([A-Z][ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +][A-Z]) (NN[ +])', r'\1+\2', output_s)
    output_s = re.sub(r'([ +][A-Z]) ([A-Z]$)', r'\1+\2', output_s)
    output_s = re.sub(r'([A-Z][A-Z])RAUM', r'\1', output_s)
    output_s = re.sub(r'-PLUSPLUS', r'', output_s)
    
    # output_s = re.sub(r'(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])', r'\1', output_s)
    # output_s = re.sub(r'(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])', r'\1', output_s)
    # output_s = re.sub(r'(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])', r'\1', output_s)
    # output_s = re.sub(r'(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])', r'\1', output_s)

    output_s = re.sub(r'__EMOTION__', r'', output_s)
    output_s = re.sub(r'__PU__', r'', output_s)
    output_s = re.sub(r'__LEFTHAND__', r'', output_s)
    output_s = re.sub(r'__EPENTHESIS__', r'', output_s)
    
    return output_s.split()

def merge_same(output: List[str]):
    return [x[0] for x in groupby(output)]


if __name__ == "__main__":
    print(apply_regex(['S', 'S+H', 'C', 'WRW']))
    print('a      b  c '.split())

