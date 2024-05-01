import re
from itertools import groupby
from typing import List, Tuple
from ..interface_post_process import IPostProcess

class PostProcess(IPostProcess):
    
    def __init__(self) -> None:
        super().__init__()
        
    def process(self, hyp: List[List[str]], gt: List[List[str]]) -> Tuple:
        hyp = apply_hypothesis(hyp)
        gt = apply_groundtruth(gt)
        return hyp, gt

def merge_same(output: List[str]):
    return [x[0] for x in groupby(output)]

def apply_hypothesis(hyp):
    # remove repetitions
    hyp = [item for item, _ in groupby(hyp)]

    # remove __LEFTHAND__ and __EPENTHESIS__ and __EMOTION__ from ctm
    hyp = [item for item in hyp if not item == '__LEFTHAND__']
    hyp = [item for item in hyp if not item == '__EPENTHESIS__ ']
    hyp = [item for item in hyp if not item == '__EMOTION__']
    # remove all words starting and ending with "__", 
    hyp = [re.sub(r'^__', '', item) for item in hyp]
    hyp = [re.sub(r'__$', '', item) for item in hyp]

    # remove all -PLUSPLUS suffixes
    hyp = [re.sub(r'-PLUSPLUS$', '', item) for item in hyp]

    # remove all cl- prefix
    hyp = [re.sub(r'^cl-', '', item) for item in hyp]
    # remove all loc- prefix
    hyp = [re.sub(r'^loc-', '', item) for item in hyp]
    # remove RAUM at the end (eg. NORDRAUM -> NORD)
    hyp = [re.sub(r'RAUM$', '', item) for item in hyp]

    #remove ''
    hyp = [item for item in hyp if not item == '']

    # remove repetitions
    hyp = [item for item, _ in groupby(hyp)]
    return hyp

def apply_groundtruth(gt):
    # remove repetitions
    gt = [item for item, _ in groupby(gt)]

    # remove __LEFTHAND__ and __EPENTHESIS__ and __EMOTION__
    gt = [item for item in gt if not item == '__LEFTHAND__']
    gt = [item for item in gt if not item == '__EPENTHESIS__ ']
    gt = [item for item in gt if not item == '__EMOTION__']

    # remove all words starting and ending with "__", 
    gt = [re.sub(r'^__', '', item) for item in gt]
    gt = [re.sub(r'__$', '', item) for item in gt]

    # remove all -PLUSPLUS suffixes
    gt = [re.sub(r'-PLUSPLUS$', '', item) for item in gt]

    # remove all cl- prefix
    gt = [re.sub(r'^cl-', '', item) for item in gt]
    # remove all loc- prefix
    gt = [re.sub(r'^loc-', '', item) for item in gt]
    # remove RAUM at the end (eg. NORDRAUM -> NORD)
    gt = [re.sub(r'RAUM$', '', item) for item in gt]

    #remove ''
    gt = [item for item in gt if not item == '']

    # join WIE AUSSEHEN to WIE-AUSSEHEN
    gt = detect_wie(gt)

    # add spelling letters to compounds (A S -> A+S)
    gt = letters_compounds(gt)

    # remove repetitions
    # remove repetitions
    gt = [item for item, _ in groupby(gt)]
    return gt
    
def letters_compounds(input):
    state = 0
    output = []
    for item in input:
        if re.match(r'^[A-Z]$', item):
            if state == 0:
                output.append(item)
                state = 1
                continue
            elif state == 1:
                output[-1] = f'{output[-1]}+{item}'
                continue
            else:
                NotImplementedError()
        else:
            output.append(item)
            state = 0
            continue
    return output

def detect_wie(input):
    state = 0
    output = []
    for item in input:
        if item == 'WIE':
            output.append(item)
            state = 1
            continue
        if item == 'AUSSEHEN':
            if state == 1:
                output[-1] = f'{output[-1]}-{item}'
                state = 0
                continue
            elif state == 0:
                output.append(item)
                continue
        else:
            output.append(item)
            continue
    return output
                

if __name__ == '__main__':
    print(apply_groundtruth(['A', 'B', 'B', 'WIE', 'AUSSEHEN', 'C', 'D', 'E', 'loc-HELLO', '__HAHA', '__LEFTHAND__']))
    print(apply_hypothesis(['A', 'B', 'B', 'WIE', 'AUSSEHEN', 'C', 'D', 'E', 'loc-HELLO', '__HAHA', '__LEFTHAND__']))