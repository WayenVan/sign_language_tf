import sys
from logging import Logger
import gc
import torch
import torch

def clean():
    gc.collect()
    torch.cuda.empty_cache()

def info(l: Logger, m):
    if l is not None:
        l.info(m)

def warn(l: Logger, m):
    if l is not None:
        l.warn(m)

def add_attributes(obj, locals: dict):
    for key, value in locals.items():
        if key != 'self' and key != '__class__': 
            setattr(obj, key, value)


def is_debugging():
    # Check if the script is executed with the -d or --debug option
    if "-d" in sys.argv or "--debug" in sys.argv:
        return True

    # Check if the script is executed with the -O or -OO option
    if sys.flags.optimize:
        return False

    # Check if the script is executed with the -X or --pdb option
    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
        return True

    # Check if the Python debugger module is imported
    if "pdb" in sys.modules:
        return True

    return False