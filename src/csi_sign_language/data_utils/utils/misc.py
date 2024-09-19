from typing import List
import os
import numpy as np


def add_exe_mode(file_path):
    os.chmod(file_path, os.stat(file_path).st_mode | 0o111)


def glosses2ctm(ids: List[str], glosses: List[List[str]], path: str):
    with open(path, "w") as f:
        for id, gloss in list(zip(ids, glosses)):
            start_time = 0
            for single_gloss in gloss:
                tl = np.random.random() * 0.1
                f.write(
                    "{} 1 {:.3f} {:.3f} {}\n".format(
                        id, start_time, start_time + tl, single_gloss
                    )
                )
                start_time += tl
