import os
import shutil
from pathlib import Path

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.data_utils.utils.misc import add_exe_mode, glosses2ctm
    from collections import defaultdict
else:
    from ..utils.misc import glosses2ctm


class Pheonix14Evaluator:
    """
    Python evaluator for the Phoenix14T, the evaluator will copy the required resources file
    from  PH14T's evaluatioin folder into workspace, and then output the hypothesis as ctm file,
    finnaly run the shell command to do the evaluation.

    This produce evaluation which follows the rules provided by the official example.
    """

    def __init__(self, data_root) -> None:
        self.data_root = Path(data_root)
        self.eval_root = (
            self.data_root / "PHOENIX-2014-T" / "evaluation" / "sign-recognition"
        )

        self.merge_file = self.eval_root / "mergectmstm.py"
        self.eval_shell = self.eval_root / "evaluatePHOENIX-2014-T-signrecognition.sh"
        self.dev_stm = self.eval_root / "PHOENIX-2014-T-groundtruth-dev.stm"
        self.test_stm = self.eval_root / "PHOENIX-2014-T-groundtruth-test.stm"

    def eval(self, work_dir, ids, hyp, mode="dev"):
        if mode not in ("dev", "test"):
            raise NotImplementedError()
        work_dir = Path(work_dir)
        if not work_dir.exists():
            work_dir.mkdir(exist_ok=True)

        self.prepare_resources(work_dir)
        glosses2ctm(ids, hyp, str(work_dir / "hyp.ctm"))
        cmd1 = f"cd {work_dir};"
        cmd2 = f"sh ./evaluatePHOENIX-2014-T-signrecognition.sh hyp.ctm {mode}"
        if os.system(cmd1 + cmd2) != 0:
            raise Exception("sclit cmd runing failed")

        result_file = work_dir / "out.hyp.ctm.sys"

        with result_file.open("r") as fid:
            for line in fid:
                line = line.strip()
                if "Sum/Avg" in line:
                    result = line
                    break

        tmp_err = result.split("|")[3].split()
        subs, inse, dele, wer = tmp_err[1], tmp_err[3], tmp_err[2], tmp_err[4]
        subs, inse, dele, wer = float(subs), float(inse), float(dele), float(wer)
        errs = [wer, subs, inse, dele]
        return errs

    def prepare_resources(self, work_dir: Path):
        merge_file = work_dir / "mergectmstm.py"
        eval_shell = work_dir / "evaluatePHOENIX-2014-T-signrecognition.sh"
        dev_stm = work_dir / "PHOENIX-2014-T-groundtruth-dev.stm"
        test_stm = work_dir / "PHOENIX-2014-T-groundtruth-test.stm"
        shutil.copyfile(str(self.eval_shell), str(eval_shell))
        shutil.copyfile(str(self.merge_file), str(merge_file))
        shutil.copyfile(str(self.dev_stm), str(dev_stm))
        shutil.copyfile(str(self.test_stm), str(test_stm))
        merge_file.chmod(merge_file.stat().st_mode | 0o111)


if __name__ == "__main__":
    data_root = "dataset/PHOENIX-2014-T-release-v3"
    evaluator = Pheonix14Evaluator(data_root)
    work_dir = "outputs/test_eval"
    ctm_file = Path(
        "dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/evaluation/sign-recognition/out.example-hypothesis-dev.ctm"
    )

    result_dict = defaultdict(list)
    with ctm_file.open("r") as f:
        for line in f:
            id = line.split()[0]
            label = line.split()[-1]
            result_dict[id].append(label)

    # transfer to ids
    ids = []
    hyp = []
    for k, v in result_dict.items():
        ids.append(k)
        hyp.append(v)

    mode = "dev"
    evaluator.eval(work_dir, ids, hyp, mode)
