import math
import torch
import numpy as np
from einops import rearrange
import warnings


class ApplyByKey:
    def __init__(self, key, transforms: list) -> None:
        self.key = key
        self.transforms = transforms

    def __call__(self, data, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        d = data[self.key]
        for t in self.transforms:
            d = t(d)
        data[self.key] = d
        return data


class Rearrange:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    def __call__(self, data) -> torch.Any:
        return rearrange(data, self.pattern)


class ToTensor:
    def __init__(self, dtype="default") -> None:
        self.dtype = dtype
        self.str2dtype = {
            "float32": torch.float32,
            "float64": torch.double,
            "default": None,
        }

    def __call__(self, data):
        data = torch.tensor(data, dtype=self.str2dtype[self.dtype])
        return data


class Rescale:
    def __init__(self, input, output) -> None:
        self.input = input
        self.output = output

    def __call__(self, video):
        video = self.output[0] + (self.output[1] - self.output[0]) * (
            video - self.input[0]
        ) / (self.input[1] - self.input[0])

        assert video.max() <= self.output[1] + 0.1, f"{video.max()}"
        assert video.min() >= self.output[0], f"{video.min()}"
        return video


class CentralCrop:
    def __init__(self, size=224) -> None:
        self.size = size

    def __call__(self, video):
        T, C, H, W = video.shape
        start_h = math.floor((H - self.size) / 2.0)
        start_w = math.floor((W - self.size) / 2.0)
        video = video[
            :, :, start_h : start_h + self.size, start_w : start_w + self.size
        ]
        return video


class TemporalAug:
    def __init__(self, t_min, t_max, min_len_scale=8, n_frame_max=400) -> None:
        """
        @param min_len_scale: result_length should >= min_len_scale * num_tokens
        """

        self.t_min = t_min
        self.t_max = t_max
        self.n_frame_max = n_frame_max
        self.min_len_scale = min_len_scale

    def __call__(self, data):
        video = data["video"]
        num_tokens = len(data["gloss"])
        vlen = len(video)

        # the
        indexes = self.temporal_augmentation(
            vlen,
            self.t_min,
            self.t_max,
            num_tokens * self.min_len_scale,
            self.n_frame_max,
        )
        ret = [video[i] for i in indexes]
        if isinstance(video, np.ndarray):
            ret = np.stack(ret)
        elif isinstance(video, torch.Tensor):
            ret = torch.stack(ret)
        else:
            raise NotImplementedError()

        data["video"] = ret
        return data

    @staticmethod
    def temporal_augmentation(vlen, t_min, t_max, min_frames, max_frames):
        if max_frames < min_frames:
            raise ValueError(
                f"max_frames: {max_frames} must be less than or equal to min_frames: {min_frames}"
            )

        if t_min == 1.0 and t_max == 1.0:
            if vlen < min_frames or vlen > max_frames:
                raise ValueError(
                    f"Video length: {vlen} must be in the range [{min_frames}, {max_frames}]"
                )
            return tuple(range(vlen))

        min_len = max(min_frames, int(vlen * t_min))
        max_len = min(max_frames, int(vlen * t_max))

        if min_len > max_len:
            # raise ValueError(
            #     f"min_len: {min_len} must be less than or equal to max_len: {max_len}"
            # )
            warnings.warn(
                f"min_len > max_len, temporally adjust t_max scale to {min_len/vlen}"
            )
            max_len = min_len

        select_len = np.random.randint(min_len, max_len + 1)

        if select_len == vlen:
            return tuple(range(vlen))
        elif select_len < vlen:
            selected_index = sorted(np.random.choice(vlen, select_len, replace=False))
        elif select_len > vlen:
            added_index = np.random.choice(vlen, select_len - vlen, replace=True)
            selected_index = sorted(np.concatenate((np.arange(vlen), added_index)))

        return selected_index

    # @staticmethod
    # def get_scaled_frame_index(
    #     vlen, tmin=1, tmax=1, min_num_frames=0, max_num_frames=400
    # ):
    #     assert max_num_frames % 4 == 0
    #     if vlen <= min_num_frames:
    #         warnings.warn(
    #             f"Video length:{vlen} is less than min_num_frames: {min_num_frames}"
    #         )
    #
    #     if tmin == 1 and tmax == 1:
    #         if vlen <= max_num_frames:
    #             frame_index = np.arange(vlen)
    #             valid_len = vlen
    #         else:
    #             sequence = np.arange(vlen)
    #             an = (vlen - max_num_frames) // 2
    #             en = vlen - max_num_frames - an
    #             frame_index = sequence[an:-en]
    #             valid_len = max_num_frames
    #
    #         if (valid_len % 4) != 0:
    #             valid_len -= valid_len % 4
    #             frame_index = frame_index[:valid_len]
    #
    #         assert len(frame_index) == valid_len, (frame_index, valid_len)
    #         return frame_index, valid_len
    #
    #     min_len = max(int(tmin * vlen), min_num_frames)
    #     max_len = min(max_num_frames, int(tmax * vlen))
    #     selected_len = np.random.randint(min_len, max_len + 1)
    #
    #     if (selected_len % 4) != 0:
    #         selected_len += 4 - (selected_len % 4)
    #
    #     if selected_len <= vlen:
    #         selected_index = sorted(
    #             np.random.permutation(np.arange(vlen))[:selected_len]
    #         )
    #     else:
    #         copied_index = np.random.randint(0, vlen, selected_len - vlen)
    #         selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))
    #
    #     if selected_len <= max_num_frames:
    #         frame_index = selected_index
    #         valid_len = selected_len
    #     else:
    #         assert False, (vlen, selected_len, min_len, max_len)
    #
    #     assert len(frame_index) == valid_len, (frame_index, valid_len)
    #
    #     return frame_index, valid_len


if __name__ == "__main__":
    t = TemporalAug(0.5, 1.5, 8, 400)
    index = t.temporal_augmentation(10, 1.0, 1.5, 8, 400)
