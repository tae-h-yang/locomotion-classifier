import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LocomotionDataset(Dataset):
    def __init__(
        self,
        data_dir,
        input_mode="both",
        depth_transform=None,
        rgb_transform=None,
        seq_len=1,  # default to 1 for non-LSTM
    ):
        self.data_dir = data_dir
        self.input_mode = input_mode
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.seq_len = seq_len

        # Group filenames by label
        self.rgb_files = []
        self.depth_files = []
        self.labels = []

        for fname in sorted(os.listdir(data_dir)):
            if fname.startswith("rgb") and input_mode in ["rgb", "both"]:
                self.rgb_files.append(fname)
                self.labels.append(fname.split("_")[-1].split(".")[0])
            elif fname.startswith("depth") and input_mode in ["depth", "both"]:
                self.depth_files.append(fname)
                self.labels.append(fname.split("_")[-1].split(".")[0])

        self.label_map = {"walk": 0, "crawl": 1, "climb": 2}

        # Sequence slicing
        self.samples = []
        n = len(self.rgb_files) if input_mode != "depth" else len(self.depth_files)
        for i in range(n - seq_len + 1):
            rgb_seq = (
                self.rgb_files[i : i + seq_len]
                if input_mode in ["rgb", "both"]
                else None
            )
            depth_seq = (
                self.depth_files[i : i + seq_len]
                if input_mode in ["depth", "both"]
                else None
            )
            label = self.labels[i + seq_len - 1]
            self.samples.append((rgb_seq, depth_seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_seq, depth_seq, label = self.samples[idx]
        sample = {}

        if self.input_mode in ["rgb", "both"] and rgb_seq:
            rgb_imgs = [
                Image.open(os.path.join(self.data_dir, f)).convert("RGB")
                for f in rgb_seq
            ]
            rgb_imgs = [self.rgb_transform(img) for img in rgb_imgs]
            sample["rgb"] = torch.stack(rgb_imgs) if self.seq_len > 1 else rgb_imgs[0]

        if self.input_mode in ["depth", "both"] and depth_seq:
            depth_imgs = [
                Image.open(os.path.join(self.data_dir, f)).convert("L")
                for f in depth_seq
            ]
            depth_imgs = [self.depth_transform(img) for img in depth_imgs]
            sample["depth"] = (
                torch.stack(depth_imgs) if self.seq_len > 1 else depth_imgs[0]
            )

        sample["label"] = torch.tensor(self.label_map[label], dtype=torch.long)
        return sample
