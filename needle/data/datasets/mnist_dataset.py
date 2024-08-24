from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip, struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(image_filename) as f:
            _, num, ros, cols = struct.unpack('>4I', f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        with gzip.open(label_filename) as f:
            _, num = struct.unpack('>2I', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        self.images = (self.images - self.images.min()) / (self.images.max() - self.images.min())
        self.images = self.images.astype(np.float32)
        assert self.images.shape[0] == self.labels.shape[0] == num
        self.transforms = transforms
        
        

    def __getitem__(self, index) -> object:
        img = self.images[index]
        label = self.labels[index]
        if len(img.shape) < 2:
            img = img.reshape(28,28,1)
            
        if self.transforms is not None:
            for tform in self.transforms:
                img = tform(img)
            
        return img, label
    
    
    def __len__(self) -> int:
        return len(self.images)  