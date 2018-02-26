from PIL import Image
import os
import json, re
import torch
import torch.utils.data as data
import torch.nn.functional as F
#from pycocotools.coco import COCO
from vocab import Vocab
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                     std = [ 0.229, 0.224, 0.225 ]),
                                ])
root = "../Coco data/val2014"
cocofile = "../Coco data/annotations/captions_val2014.json"

def image_loader(img_name, transform=None):
    """load image, returns tensor"""
    image = Image.open(img_name).convert('RGB')
    image = transform(image).float()
    image = torch.Tensor(image)
    #image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


class CocoData(data.Dataset):
    def __init__(self, root, cocofile, vocab, transform=None):
        with open(cocofile, 'r') as f:
            coco = json.load(f)
        self.anns = coco['annotations']
        self.vocab = vocab
        self.transform = transform
        self.images = coco['images']
        self.image_dic = {}
        for ele in self.images:
            self.image_dic.update({ele['id']: ele['file_name']})
        self.root = root

    def __getitem__(self, idx):
        caption = self.anns[idx]['caption']
        tokens = re.findall(r"[\w']+", str(caption).lower())
        image_id = self.anns[idx]['image_id']
        filename = os.path.join(self.root, self.image_dic[image_id])

        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        caption = torch.LongTensor(caption)
        #caption = Variable(caption, requires_grad=False)
        caption = caption.unsqueeze(0)
        image = image_loader(filename, transform)
        return image, caption

    def __len__(self):
        return len(self.anns)


def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[1].size(1), reverse=True)
    images, captions = zip(*data)
    N = len(images) # batch_size
    images = torch.cat(images, 0) # image loader already adds a batch dimension

    max_len = captions[0].size(1)
    padded_caps = [F.pad(cap, (0, max_len - cap.size(1))) for cap in captions]
    caption_out = torch.cat(padded_caps, 0)
    caption_in = caption_out[:, :-1]
    return images, caption_in, caption_out


def get_loader(root, cocofile, vocab, transform=None, batch_size=64, shuffle=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoData(root=root, cocofile=cocofile, vocab=vocab, transform=transform) # look above for definition

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader
