import json
import numpy as np
import re
from PIL import Image
import torch

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

with open("data/annotations/captions_val2014.json", 'r') as f:
    coco = json.load(f)

anns = coco['annotations']
images = coco['images']
image_dic = {}
for ele in images:
    image_dic.update({ele['id']: ele['file_name']})

print len(anns)
#print anns[0]
for i in range(1):
    idx = np.random.randint(len(anns))
    idx = 75101
    caption = anns[idx]['caption']
    image_id = anns[idx]['image_id']

    print 'image_id'
    print re.findall(r"[\w']+", str(caption).lower())
    print image_dic[image_id]
    print "./data/val2014/" + image_dic[image_id]

    image_id = str(anns[idx]['image_id'])
    print image_id
    if len(image_id) < 6:
        for i in range(6-len(image_id)):
            image_id = "0" + image_id
    fname = "../Coco data/val2014/COCO_val2014_000000" + image_id + ".jpg"
    print fname
    img = Image.open(fname)
img.show()
#img = transform(img)

#pilTrans = transforms.ToPILImage()
#pilImg = pilTrans(img)
#pilImg.show()
