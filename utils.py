from inclusion import *

def cut_empty(names, seg_df):
    return [name for name in names if (type(seg_df.loc[name]['EncodedPixels']) != float)]

def get_mask(img_id, df):
    shape = (768, 768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if (type(masks) == float): return img.reshape(shape)
    if (type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    return img.reshape(shape).T

def load_pretrained(model, path):
    weights = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights, strict=False)
    return model

def get_base():
    layers = cut_model(arch(True), cut)
    return 
