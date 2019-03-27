from inclusion import * 
from dataset import *
from utils import *
from losses import *

arch = resnet34

train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=random_state)

PRETRAINED = './models/resnet34_512_p1'

def train_segmentation():
    """
    Function for training the image segmentation model
    """

    start = time.time()

    model_base = load_pretrained(get_base(), PRETRAINED)
    cut, lr = model_meta[arch]
    m = to_gpu(Unet34(model_base))
    model = UnetModel(m)

    sz = 256
    bs = 64

    md = get_data(sz, bs)

    learn = ConvLearner(md, model)
    learn.opt_fn = optim.Adam()
    learn.crit = LossBinary(jaccard_weight=5)
    learn.metrics = [accuracy_thresh(0.5), dice, IoU]
    wd = 1e-7
    lr = 1e-2

    learn.freeze_to(1)
    learn.fit(lr, 1, wds=wd, cycle_len=1, use_clr=(5,8))
    learn.unfreeze() # unfreeze encoder
    learn.bn_freeze(True)

    lrs = np.array([lr/100, lr/10, lr])
    learn.fit(lrs/3, 2, wds=wd, cycle_len=2, use_clr=(20,8))

    learn.save('./models/weighted_unet_256_p1')

    sz = 384
    bs = 32

    md = get_data(sz, bs)
    learn.set_data(md)
    learn.unfreeze()
    learn.bn_freeze(True)

    learn.fit(lrs/5, 1, wds=wd, cycle_len=2, use_clr(10,8)) # first increase in image size with decreased bs
    learn.save('./models/weighted_unet_384_p1')

    sz = 512
    bs = 16

    md = get_data(sz, bs)
    learn.set_data(md)
    learn.unfreeze()
    learn.bn_freeze(True)

    learn.fit(lrs/10, 2, wds=wd, cycle_len=1, use_clr=(10,8), best_save_name='./models/weighted_unet_512_p1') # second increase in image size with further decreased bs

    sz = 768
    bs = 8

    md = get_data(sz, bs)
    learn.set_data(md)
    learn.unfreeze()
    learn.bn_freeze(True)

    learn.fit(lrs/50, 10, wds=5e-8, cycle_len=1, use_clr=(10,10), best_save_name='./models/weighted_unet_768_p1') # full image size with further decreased bs

    learn.crit = MixedLoss(10., 2.)
    learn.fit(lrs/50, 10, wds=5e-8, cycle_len=1, use_clr=(10,10), best_save_name='./models/weighted_unet_768_p2') # full image size with further decreased bs (final run)

    learn.save('./models/weighted_unet_768_final')

    print(f'Training finished in {time.time() - start) / 60 :.3} minutes.')

if __name__ == "__main__":
    main()
