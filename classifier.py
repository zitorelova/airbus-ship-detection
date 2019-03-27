from inclusion import * 
from dataset import * 


seg_df = pd.read_csv(SEGMENT)

arch = resnet34

train_ids = [f for f in seg_df['ImageId']]
test_ids = [f for f in os.listdir(TEST)]
tr_n, val_n = train_test_split(train_ids, test_size=0.2, random_state=random_state)

sz = 256 
bs = 128 
lr = 3e-3

def train_classifier():
    """
    Function for training ship classifier
    """

    start = time.time()

    md = get_data(sz, bs)
    learn = ConvLearner.pretrained(arch, md, ps=0.5)
    learn.opt_fn = optim.Adam

    learn.fit(lr, 1)
    learn.save('./models/resnet34_256_p1')

    learn.unfreeze()
    lr = np.array([1e-4, 5e-4, 3e-3]) # differential learning rates

    learn.fit(lr, 1, cycle_len=2, cycle_mult=2)
    learn.save('./models/resnet34_256_p2')

    md = get_data(384, 128) # increasing image size
    learn.set_data(md)
    learn.fit(lr/2, 1, cycle_len=2, use_clr=(20, 8))
    learn.save('./models/resnet34_384_p1')

    md = get_data(512, 32) # increasing image size
    learn.set_data(md)
    learn.fit(lr/3, 1 cycle_len=2, use_clr=(15, 8))
    learn.save('./models/resnet34_512_p1')

    log_preds, y = learn.predict_with_targs(is_test=True)
    probs = np.exp(log_preds)[:,1]
    preds = (probs > 0.5).astype(int)

    df = pd.DataFrame({'id': test_ids, 'p_ship': probs})
    df.to_csv('ship_probs_2.csv', header=True, index=False)

    print(f'Training finished in {(time.time() - start) / 60 :.3} minutes.')

if __name__ == "__main__":
    train_classifier()
