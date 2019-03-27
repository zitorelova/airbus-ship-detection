from inclusion import * 

arch = resnet34

class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform=None):
        super(pdFilesDataset, self).__init__(fnames, path, transform)
        self.seg_df = pd.read_csv(SEGMENT).set_index('ImageId')

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img
        else: return cv2.resize(img, (self.sz, self.sz))

    def get_y(self, i):
        if(self.path == TEST): return 0

    def get_c(self): return 2

def get_data(sz, bs):
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.NO),
            RandomDihedral(tfm_y=TfmType.NO),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
            aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN),
            (val_n,TRAIN), tfms, test=(test_ids,TEST))
    md = ImageData(PATH, ds, bs, num_workers=num_workers, classes=None)
    return md
