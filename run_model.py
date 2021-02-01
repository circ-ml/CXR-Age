"""Main testing script for the composite outcome experiment. Purpose is to determine whether using composite outcomes improves DL performance for prognosis

Usage:
  run_model.py <image_dir> <model_path> <output_file> [--checkFiles] [--modelarch=MODELARCH] [--type=TYPE] [--dataframe=DF] [--target=TARGET] [--split=SPLIT] [--size=SIZE]
  run_model.py (-h | --help)
Examples:
  run_model.py /path/to/images /path/to/model /path/to/write/output.csv
Options:
  -h --help                    Show this screen.
  --modelarch=MODELARCH        CNN model architecture to train [default: Resnet34]
  --type=TYPE                  Type of output [default: Discrete]
  --dataframe=DF               Optional data frame to select which images are of interest [default: None]
  --target=TARGET              If optional df is specified, then need to include the target variable [default: None]
  --split=SPLIT                If split, then split on the Dataset column keeping only the Te values [default: False]
  --checkFiles                 Should we check whether df files actually exist?
  --size=SIZE                  Resize to this size [Default:224]
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from docopt import docopt
import pandas as pd
import fastai
from fastai.vision import *
import pretrainedmodels
from sklearn.metrics import *
from fastai.callbacks import *
import math
import time
import SimpleArchs

###TODO Add optional checkpointing (optional result file to append to, skipping loop iteration if model exists)
tfms_test = get_transforms(do_flip = False,max_warp = None)



    
def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, scale:float=1.35) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.dl(ds_type)
    ds = dl.dataset
    old = ds.tfms
    #activ = ifnone(activ, _loss_func2activ(learn.loss_func))
    augm_tfm = [o for o in learn.data.train_ds.tfms if o.tfm not in
               (crop_pad, flip_lr, dihedral, zoom)]
    try:
        pbar = master_bar(range(8))
        for i in pbar:
            row = 1 if i&1 else 0
            col = 1 if i&2 else 0
            #flip = i&4
            d = {'row_pct':row, 'col_pct':col, 'is_random':False}
            tfm = [*augm_tfm, zoom(scale=scale, **d), crop_pad(**d)]
            #if flip: tfm.append(flip_lr(p=1.))
            #import pdb; pdb.set_trace()
            ds.tfms = tfm
            yield get_preds(learn.model, dl, pbar=pbar, activ=activ)[0]
    finally: ds.tfms = old


def _TTA(learn:Learner, beta:float=0.4, scale:float=1.35, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, with_loss:bool=False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds,y = learn.get_preds(ds_type, activ=activ)
    all_preds = list(_tta_only(learn,ds_type=ds_type, activ=activ, scale=scale))
    avg_preds = torch.stack(all_preds).mean(0)
    sd_preds = torch.stack(all_preds).std(0)
    if beta is None: return preds,avg_preds,y,sd_preds
    else:
        final_preds = preds*beta + avg_preds*(1-beta)
        if with_loss:
            with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)
            return final_preds, y, loss,sd_preds
        return final_preds, y,sd_preds

num_workers = 16
bs = 32
if __name__ == '__main__':

    arguments = docopt(__doc__)
  
    ###Grab image directory
    image_dir = arguments['<image_dir>']
    
    mdl_path = arguments['<model_path>']
    size = int(arguments['--size'])

    ###set model architecture
    m = arguments['--modelarch'].lower()
    if(arguments['--dataframe']=="None"):
        files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir,f))] 
        ###Results
        output_df = pd.DataFrame(columns = ['File','Dummy','Prediction'])
        
        output_df['File'] = files
        if(arguments['--type'].lower()=="discrete"):
            output_df['Dummy'] = np.random.randint(0,2,len(files))
        else:
            output_df['Dummy'] = np.random.random_sample(len(files))
        col = 'Dummy'
    else:
        output_df = pd.read_csv(arguments['--dataframe'])
        locs = []
        if(arguments['--checkFiles']):
            for i in range(0,output_df.shape[0]):
                if(os.path.exists(os.path.join(image_dir,output_df.iloc[i,0]))):
                    locs.append(i)
                else:
                    print(output_df.iloc[i,0])
            output_df = output_df.iloc[locs,:]
        
            output_df = output_df.reset_index(drop=True)  
        col = arguments['--target']
        
        if(arguments["--split"]!="False"):
            output_df = output_df[output_df.Dataset=="Te",]
    if(arguments["--type"].lower()=="continuous"):
        imgs = (ImageList.from_df(df=output_df,path=image_dir)
                                .split_none()
                                .label_from_df(cols=col,label_cls=FloatList)
                                .transform(tfms_test,size=size)
                                .databunch(num_workers = num_workers,bs=bs).normalize(imagenet_stats))
    else:
        imgs = (ImageList.from_df(df=output_df,path=image_dir)
                                .split_none()
                                .label_from_df(cols=col)
                                .transform(tfms_test,size=size)
                                .databunch(num_workers = num_workers,bs=bs).normalize(imagenet_stats))
                                
                                
    manual = False
    
    #Compute # of output nodes
    if(arguments['--type'].lower()=="continuous"):
        out_nodes = 1
    else:
        out_nodes = 2
    
    
    if(m=="inceptionv4"):
        def get_model(pretrained=True, model_name = 'inceptionv4', **kwargs ): 
            if pretrained:
                arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            else:
                arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            return arch

        def get_cadene_model(pretrained=True, **kwargs ): 
            return fastai_inceptionv4
        custom_head = create_head(nf=2048*2, nc=37, ps=0.75, bn_final=False) 
        fastai_inceptionv4 = nn.Sequential(*list(children(get_model(model_name = 'inceptionv4'))[:-2]),custom_head) 

    
    ###Based on the input model, create a cnn learner object
    
    elif(m=="resnet50"):
        mdl = fastai.vision.models.resnet50
    elif(m=="resnet34"):
        mdl = fastai.vision.models.resnet34
    elif(m=="resnet16"):
        mdl = fastai.vision.models.resnet16
    elif(m=="resnet101"):
        mdl = fastai.vision.models.resnet101
    elif(m=="resnet152"):
        mdl = fastai.vision.models.resnet152
    elif(m=="densenet121"):
        mdl = fastai.vision.models.densenet121
    elif(m=="densenet169"):
        mdl = fastai.vision.models.densenet169
    elif(m=="age"):
        mdl=fastai.vision.models.resnet34
    elif(m=="larget"):
        manual = True
        mdl = SimpleArchs.get_simple_model("LargeT",out_nodes)
    elif(m=="largew"):
        manual = True
        mdl = SimpleArchs.get_simple_model("LargeW",out_nodes)
    elif(m=="small"):
        manual = True
        mdl = SimpleArchs.get_simple_model("Small",out_nodes)
    elif(m=="tiny"):
        manual = True
        mdl = SimpleArchs.get_simple_model("Tiny",out_nodes)
    elif(m=="age"):
        mdl = fastai.vision.models.resnet34
    else:
        print("Sorry, model: " + m + " is not yet supported... coming soon!")
        quit()
    
    
    
    if(m=='inceptionv4'):
        learn = cnn_learner(imgs, get_cadene_model, metrics=accuracy)
    elif(manual):
        learn = Learner(imgs,mdl)
    else:
        learn = cnn_learner(imgs, mdl, metrics=accuracy)
        
    if(m=="age"):
        numFeatures = 16
        learn.model[1] = nn.Sequential(*learn.model[1][:-5],nn.Linear(1024,512,bias=True),nn.ReLU(inplace=True),nn.BatchNorm1d(512),nn.Dropout(p=0.5),
                             nn.Linear(512,numFeatures,bias=True),nn.ReLU(inplace=True),nn.BatchNorm1d(numFeatures),
                             nn.Linear(numFeatures,1,bias=True)).cuda()
                             
                             
    learn.load(mdl_path)
    if(arguments['--type'].lower()=="discrete"):
        preds,y,sd_preds = _TTA(learn,ds_type = DatasetType.Fix,activ=nn.Softmax())
    
    ###output predictions as column with model name
        output_df['Prediction'] = np.array(preds[:,1])
        output_df['SD_Prediction'] = np.array(sd_preds[:,1])
    else:
        preds,y,sd_preds = _TTA(learn,ds_type = DatasetType.Fix)
    
    ###output predictions as column with model name
        output_df['Prediction'] = np.array(preds)
        output_df['SD_Prediction'] = np.array(sd_preds)


    if(m=="age"):
        import pdb; pdb.set_trace()
        arr = np.array(output_df.Prediction)
        arr = arr * 8.03342449139388 + 63.8723890235948
        arr = arr * 6.75523 - 0.03771*arr*arr -213.77257 
        output_df['CXR_Age'] = arr
        output_df = output_df.drop(["Prediction"],axis=1)
        output_df = output_df.drop(["Dummy"],axis=1)
    output_df.to_csv(arguments['<output_file>'])
