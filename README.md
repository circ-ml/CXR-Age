# CXR-Age: Deep learning to estimate biological age from chest radiographs

![CXR-Age Grad-CAM](/images/GradCAM_Github_020121.png)

[Raghu VK*, Weiss, J., Hoffmann, U., Aerts HJWL, and Lu, MT. Deep learning to estimate biological age from chest radiographs. Journal of the American College of Cardiology: Cardiovascular Imaging 2021; Epub ahead of print](<https://authors.elsevier.com/a/1clbm,i2Xrn9-f>) *Equal contribution



## Overview
Age-related chronic disease causes 60% of deaths in the US. Primary prevention (e.g. statin to prevent cardiovascular disease) and screening (e.g. screening for lung cancer with chest CT) interventions are based on chronological age, but chronological age is an imperfect measure of aging. A measure of biological age that more accurately predicts longevity and disease would enable healthcare providers to better personalize care and help researchers address factors underlying the aging process.

Chest x-rays (radiographs or CXRs) are among the most common diagnostic imaging tests in medicine. We hypothesized that a convolutional neural network (CNN) could extract information from these x-rays to estimate a person's chest x-ray age (or CXR-Age) - a summary measure of overall health based on the chest x-ray image. We tested whether CXR-Age predicted life expectancy beyond chronological age.

CXR-Age outputs a number (in years) reflecting all-cause mortality risk based on only a single chest radiograph image. CXR-Age was developed in two steps. First, it was trained to predict chronological age in over 100,000 images from publicly available chest x-ray datasets (CheXpert, PadCHEST, and NIH CXR-14). Then, the model was fine-tuned to estimate a "biological age" in XX persons from the Prostate, Lung, Colorectal and Ovarian cancer screening trial (PLCO), a randomized controlled trial of chest x-ray to screen healthy persons for lung cancer. 

CXR-Age was tested (referred to as "validation" in the publication) in an independent cohort of 40,967 individuals from PLCO and externally tested in 5,493 heavy smokers from the National Lung Screening Trial (NLST). CXR-LC predicted long-term all-cause (CXR-Age HR 2.26 per 5 years; p < 0.001) and cardiovascular mortality (CXR-Age cause specific HR 2.45 per 5 years, p< 0.001) in the PLCO testing dataset. Similar results were found in NLST. Adding CXR-Age to a multivariable model of conventional cardiovascular risk factors resulted in significant improvements for predicting both outcomes in both datasets. 

**Central Illustration of CXR-Age**
![CXR-Age Central Illustration](/images/Central_Illustration.png)

This repo contains data intended to promote reproducible research. It is not for clinical care or commercial use. 

For those without programming expertise, please use our user-friendly webserver: https://cxrage.org/#home

## Installation
This inference code was tested on Ubuntu 18.04.3 LTS, conda version 4.8.0, python 3.7.7, fastai 1.0.61, cuda 10.2, pytorch 1.5.1 and cadene pretrained models 0.7.4. A full list of dependencies is listed in `environment.yml`. 

Inference can be run on the GPU or CPU, and should work with ~4GB of GPU or CPU RAM. For GPU inference, a CUDA 10 capable GPU is required.

For the model weights to download, Github's large file service must be downloaded and installed: https://git-lfs.github.com/ 

This example is best run in a conda environment:

```bash
git lfs clone https://github.com/vineet1992/CXR-Age/
cd location_of_repo
conda env create -n CXR_Age -f environment.yml
conda activate CXR_Age
python run_model.py dummy_datasets/test_images/ development/models/PLCO_Fine_Tuned_120419 output/output.csv --modelarch=age --type=continuous --size=224
```

To generate saliency maps for each estimate, add "--saliency=path/to/output/saliency/maps". Next is a complete example of this command

```bash
python run_model.py dummy_datasets/test_images/ development/models/PLCO_Fine_Tuned_120419 output/output.csv --modelarch=age --type=continuous --size=224 --saliency=saliency_maps
```
Dummy image files are provided in `dummy_datasets/test_images/;`. Weights for the CXR-Age model are in `development/models/PLCO_Fine_Tuned_120419.pth`. 

## Datasets
PLCO (NCT00047385) data used for model development and testing are available from the National Cancer Institute (NCI, https://biometry.nci.nih.gov/cdas/plco/). NLST (NCT01696968) testing data is available from the NCI (https://biometry.nci.nih.gov/cdas/nlst/) and the American College of Radiology Imaging Network (ACRIN, https://www.acrin.org/acrin-nlstbiorepository.aspx). Due to the terms of our data use agreement, we cannot distribute the original data. Please instead obtain the data directly from the NCI and ACRIN.

The `data` folder provides the image filenames and the CXR-Age estimates. "File" refers to image filenames and "CXR-Age" refers to the CXR-Age estimate: 
* `PLCO_Age_Estimates.csv` contains the CXR-Age estimates in the PLCO testing dataset.
* `NLST_Age_Estimates.csv` contains the CXR-Age estimate in the NLST testing dataset. The format for "File" is (original participant directory)_(original DCM filename).png


## Image processing
PLCO radiographs were provided as scanned TIF files by the NCI. TIFs were converted to PNGs with a minimum dimension of 512 pixels with ImageMagick v6.8.9-9. 

Many of the PLCO radiographs were rotated 90 or more degrees. To address this, we developed a CNN to identify rotated radiographs. First, we trained a CNN using the resnet34 architecture to identify synthetically rotated radiographs from the [CXR14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf). We then fine tuned this CNN using 11,000 manually reviewed PLCO radiographs. The rotated radiographs identified by this CNN in `preprocessing/plco_rotation_github.csv` were then corrected using ImageMagick. 

```bash
cd path_for_PLCO_tifs
mogrify -path destination_for_PLCO_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
cd path_for_PLCO_pngs
while IFS=, read -ra cols; do mogrify -rotate 90 "${cols[0]}"; done < /path_to_repo/preprocessing/plco_rotation_github.csv
```

NLST radiographs were provided as DCM files by ACRIN. We chose to first convert them to TIF using DCMTK v3.6.1, then to PNGs with a minimum dimension of 512 pixels through ImageMagick to maintain consistency with the PLCO radiographs:

```bash
cd path_to_NLST_dcm
for x in *.dcm; do dcmj2pnm -O +ot +G $x "${x%.dcm}".tif; done;
mogrify -path destination_for_NLST_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
```


The orientation of several NLST chest radiographs was manually corrected:

```
cd destination_for_NLST_pngs
mogrify -rotate "90" -flop 204025_CR_2000-01-01_135015_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030903.1123713.1.png
mogrify -rotate "-90" 208201_CR_2000-01-01_163352_CHEST_CHEST_n1__00000_2.16.840.1.113786.1.306662666.44.51.9597.png
mogrify -flip -flop 208704_CR_2000-01-01_133331_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030718.1122210.1.png
mogrify -rotate "-90" 215085_CR_2000-01-01_112945_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030605.1101942.1.png
```

## Acknowledgements
I thank the NCI and ACRIN for access to trial data, as well as the PLCO and NLST participants for their contribution to research. I would also like to thank the fastai and Pytorch communities as well as the National Academy of Medicine for their support of this work. A GPU used for this research was donated as an unrestricted gift through the Nvidia Corporation Academic Program. The statements contained herein are mine alone and do not represent or imply concurrence or endorsements by the above individuals or organizations.


