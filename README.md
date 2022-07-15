# Readme
This work is published as paper in DCC 2022.
## Dataset        

Training dataset includes 600K cropped 256*256 images generated from LIU4K dataset. 

Reference: J. Liu, D. Liu, W. Yang, S. Xia, X. Zhang and Y. Dai, "A Comprehensive Benchmark for Single Image Compression Artifact Reduction," in IEEE Transactions on Image Processing, vol. 29, pp. 7845-7860, 2020, doi: 10.1109/TIP.2020.3007828.

## Training procedure

The training of modnet adopts the weights of the highest rate fixed-rate model as initialization. Download the baseline_model.zip and unzip it into the baseline_model/ folder and follow the instructions below to train modnet.

download address : https://drive.google.com/file/d/1qoOxiiRT_vQQfgN4v7epYaTH4CbXTfVu/view?usp=sharing

### For mse model

In the /RDM4NIC/ directory run command 'python train.py --data *' 

### For msssim model

In the /RDM4NIC/ directory run command 'python train_msssim.py --data *'

## Testing procedure

In the /RDM_in_NIC/ directory run command 'python test.py --lmd * --input *' . In * ,you can config as you need. Note that you should put the trained model into folder 'proposed_model'





