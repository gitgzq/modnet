# Readme
This work is published as paper in DCC 2022. 
Title: Rate Distortion Characteristic Modeling for Neural Image Compression
Cite as: Jia C, Ge Z, Wang S, et al. Rate distortion characteristic modeling for neural image compression[C]//2022 Data Compression Conference (DCC). IEEE, 2022: 202-211.


## Environment
First run 
```
pip install -r requirements.txt
```
for the installazation of python packages.

### Python Interface installation (you may encounter some bugs if use python2)

You need these files in the 'Util' folder：*AE.cpp*, *My_Range_Coder.h*, *My_Range_Encoder.cpp*, *My_Range_Decoder.cpp*.

**Step. 1**：  

Change the Python.h direction in the AE.cpp file for the direction of Python.h in your system. (e.g. /usr/include/python3.6m/Python.h）

**Step. 2**：

```
python setup.py build
python setup.py install
```

## Dataset        

Training dataset includes 600K cropped 256*256 images generated from LIU4K dataset. 

Reference: J. Liu, D. Liu, W. Yang, S. Xia, X. Zhang and Y. Dai, "A Comprehensive Benchmark for Single Image Compression Artifact Reduction," in IEEE Transactions on Image Processing, vol. 29, pp. 7845-7860, 2020, doi: 10.1109/TIP.2020.3007828.

## Training procedure

The training of modnet adopts the weights of the highest rate fixed-rate model as initialization. Download the baseline_model.zip and unzip it into the 'baseline_model' folder and follow the instructions below to train modnet.

download url : https://drive.google.com/file/d/1qoOxiiRT_vQQfgN4v7epYaTH4CbXTfVu/view?usp=sharing

### For mse model

In the /RDM4NIC/ directory run
```
python train.py --data * 
```
In * ,you can config as you need.  (same below)

### For msssim model

In the /RDM4NIC/ directory run
```
python train_msssim.py --data *
```
## Testing procedure

Put the trained model into folder 'proposed_model' and run
```
python test.py --lmd * --input *' . 
```




