# TWIST
This is the pytorch implementation of TWIST. 

This paper has been accepted by [IEEE internet of things journal 2025.](https://doi.org/10.1109/JIOT.2025.3561542)

## Dependencies
The code is build on python 3.8, pytorch 1.11 and cuda 11.3. Run the following code to install python packages.

```
pip install pandas tables scipy h5py 
```

## Datasets
The original datasets can be download from [https://github.com/Davidham3/STSGCN], [https://github.com/liyaguang/DCRNN], [https://github.com/JIANGYUE61610306/SAGDFN], [https://github.com/LiuZH-19/ESG] and [https://github.com/liuxu77/LargeST]. Also, you can generate train、val、and test datesets by running the following code:
```
python data.py
```
Please note that data in different formats should be read in different ways. Moreover, different datasets has different sampling interval, it is necessary to change the the corresponding parameters (like "tod" and "dow") in the code! If you want to generate the datasets for long-term forecasting, please change the size of slide windows (if generate 60 steps, set "-11" to "-59" and "13" to "61" in line 79 and 82).

## Train Commands
Run the following code to train the model.

```
python train.py
```

Run the following code can test the pre-trained model.

```
python test.py
```
## Citation
If you find our work is helpful, please cite as:


```
@ARTICLE{10966151,
  author={Wang, Peng and Feng, Longxi and Zhang, Wenhao and Hui, Kanghua},
  journal={IEEE Internet of Things Journal}, 
  title={TWIST: An Efficient Spatial-Temporal Transformer With Temporal Window and Sparse Attention for Traffic Forecasting}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Forecasting;Computational modeling;Feature extraction;Transformers;Predictive models;Correlation;Attention mechanisms;Accuracy;Stacking;Long short term memory;Spatial-temporal data;Attention mechanism;Traffic forecasting;Deep learning},
  doi={10.1109/JIOT.2025.3561542}}
```
