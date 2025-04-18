# TWIST
This is the pytorch implementation of TWIST. 

This paper has been accepted by [IEEE internet of things journal 2025.](https://ieeexplore.ieee.org/document/10966151)

## Dependencies
Run the following code to install python packages.

```
pip install pandas tables scipy 
```

## Datasets
The original datasets can be download from [https://github.com/Davidham3/STSGCN], [https://github.com/liyaguang/DCRNN], [https://github.com/JIANGYUE61610306/SAGDFN] and [https://github.com/liuxu77/LargeST]. Also, you can generate train、val、and test datesets by running the following code:
```
python data.py
```
Please note that data in different formats should be read in different ways. Moreover, different datasets has different sampling interval, it is necessary to change the the corresponding parameters (like "tod" and "dow") in the code! If you want to generate the datasets for long-term forecasting, please change the size of slide windows (if 60, set "-11" to "-59" and "13" to "61").

## Train Commands
Run the following code to train the model.

```
python train.py
```

The pre-trained models are provided, and run the following code to test the model.

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
