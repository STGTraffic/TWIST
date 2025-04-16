# TWIST
This is the pytorch implementation of TWIST. 

This paper has been accepted by [IEEE internet of things journal 2025.](https://doi.org/10.1016/j.inffus.2025.102978)

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
Please note that different datasets has different sampling interval, it is necessary to change the the corresponding parameters (like "tod") in the code!

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
@article{wang2025hybrid,
  title={Hybrid spatial--temporal graph neural network for traffic forecasting},
  author={Wang, Peng and Feng, Longxi and Zhu, Yijie and Wu, Haopeng},
  journal={Information Fusion},
  pages={102978},
  year={2025},
  publisher={Elsevier}
}
```
