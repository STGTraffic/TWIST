# Author: @liyaguang
# Url: https://github.com/liyaguang/DCRNN/blob/master/scripts/generate_training_data.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

def add_gaussian_noise(data, mean, std_dev, proportion):
    """
    给数据添加高斯噪声。
    
    参数：
    data (np.ndarray): 原始数据。
    mean (float): 高斯噪声的均值。
    std_dev (float): 高斯噪声的标准差。
    proportion (float): 添加噪声的比例，取值范围在0到1之间。
    
    返回：
    np.ndarray: 添加噪声后的数据。
    """
    noisy_data = data.copy()
    num_samples = data.shape[0]
    num_noisy_samples = int(num_samples * proportion)
    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    
    noise = np.random.normal(mean, std_dev, data[noisy_indices].shape)
    
    # 确保噪声为非负数
    noise = np.maximum(noise, 0)
    
    noisy_data[noisy_indices] += noise
    return noisy_data

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes,  = df.shape 
    df = pd.DataFrame(df)
    data = np.expand_dims(df.values, axis=-1) #(m,n,1)
    #data = pd.Dataframe(data)
    data_list = [data]
    if add_time_in_day:
        # numerical time_of_day
        tod = [i % 288 /
               288 for i in range(data.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(tod_tiled)


    if add_day_in_week:
        # numerical day_of_week
        dow = [(i // 288) % 7 for i in range(data.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(dow_tiled)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    # df = pd.read_hdf(args.traffic_df_filename)
    df = np.load(args.traffic_df_filename)['data']
    # print(df)
    df = df[:, :, 0]
    print(df.shape)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,
    )

    # 添加高斯噪声
    #noise_mean = 10
    #noise_std_dev = 500
    #noise_proportion = 0.05  # 调整为20%、40%或60%
    #x = add_gaussian_noise(x, noise_mean, noise_std_dev, noise_proportion)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )



def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/PEMS07", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/PEMS07/PEMS07.npz",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
