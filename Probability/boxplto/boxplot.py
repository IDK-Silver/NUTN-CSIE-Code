import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def calculate_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return iqr


def find_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if x < lower_bound or x > upper_bound]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_path",
        help="The data path that want to boxplot"
    )

    parser.add_argument(
        "result_image_path",
        help="The image of boxplot"
    )

    args = parser.parse_args()

    # load cvs data

    df = pd.read_csv(args.data_path) 
    labels = [data for data in df.axes[1]]
    datas = [df[lab].tolist() for lab in labels]

    # init plot object
    fig, ax = plt.subplots()

    # set plot data and info
    ax.set_title('Box Plot of Nuclear Power')
    ax.set_ylabel('GW')

    # plot boxplot
    ax.boxplot(datas, tick_labels=labels)

    

    save_img_filepath = os.path.abspath(
        args.result_image_path
    )

    # save result
    plt.savefig(save_img_filepath)
    print(f'the boxplot image save to {save_img_filepath}')

    # calculate iqr for each data
    iqr_list = [calculate_iqr(data) for data in datas]
    # print the result
    for index, iqr in enumerate(iqr_list):
        print(f'IQR {labels[index]}: {iqr}')

    # calculate outliers for each data
    outliers_list = [find_outliers(data) for data in datas]
    # print the result
    for index, outliers in enumerate(outliers_list):
        print(f'Outliers {labels[index]}: {outliers}')