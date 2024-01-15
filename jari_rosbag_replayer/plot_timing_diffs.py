#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # parsing args for reading the inputted directory
    parser = argparse.ArgumentParser(description='Plot timing diffs')
    parser.add_argument('dir', type=str, help='directory to analyze')
    args = parser.parse_args()
    dir = args.dir

    filelist = [dir + file for file in os.listdir(dir) if os.path.isfile(dir + file)]

    # read csv data from files and plot the data in the same figure
    for file in filelist:
        print(f"reading {file}")
        # data = np.genfromtxt(file, delimiter=',', filling_values=np.nan)

        with open(file, 'r') as f:
            lines = f.readlines()

        with open(file, 'w') as f:
            for line in lines:
                # 空白（スペース、タブ）を削除
                cleaned_line = line.replace(" ", "").replace(",", "")
                f.write(cleaned_line + '\n')

        data = np.loadtxt(file, delimiter=',')
        data /= 1000000 # convert ns to ms
        plt.plot(data, label=file)
    # plt.legend()
    plt.title('frame offset between control command and simulation')
    plt.ylabel('stamp difference (ms)')
    plt.xlabel('frame number')
    plt.xlim(0, 800)
    plt.ylim(0, 30.0)
    # save the figure
    plt.savefig(dir + 'stamp_diff.svg')
    # png
    plt.savefig(dir + 'stamp_diff.png')
    plt.show()



