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

    # filelist = [dir + file for file in os.listdir(dir) if os.path.isfile(dir + file)]
    filelist = [dir + file for file in os.listdir(dir) if os.path.isfile(dir + file) and file.endswith('.csv')]

    averages = []
    stds = []

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
                if cleaned_line != "\n":
                    f.write(cleaned_line)

        data = np.loadtxt(file, delimiter=',')
        data /= 1000000 # convert ns to ms
        data = data[data < 50]
        averages.append(np.mean(data))
        stds.append(np.std(data))
        plt.plot(data, label=file)
    # plt.legend()
    plt.title('frame offset between control command and simulation')
    plt.ylabel('stamp difference (ms)')
    plt.xlabel('frame number')
    plt.xlim(0, 1000)
    # plt.ylim(0, 30.0)
    # save the figure
    plt.savefig(dir + 'stamp_diff.svg')
    # png
    plt.savefig(dir + 'stamp_diff.png')
    plt.show()
    plt.close()
    
    # plot the averages as x and stds as y
    plt.plot(averages, stds, 'o')
    plt.title('averages and stds of stamp difference in each execution')
    plt.ylabel('std [ms]')
    plt.xlabel('average [ms]')
    plt.xlim(0, 30.0)
    plt.ylim(0, 15.0)
    plt.savefig(dir +'stamp_diff_std.svg')
    plt.savefig(dir +'stamp_diff_std.png')
    plt.show()
    plt.close()
    



