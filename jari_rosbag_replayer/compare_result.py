#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os


class SeriesData:
    def __init__(self, name, dir):
        self.name = name
        self.dir = dir + name + '/'
        self.data = self.load_npy_file('data')
        self.stamp = self.load_npy_file('stamp')
        self.analyze_timestamp()

    def load_npy_file(self, name):
        npy_file = np.load(self.dir + name + '.npy')
        return npy_file

    def analyze_timestamp(self):
        stamp_diff = np.diff(self.stamp)
        stamp_diff = np.append(stamp_diff, stamp_diff[-1])
        plt.plot(self.stamp, stamp_diff)
        plt.title('stamp_diff of ' + self.name)
        plt.ylabel('stamp_diff (s)')
        plt.xlabel('time (s)')
        plt.savefig(self.dir + 'stamp_diff.svg')
        plt.close()
        np.save(self.dir + 'stamp_diff.npy', stamp_diff)


def get_diff_series(series1: SeriesData, series2: SeriesData):
    series2_data_stamp1 = np.interp(series1.stamp, series2.stamp, series2.data)
    return series1.data - series2_data_stamp1


class Result:
    def __init__(self, dir):
        self.dir = dir
        self.ttc_sim = SeriesData('ttc_sim', dir)
        self.ttc_real = SeriesData('ttc_real', dir)
        self.distance_sim = SeriesData('distance_sim', dir)
        self.distance_real = SeriesData('distance_real', dir)
        self.speed_sim = SeriesData('speed_sim', dir)
        self.speed_real = SeriesData('speed_real', dir)

    def analyze(self):
        self.ttc_diff_array = get_diff_series(self.ttc_sim, self.ttc_real)
        self.distance_diff_array = get_diff_series(self.distance_sim, self.distance_real)
        self.speed_diff_array = get_diff_series(self.speed_sim, self.speed_real)

        np.save(self.dir + 'ttc_diff_stamp_sim.npy', self.ttc_diff_array)
        np.save(self.dir + 'distance_diff_stamp_sim.npy', self.distance_diff_array)
        np.save(self.dir + 'speed_diff_stamp_sim.npy', self.speed_diff_array)
        #         plot diff series
        self.save_plot(self.ttc_sim.stamp, self.ttc_diff_array, 'ttc_diff (sim - real)', 'ttc (s)', 'time (s)',
                       'ttc_diff.svg')
        self.save_plot(self.distance_sim.stamp, self.distance_diff_array, 'distance_diff (sim - real)', 'distance (m)',
                       'time (s)', 'distance_diff.svg')
        self.save_plot(self.speed_sim.stamp, self.speed_diff_array, 'speed_diff (sim - real)', 'speed (m/s)',
                       'time (s)', 'speed_diff.svg')

    def save_plot(self, timestamp, data, title, ylabel, xlabel, file_name):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(timestamp, data)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        fig.savefig(self.dir + file_name)
        plt.close(fig)


if __name__ == '__main__':
    # make a list of folders in 'analysis' folder
    dir_list = os.listdir('analysis')
    # remove files from the list
    dir_list = [dir for dir in dir_list if os.path.isdir('analysis/' + dir)]
    result_list = []
    for dir in dir_list:
        result_list.append(Result('analysis/' + dir + '/'))

    for result in result_list:
        result.analyze()

    #     plot all diff series in one figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for result in result_list:
        ax.plot(result.ttc_sim.stamp, result.ttc_diff_array, label=result.dir)
    ax.legend()
    ax.set_title('ttc_diff (sim - real)')
    ax.set_ylabel('ttc (s)')
    ax.set_xlabel('time (s)')
    fig.savefig('analysis/ttc_diff.svg')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for result in result_list:
        ax.plot(result.distance_sim.stamp, result.distance_diff_array, label=result.dir)
    ax.legend()
    ax.set_title('distance_diff (sim - real)')
    ax.set_ylabel('distance (m)')
    ax.set_xlabel('time (s)')
    fig.savefig('analysis/distance_diff.svg')

    min_length = len(result_list[0].distance_diff_array)
    for result in result_list:
        if len(result.distance_diff_array) < min_length:
            min_length = len(result.distance_diff_array)
    distance_diff_average = np.zeros(min_length)
    for result in result_list:
        distance_diff_average += np.resize(result.distance_diff_array, min_length)
    distance_diff_average /= len(result_list)
    np.save('analysis/distance_diff_stamp_sim_average.npy', distance_diff_average)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for result in result_list:
        ax.plot(np.resize(result.distance_sim.stamp, min_length), np.resize(result.distance_diff_array, min_length) - distance_diff_average, label=result.dir)
    ax.set_title('distance_diff of diff (sim - real)')
    ax.set_ylabel('distance (m)')
    ax.set_xlabel('time (s)')
    fig.savefig('analysis/distance_diff_average.svg')


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for result in result_list:
        ax.plot(result.speed_sim.stamp, result.speed_sim.data, label=result.dir)
    ax.legend()
    ax.set_title('speed (sim)')
    ax.set_ylabel('speed (m/s)')
    ax.set_xlabel('time (s)')
    fig.savefig('analysis/speed_sim.svg')

    min_length = len(result_list[0].speed_sim.data)
    for result in result_list:
        if len(result.speed_sim.data) < min_length:
            min_length = len(result.speed_sim.data)
    speed_sim_average = np.zeros(min_length)
    for result in result_list:
        speed_sim_average += np.resize(result.speed_sim.data, min_length)
    speed_sim_average /= len(result_list)
    np.save('analysis/speed_diff_stamp_sim_average.npy', speed_sim_average)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for result in result_list:
        ax.plot(np.resize(result.speed_sim.stamp, min_length), np.resize(result.speed_sim.data, min_length) - speed_sim_average, label=result.dir)
    ax.set_title('speed diffs( base sim speed average)')
    ax.set_ylabel('speed diff (m/s)')
    ax.set_xlabel('time (s)')
    fig.savefig('analysis/speed_sim_average.svg')
    plt.close(fig)



