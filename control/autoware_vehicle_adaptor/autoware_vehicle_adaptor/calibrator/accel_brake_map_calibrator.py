import numpy as np
import scipy.interpolate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from autoware_vehicle_adaptor.calibrator import actuation_map_csv_writer
from autoware_vehicle_adaptor.scripts import actuation_map_2d
from autoware_vehicle_adaptor.calibrator import collected_data_counter
import GPy
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
import matplotlib.pyplot as plt
import json
from itertools import islice
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

control_dt = 0.033
default_map_accel = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
default_map_brake = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
default_map_vel=[0.0,1.39,2.78,4.17,5.56,6.94,8.33,9.72,11.11,12.5,13.89]
class AddDataFromCSV:
    def __init__(self):
        self.accel_data_input = []
        self.accel_data_output = []
        self.brake_data_input = []
        self.brake_data_output = []
        self.accel_data_input_for_poly = []
        self.accel_data_output_for_poly = []
        self.brake_data_input_for_poly = []
        self.brake_data_output_for_poly = []
        self.accel_data_input_for_GP = []
        self.accel_data_output_for_GP = []
        self.brake_data_input_for_GP = []
        self.brake_data_output_for_GP = []
        self.accel_data_input_for_NN = []
        self.accel_data_output_for_NN = []
        self.brake_data_input_for_NN = []
        self.brake_data_output_for_NN = []
        self.polynomial_regression_performed = False
        self.dataloader_weights_accel = None
        self.dataloader_weights_brake = None
        self.extracted_indices_accel = None
        self.extracted_indices_brake = None
        self.calc_base_map_error_performed = False
        self.collected_data_counter = collected_data_counter.CollectedDataCounter()
    def clear_data(self):
        self.accel_data_input = []
        self.accel_data_output = []
        self.brake_data_input = []
        self.brake_data_output = []
        self.accel_data_output_residual = []
        self.brake_data_output_residual = []
    def initialize_for_calibration(self):
        # Polynomial
        self.accel_data_input_for_poly = self.accel_data_input.copy()
        self.accel_data_output_for_poly = self.accel_data_output.copy()
        self.brake_data_input_for_poly = self.brake_data_input.copy()
        self.brake_data_output_for_poly = self.brake_data_output.copy()
        self.dataloader_weights_accel = None
        self.dataloader_weights_brake = None
        self.extracted_indices_accel = None
        self.extracted_indices_brake = None
        self.reset_GP_and_NN()
        if self.calc_base_map_error_performed:
            self.calc_base_map_error_performed = False
            print("calc_base_map_error is reset")
    def reset_GP_and_NN(self):
        self.accel_data_input_for_GP = self.accel_data_input_for_poly.copy()
        self.accel_data_output_for_GP = self.accel_data_output_for_poly.copy()
        self.brake_data_input_for_GP = self.brake_data_input_for_poly.copy()
        self.brake_data_output_for_GP = self.brake_data_output_for_poly.copy()
        self.accel_data_input_for_NN = self.accel_data_input_for_poly.copy()
        self.accel_data_output_for_NN = self.accel_data_output_for_poly.copy()
        self.brake_data_input_for_NN = self.brake_data_input_for_poly.copy()
        self.brake_data_output_for_NN = self.brake_data_output_for_poly.copy()
        if self.polynomial_regression_performed:
           self.polynomial_regression_performed = False
           print("polynomial regression as a preprocessing for GP and NN is reset")
    def add_data_from_csv(self, dir_name,smoothing_window = 10, acc_change_threshold=0.2):
        kinematic = np.loadtxt(
            dir_name + "/kinematic_state.csv", delimiter=",", usecols=[0, 1, 4, 5, 7, 8, 9, 10, 47]
        )
        vel = kinematic[:, 8]
        accel_data = np.loadtxt(dir_name + '/accel_cmd.csv', delimiter=',')
        accel = accel_data[:, 2]
        brake_data = np.loadtxt(dir_name + '/brake_cmd.csv', delimiter=',')
        brake = brake_data[:, 2]
        loc_acc = np.loadtxt(dir_name + "/acceleration.csv", delimiter=",", usecols=[0, 1, 3])
        acc = loc_acc[:, 2]
        min_time_stamp = max(
            [
                kinematic[0, 0] + 1e-9 * kinematic[0, 1],
                accel_data[0, 0] + 1e-9 * accel_data[0, 1],
                brake_data[0, 0] + 1e-9 * brake_data[0, 1],
                loc_acc[0, 0] + 1e-9 * loc_acc[0, 1],
            ]
        )
        max_time_stamp = min(
            [
                kinematic[-1, 0] + 1e-9 * kinematic[-1, 1],
                accel_data[-1, 0] + 1e-9 * accel_data[-1, 1],
                brake_data[-1, 0] + 1e-9 * brake_data[-1, 1],
                loc_acc[-1, 0] + 1e-9 * loc_acc[-1, 1],
            ]
        )
        data_num = int((max_time_stamp - min_time_stamp)/control_dt)
        data_time_stamps = min_time_stamp + control_dt * np.arange(data_num)
        vel_interp = scipy.interpolate.interp1d(kinematic[:, 0] + 1e-9 * kinematic[:, 1], vel)(data_time_stamps)
        accel_interp = scipy.interpolate.interp1d(accel_data[:, 0] + 1e-9 * accel_data[:, 1], accel)(data_time_stamps) 
        brake_interp = scipy.interpolate.interp1d(brake_data[:, 0] + 1e-9 * brake_data[:, 1], brake)(data_time_stamps)
        acc_interp = scipy.interpolate.interp1d(loc_acc[:, 0] + 1e-9 * loc_acc[:, 1], acc)(data_time_stamps)
        for i in range(smoothing_window, data_num-smoothing_window):
            vel_window = vel_interp[i-smoothing_window:i+smoothing_window+1]
            accel_window = accel_interp[i-smoothing_window:i+smoothing_window+1]
            brake_window = brake_interp[i-smoothing_window:i+smoothing_window+1]
            acc_window = acc_interp[i-smoothing_window:i+smoothing_window+1]
            acc_change = acc_window.max() - acc_window.min()
            if acc_change >= acc_change_threshold:
                continue
            if np.all(brake_window <= 0):
                self.accel_data_input.append(
                    np.array([vel_window.mean(),accel_window.mean()])
                )
                self.accel_data_output.append(acc_window.mean())
            elif np.all(accel_window <= 0):
                self.brake_data_input.append(
                    np.array([vel_window.mean(),brake_window.mean()])
                )
                self.brake_data_output.append(acc_window.mean())
        self.initialize_for_calibration()
    def outlier_exclusion_by_linear_regression(self):
        self.collected_data_counter.clear()
        for i in range(len(self.accel_data_input)):
            self.collected_data_counter.add_data_point(self.accel_data_input[i][0], self.accel_data_output[i], i)
        inlier_accel = self.collected_data_counter.outlier_exclusion_by_liniear_regression(self.accel_data_input,self.accel_data_output)
        self.collected_data_counter.clear()
        for i in range(len(self.brake_data_input)):
            self.collected_data_counter.add_data_point(self.brake_data_input[i][0], self.brake_data_output[i], i)
        inlier_brake = self.collected_data_counter.outlier_exclusion_by_liniear_regression(self.brake_data_input,self.brake_data_output)
        self.accel_data_input = np.array(self.accel_data_input)[inlier_accel].tolist()
        self.accel_data_output = np.array(self.accel_data_output)[inlier_accel].tolist()
        self.brake_data_input = np.array(self.brake_data_input)[inlier_brake].tolist()
        self.brake_data_output = np.array(self.brake_data_output)[inlier_brake].tolist()
        self.initialize_for_calibration()
    def calc_data_loader_weights(self,maximum_weight=0.02):
        self.collected_data_counter.clear()
        for i in range(len(self.accel_data_input)):
            self.collected_data_counter.add_data_point(self.accel_data_input[i][0], self.accel_data_output[i], i)
        self.dataloader_weights_accel = self.collected_data_counter.calc_weights(maximum_weight)
        self.collected_data_counter.clear()
        for i in range(len(self.brake_data_input)):
            self.collected_data_counter.add_data_point(self.brake_data_input[i][0], self.brake_data_output[i], i + len(self.accel_data_input))
        self.dataloader_weights_brake = self.collected_data_counter.calc_weights(maximum_weight)
        if self.extracted_indices_accel is not None:
            self.dataloader_weights_accel = np.array(self.dataloader_weights_accel)[self.extracted_indices_accel].tolist()
        if self.extracted_indices_brake is not None:
            self.dataloader_weights_brake = np.array(self.dataloader_weights_brake)[self.extracted_indices_brake].tolist()
    def extract_data_for_calibration(self,max_data_num=50):
        self.collected_data_counter.clear()
        for i in range(len(self.accel_data_input)):
            self.collected_data_counter.add_data_point(self.accel_data_input[i][0], self.accel_data_output[i], i)
        self.extracted_indices_accel = self.collected_data_counter.get_extracted_indices(max_data_num)
        self.collected_data_counter.clear()
        for i in range(len(self.brake_data_input)):
            self.collected_data_counter.add_data_point(self.brake_data_input[i][0], self.brake_data_output[i], i)
        self.extracted_indices_brake = self.collected_data_counter.get_extracted_indices(max_data_num)
        self.accel_data_input_for_poly = []
        self.accel_data_output_for_poly = []
        self.brake_data_input_for_poly = []
        self.brake_data_output_for_poly = []
        for i in self.extracted_indices_accel:
            self.accel_data_input_for_poly.append(self.accel_data_input[i])
            self.accel_data_output_for_poly.append(self.accel_data_output[i])
        for i in self.extracted_indices_brake:
            self.brake_data_input_for_poly.append(self.brake_data_input[i])
            self.brake_data_output_for_poly.append(self.brake_data_output[i])
        self.reset_GP_and_NN()
        if self.dataloader_weights_accel is not None:
            self.dataloader_weights_accel = np.array(self.dataloader_weights_accel)[self.extracted_indices_accel].tolist()
        if self.dataloader_weights_brake is not None:
            self.dataloader_weights_brake = np.array(self.dataloader_weights_brake)[self.extracted_indices_brake].tolist()
    def calc_base_map_error(self,base_map_dir):
        accel_map_path = base_map_dir + "/accel_map.csv"
        brake_map_path = base_map_dir + "/brake_map.csv"
        self.base_accel_map = actuation_map_2d.ActuationMap2D(accel_map_path)
        self.base_brake_map = actuation_map_2d.ActuationMap2D(brake_map_path)
        for i in range(len(self.accel_data_input)):
            accel_data = self.accel_data_input_for_poly[i]
            self.accel_data_output_for_poly[i] = self.accel_data_output_for_poly[i] - self.base_accel_map.get_sim_actuation(accel_data[0], accel_data[1])
        for i in range(len(self.brake_data_input)):
            brake_data = self.brake_data_input_for_poly[i]
            self.brake_data_output_for_poly[i] = self.brake_data_output_for_poly[i] - self.base_brake_map.get_sim_actuation(brake_data[0], brake_data[1])
        self.reset_GP_and_NN()
        self.calc_base_map_error_performed = True
    def calc_calibration_error(self,map_dir):
        accel_map_path = map_dir + "/accel_map.csv"
        brake_map_path = map_dir + "/brake_map.csv"
        accel_map = actuation_map_2d.ActuationMap2D(accel_map_path)
        brake_map = actuation_map_2d.ActuationMap2D(brake_map_path)
        accel_output_predicted = []
        brake_output_predicted = []
        for i in range(len(self.accel_data_input)):
            accel_data = self.accel_data_input[i]
            accel_output_predicted.append(accel_map.get_sim_actuation(accel_data[0], accel_data[1]))
        for i in range(len(self.brake_data_input)):
            brake_data = self.brake_data_input[i]
            brake_output_predicted.append(brake_map.get_sim_actuation(brake_data[0], brake_data[1]))
        accel_error = np.mean(np.abs(np.array(accel_output_predicted) - np.array(self.accel_data_output)))
        brake_error = np.mean(np.abs(np.array(brake_output_predicted) - np.array(self.brake_data_output)))
        print("accel_error: ", accel_error)
        print("brake_error: ", brake_error)
    def plot_training_data(self,save_dir=None):
        plt.scatter(
            np.array(self.accel_data_input_for_poly)[:, 0],
            np.array(self.accel_data_output_for_poly),
            label = "accel_data"
        )
        plt.scatter(
            np.array(self.brake_data_input_for_poly)[:, 0],
            np.array(self.brake_data_output_for_poly),
            label = "brake_data"
        )
        plt.title("vel vs acc plots")
        plt.xlabel("vel_obs [m/s]")
        plt.ylabel("acc_obs [m/s^2]")
        plt.legend()
        if save_dir is not None:
            plt.savefig(save_dir + "/training_data.png")
            plt.close()
        else:
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.array(self.accel_data_input_for_poly)[:, 0], np.array(self.accel_data_input_for_poly)[:, 1], np.array(self.accel_data_output_for_poly))
        ax.set_xlabel("vel_obs [m/s]")
        ax.set_ylabel("accel_input [m/s^2]")
        ax.set_zlabel("acc_obs [m/s^2]")
        if save_dir is not None:
            plt.savefig(save_dir + "/accel_3d.png")
            plt.close()
        else:
            plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.array(self.brake_data_input_for_poly)[:, 0], np.array(self.brake_data_input_for_poly)[:, 1], np.array(self.brake_data_output_for_poly))
        ax.set_xlabel("vel_obs [m/s]")
        ax.set_ylabel("brake_input [m/s^2]")
        ax.set_zlabel("acc_obs [m/s^2]")
        if save_dir is not None:
            plt.savefig(save_dir + "/brake_3d.png")
            plt.close()
        else:
            plt.show()

class CalibratorByPolynomialRegression(AddDataFromCSV):
    def __init__(self):
        super().__init__()
        self.accel_polynomial_model = None
        self.brake_polynomial_model = None
        self.degree = 3
        self.alpha_1 = 1e-5
        self.alpha_2 = 1e-5
    def calibrate_by_polynomial_regression(self,degree=None, alpha_1=None, alpha_2=None):
        if degree is not None:
            self.degree = degree
        if alpha_1 is not None:
            self.alpha_1 = alpha_1
        if alpha_2 is not None:
            self.alpha_2 = alpha_2
        alpha = self.alpha_1 + self.alpha_2
        l1_ratio = self.alpha_1 / alpha
        self.accel_data_output_for_NN.clear()
        self.brake_data_output_for_NN.clear()
        accel_data_output_poly_residual = []
        brake_data_output_poly_residual = []
        accel_data_input_poly = PolynomialFeatures(degree=self.degree).fit_transform(self.accel_data_input_for_poly)
        brake_data_input_poly = PolynomialFeatures(degree=self.degree).fit_transform(self.brake_data_input_for_poly)
        self.accel_polynomial_model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio).fit(accel_data_input_poly, self.accel_data_output_for_poly)
        self.brake_polynomial_model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio).fit(brake_data_input_poly, self.brake_data_output_for_poly)
        for i in range(len(self.accel_data_input_for_poly)):
            accel_data_output_poly_residual.append(
                self.accel_data_output_for_poly[i] - self.accel_polynomial_model.predict(accel_data_input_poly[i].reshape(1, -1))[0]
            )
        for i in range(len(self.brake_data_input_for_poly)):
            brake_data_output_poly_residual.append(
                self.brake_data_output_for_poly[i] - self.brake_polynomial_model.predict(brake_data_input_poly[i].reshape(1, -1))[0]
            )
        self.accel_data_input_for_GP = self.accel_data_input_for_poly.copy()
        self.accel_data_output_for_GP = accel_data_output_poly_residual.copy()
        self.brake_data_input_for_GP = self.brake_data_input_for_poly.copy()
        self.brake_data_output_for_GP = brake_data_output_poly_residual.copy()       
        self.accel_data_input_for_NN = self.accel_data_input_for_poly.copy()
        self.accel_data_output_for_NN = accel_data_output_poly_residual.copy()
        self.brake_data_input_for_NN = self.brake_data_input_for_poly.copy()
        self.brake_data_output_for_NN = brake_data_output_poly_residual.copy()
        self.polynomial_regression_performed = True
    def calc_accel_brake_map_poly(self, map_vel=default_map_vel, map_accel=default_map_accel, map_brake=default_map_brake):
        self.accel_map_matrix_poly = np.zeros((len(map_accel), len(map_vel)))
        self.brake_map_matrix_poly = np.zeros((len(map_brake), len(map_vel)))
        for i in range(len(map_vel)):
            for j in range(len(map_accel)):
                self.accel_map_matrix_poly[j, i] = self.accel_polynomial_model.predict(
                    PolynomialFeatures(degree=self.degree).fit_transform(np.array([[map_vel[i], map_accel[j]]]))
                )[0]
                if self.calc_base_map_error_performed:
                    self.accel_map_matrix_poly[j, i] += self.base_accel_map.get_sim_actuation(map_vel[i], map_accel[j])
            for j in range(len(map_brake)):
                self.brake_map_matrix_poly[j, i] = self.brake_polynomial_model.predict(
                    PolynomialFeatures(degree=self.degree).fit_transform(np.array([[map_vel[i], map_brake[j]]]))
                )[0]
                if self.calc_base_map_error_performed:
                    self.brake_map_matrix_poly[j, i] += self.base_brake_map.get_sim_actuation(map_vel[i], map_brake[j])
            self.accel_map_matrix_poly[0] = 0.5 * (self.brake_map_matrix_poly[0] + self.accel_map_matrix_poly[0])
            self.brake_map_matrix_poly[0] = self.accel_map_matrix_poly[0]
    def save_accel_brake_map_poly(self,map_vel, map_accel, map_brake, save_dir="."):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.calc_accel_brake_map_poly(map_vel, map_accel, map_brake)
        actuation_map_csv_writer.map_csv_writer(map_vel, map_accel, self.accel_map_matrix_poly, save_dir + "/accel_map.csv")
        actuation_map_csv_writer.map_csv_writer(map_vel, map_brake, self.brake_map_matrix_poly, save_dir + "/brake_map.csv")

class CalibratorByGaussianProcessRegression(AddDataFromCSV):
    def __init__(self):
        super().__init__()
        self.accel_GP_model = None
        self.brake_GP_model = None
    def calibrate_by_GP(self,inducting_num=100,):
        induction_index_accel = np.random.choice(len(self.accel_data_input_for_GP), inducting_num, replace=False)
        induction_index_brake = np.random.choice(len(self.brake_data_input_for_GP), inducting_num, replace=False)
        kernel_accel = GPy.kern.RBF(input_dim=self.accel_data_input_for_GP[0].shape[0], ARD=True)
        kernel_brake = GPy.kern.RBF(input_dim=self.brake_data_input_for_GP[0].shape[0], ARD=True)
        X_accel = np.array(self.accel_data_input_for_GP)
        Y_accel = np.array(self.accel_data_output_for_GP).reshape(-1, 1)
        X_brake = np.array(self.brake_data_input_for_GP)
        Y_brake = np.array(self.brake_data_output_for_GP).reshape(-1, 1)
        self.accel_GP_model = GPy.models.SparseGPRegression(
            X_accel, Y_accel, kernel_accel, Z=X_accel[induction_index_accel]
        )
        self.brake_GP_model = GPy.models.SparseGPRegression(
            X_brake, Y_brake, kernel_brake, Z=X_brake[induction_index_brake]
        )
        self.accel_GP_model.optimize()
        self.brake_GP_model.optimize()
    def calc_accel_brake_map_GP(self, map_vel=default_map_vel, map_accel=default_map_accel, map_brake=default_map_brake):
        self.accel_map_matrix_GP = np.zeros((len(map_accel), len(map_vel)))
        self.brake_map_matrix_GP = np.zeros((len(map_brake), len(map_vel)))
        for i in range(len(map_vel)):
            for j in range(len(map_accel)):
                self.accel_map_matrix_GP[j, i] = self.accel_GP_model.predict(
                    np.array([[map_vel[i], map_accel[j]]])
                )[0]
                if self.polynomial_regression_performed:
                    self.accel_map_matrix_GP[j, i] += self.accel_polynomial_model.predict(
                        PolynomialFeatures(degree=self.degree).fit_transform(np.array([[map_vel[i], map_accel[j]]]))
                    )[0]
                if self.calc_base_map_error_performed:
                    self.accel_map_matrix_GP[j, i] += self.base_accel_map.get_sim_actuation(map_vel[i], map_accel[j])
            for j in range(len(map_brake)):
                self.brake_map_matrix_GP[j, i] = self.brake_GP_model.predict(
                    np.array([[map_vel[i], map_brake[j]]])
                )[0]
                if self.polynomial_regression_performed:
                    self.brake_map_matrix_GP[j, i] += self.brake_polynomial_model.predict(
                        PolynomialFeatures(degree=self.degree).fit_transform(np.array([[map_vel[i], map_brake[j]]]))
                    )[0]
                if self.calc_base_map_error_performed:
                    self.brake_map_matrix_GP[j, i] += self.base_brake_map.get_sim_actuation(map_vel[i], map_brake[j])
            self.accel_map_matrix_GP[0] = 0.5 * (self.accel_map_matrix_GP[0] + self.brake_map_matrix_GP[0])
            self.brake_map_matrix_GP[0] = self.accel_map_matrix_GP[0]
    def save_accel_brake_map_GP(self, map_vel=default_map_vel, map_accel=default_map_accel, map_brake=default_map_brake, save_dir="."):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.calc_accel_brake_map_GP(map_vel, map_accel, map_brake)
        actuation_map_csv_writer.map_csv_writer(map_vel, map_accel, self.accel_map_matrix_GP, save_dir + "/accel_map.csv")
        actuation_map_csv_writer.map_csv_writer(map_vel, map_brake, self.brake_map_matrix_GP, save_dir + "/brake_map.csv")
    def data_augmentation_by_GP(self,map_vel=default_map_vel, map_accel=default_map_accel, map_brake=default_map_brake,minimum_cell_number=10):
        cell_accel_count = np.zeros((len(map_accel)-1,len(map_vel)-1))
        cell_brake_count = np.zeros((len(map_brake)-1,len(map_vel)-1))
        for i in range(len(self.accel_data_input)):
            for j in range(len(map_vel)-1):
                for k in range(len(map_accel)-1):
                    if map_vel[j] <= self.accel_data_input_for_NN[i][0] < map_vel[j+1] and map_accel[k] <= self.accel_data_input_for_NN[i][1] < map_accel[k+1]:
                        cell_accel_count[k,j] += 1
        for i in range(len(self.brake_data_input)):
            for j in range(len(map_vel)-1):
                for k in range(len(map_brake)-1):
                    if map_vel[j] <= self.brake_data_input_for_NN[i][0] < map_vel[j+1] and map_brake[k] <= self.brake_data_input_for_NN[i][1] < map_brake[k+1]:
                        cell_brake_count[k,j] += 1
        for j in range(len(map_vel)-1):
            for k in range(len(map_accel)-1):
                if cell_accel_count[k,j] < minimum_cell_number:
                    for i in range(minimum_cell_number-int(cell_accel_count[k,j])):
                        new_accel_data_input = np.array([np.random.uniform(map_vel[j],map_vel[j+1]),np.random.uniform(map_accel[k],map_accel[k+1])])
                        self.accel_data_input_for_NN.append(new_accel_data_input)
                        self.accel_data_output_for_NN.append(
                            self.accel_GP_model.predict(new_accel_data_input.reshape(1,-1))[0][0,0]
                        )
        for j in range(len(map_vel)-1):
            for k in range(len(map_brake)-1):
                if cell_brake_count[k,j] < minimum_cell_number:
                    for i in range(minimum_cell_number-int(cell_brake_count[k,j])):
                        new_brake_data_input = np.array([np.random.uniform(map_vel[j],map_vel[j+1]),np.random.uniform(map_brake[k],map_brake[k+1])])
                        self.brake_data_input_for_NN.append(new_brake_data_input)
                        self.brake_data_output_for_NN.append(
                            self.brake_GP_model.predict(new_brake_data_input.reshape(1,-1))[0][0,0]
                        )
class EarlyStopping:
    """Class for early stopping in NN training."""

    def __init__(self, initial_loss, tol=0.01, patience=30):
        self.epoch = 0  # Initialise the counter for the number of epochs being monitored.
        self.best_loss = float("inf")  # Initialise loss of comparison with infinity 'inf'.
        self.patience = patience  # Initialise the number of epochs to be monitored with a parameter
        self.initial_loss = initial_loss
        self.tol = tol

    def __call__(self, current_loss):
        current_loss_num = current_loss
        if current_loss_num + self.tol * self.initial_loss > self.best_loss:
            self.epoch += 1
        else:
            self.epoch = 0
        if current_loss_num < self.best_loss:
            self.best_loss = current_loss_num
        if self.epoch >= self.patience:
            return True
        return False

    def reset(self):
        self.epoch = 0

class CalibrationNN(nn.Module):
    def __init__(
        self,
    ):
        super(CalibrationNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
def validate_in_batches(model, criterion, X_val, Y_val, batch_size=10000):
    model.eval()
    val_loss = 0.0
    num_batches = (X_val.size(0) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_val.size(0))
            
            X_batch = X_val[start_idx:end_idx]
            Y_batch = Y_val[start_idx:end_idx]
            
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            val_loss += loss.item() * (end_idx - start_idx)
    
    val_loss /= X_val.size(0)
    return val_loss
class CalibratorByNeuralNetwork(AddDataFromCSV):
    def __init__(self):
        super().__init__()
        self.accel_NN_model = None
        self.brake_NN_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_iter = 10000
        self.tol = 1e-5
        self.alpha_1 = 1e-7
        self.alpha_2 = 1e-7
        self.max_sample_per_epoch = 2500
    def train_calibrator_NN(
        self,
        model,
        X,
        y,
        batch_size,
        learning_rates,
        patience,
        weights=None,
    ):
        sample_size = X.shape[0]
        print("sample size: ", sample_size)
        print("patience: ", patience)
        num_train = int(3 * sample_size / 4)
        id_all = np.random.choice(sample_size, sample_size, replace=False)
        id_train = id_all[:num_train]
        id_val = id_all[num_train:]
        X_train = X[id_train]
        y_train = y[id_train]
        X_val = X[id_val]
        y_val = y[id_val]
        # Define the loss function.
        criterion = nn.L1Loss()
        # Define the optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])
        # Define the initial loss.
        initial_loss = validate_in_batches(model,criterion,X_train, y_train.view(-1, 1))
        print("initial_loss: ", initial_loss)
        # Define the early stopping.
        early_stopping = EarlyStopping(initial_loss, tol=self.tol, patience=patience)
        # Data Loader
        if weights is None:
            train_dataset = DataLoader(
                TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
            )
        else:
            weights_train = np.array(weights)[id_train]
            sampler = WeightedRandomSampler(weights=weights_train, num_samples = len(weights_train), replacement=True)
            train_dataset = DataLoader(
                TensorDataset(X_train, y_train), batch_size=batch_size, sampler = sampler
            )
        # learning_rate index
        learning_rate_index = 0
        # Print learning rate
        print("learning rate: ", learning_rates[learning_rate_index])
        for i in range(self.max_iter):
            model.train()
            for X_batch, y_batch in islice(train_dataset, round(self.max_sample_per_epoch/batch_size) + 1):
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                for w in model.parameters():
                    loss += self.alpha_1 * torch.norm(w, 1) + self.alpha_2 * torch.norm(w, 2) ** 2
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = validate_in_batches(model,criterion,X_val, y_val.view(-1, 1), batch_size)
            if i % 10 == 1:
                print(val_loss, i)
            if early_stopping(val_loss):
                learning_rate_index += 1
                if learning_rate_index >= len(learning_rates):
                    break
                else:
                    print("update learning rate to ", learning_rates[learning_rate_index])
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=learning_rates[learning_rate_index]
                    )
                    early_stopping.reset()
    def calibrate_by_NN(self,learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], patience=50, batch_size=100,
                        max_iter=None, tol=None, alpha_1=None, alpha_2=None):
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        if alpha_1 is not None:
            self.alpha_1 = alpha_1
        if alpha_2 is not None:
            self.alpha_2 = alpha_2
        self.accel_NN_model = CalibrationNN().to(self.device)
        self.brake_NN_model = CalibrationNN().to(self.device)


        X_accel = torch.tensor(np.array(self.accel_data_input_for_NN), dtype=torch.float32,device=self.device)
        y_accel = torch.tensor(np.array(self.accel_data_output_for_NN), dtype=torch.float32,device=self.device)
        X_brake = torch.tensor(np.array(self.brake_data_input_for_NN), dtype=torch.float32,device=self.device)
        y_brake = torch.tensor(np.array(self.brake_data_output_for_NN), dtype=torch.float32,device=self.device)

        self.train_calibrator_NN(
            self.accel_NN_model,
            X_accel,
            y_accel,
            batch_size,
            learning_rates,
            patience,
        )
        self.train_calibrator_NN(
            self.brake_NN_model,
            X_brake,
            y_brake,
            batch_size,
            learning_rates,
            patience,
        )
    def calc_accel_brake_map_NN(self,map_vel=default_map_vel, map_accel=default_map_accel, map_brake=default_map_brake):
        self.accel_NN_model.eval()
        self.brake_NN_model.eval()
        self.accel_map_matrix_NN = np.zeros((len(map_accel), len(map_vel)))
        self.brake_map_matrix_NN = np.zeros((len(map_brake), len(map_vel)))
        for i in range(len(map_vel)):
            for j in range(len(map_accel)):
                self.accel_map_matrix_NN[j, i] = self.accel_NN_model(
                    torch.tensor([map_vel[i], map_accel[j]], dtype=torch.float32,device=self.device)
                ).item()
                if self.polynomial_regression_performed:
                    self.accel_map_matrix_NN[j, i] += self.accel_polynomial_model.predict(
                        PolynomialFeatures(degree=self.degree).fit_transform(np.array([[map_vel[i], map_accel[j]]]))
                    )[0]
                if self.calc_base_map_error_performed:
                    self.accel_map_matrix_NN[j, i] += self.base_accel_map.get_sim_actuation(map_vel[i], map_accel[j])
            for j in range(len(map_brake)):
                self.brake_map_matrix_NN[j, i] = self.brake_NN_model(
                    torch.tensor([map_vel[i], map_brake[j]], dtype=torch.float32,device=self.device)
                ).item()
                if self.polynomial_regression_performed:
                    self.brake_map_matrix_NN[j, i] += self.brake_polynomial_model.predict(
                        PolynomialFeatures(degree=self.degree).fit_transform(np.array([[map_vel[i], map_brake[j]]]))
                    )[0]
                if self.calc_base_map_error_performed:
                    self.brake_map_matrix_NN[j, i] += self.base_brake_map.get_sim_actuation(map_vel[i], map_brake[j])
            self.accel_map_matrix_NN[0] = 0.5 * (self.accel_map_matrix_NN[0] + self.brake_map_matrix_NN[0])
            self.brake_map_matrix_NN[0] = self.accel_map_matrix_NN[0]
    def save_accel_brake_map_NN(self,map_vel=default_map_vel, map_accel=default_map_accel, map_brake=default_map_brake, save_dir="."):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.calc_accel_brake_map_NN(map_vel, map_accel, map_brake)
        actuation_map_csv_writer.map_csv_writer(map_vel, map_accel, self.accel_map_matrix_NN, save_dir + "/accel_map.csv")
        actuation_map_csv_writer.map_csv_writer(map_vel, map_brake, self.brake_map_matrix_NN, save_dir + "/brake_map.csv")

class CalibratorByEnsembleNN(CalibratorByNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.accel_NN_models = []
        self.brake_NN_models = []
        self.ensemble_num = 5
    def calibrate_by_ensemble_NN(self,learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], patience=50, batch_size=100,
                                 max_iter=None, tol=None, alpha_1=None, alpha_2=None, ensemble_num=None, clear_model=False):
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        if alpha_1 is not None:
            self.alpha_1 = alpha_1
        if alpha_2 is not None:
            self.alpha_2 = alpha_2
        if ensemble_num is not None:
            if clear_model:
                self.ensemble_num = ensemble_num
            elif ensemble_num != self.ensemble_num:
                print("ensemble_num is not updated because clear_model is False")
        if clear_model:
            self.accel_NN_models.clear()
            self.brake_NN_models.clear()
        for i in range(self.ensemble_num):
            print("______________________________")
            print("ensemble number: ", i)
            print("______________________________")
            self.accel_NN_models.append(CalibrationNN().to(self.device))
            self.brake_NN_models.append(CalibrationNN().to(self.device))
            X_accel = torch.tensor(np.array(self.accel_data_input_for_NN), dtype=torch.float32,device=self.device)
            y_accel = torch.tensor(np.array(self.accel_data_output_for_NN), dtype=torch.float32,device=self.device)
            X_brake = torch.tensor(np.array(self.brake_data_input_for_NN), dtype=torch.float32,device=self.device)
            y_brake = torch.tensor(np.array(self.brake_data_output_for_NN), dtype=torch.float32,device=self.device)
            print("calibrate accel")
            self.train_calibrator_NN(
                self.accel_NN_models[i],
                X_accel,
                y_accel,
                batch_size,
                learning_rates,
                patience,
                self.dataloader_weights_accel,
            )
            print("calibrate brake")
            self.train_calibrator_NN(
                self.brake_NN_models[i],
                X_brake,
                y_brake,
                batch_size,
                learning_rates,
                patience,
                self.dataloader_weights_brake,
            )
    def calc_accel_brake_map_ensemble_NN(self,map_vel=default_map_vel, map_accel=default_map_accel, map_brake=default_map_brake):
        self.accel_map_matrix_ensemble_NN = np.zeros((self.ensemble_num, len(map_accel), len(map_vel)))
        self.brake_map_matrix_ensemble_NN = np.zeros((self.ensemble_num, len(map_brake), len(map_vel)))
        for k in range(self.ensemble_num):
            for i in range(len(map_vel)):
                for j in range(len(map_accel)):
                    self.accel_map_matrix_ensemble_NN[k, j, i] = self.accel_NN_models[k](
                        torch.tensor([map_vel[i], map_accel[j]], dtype=torch.float32,device=self.device)
                    ).item()
                    if self.polynomial_regression_performed:
                        self.accel_map_matrix_ensemble_NN[k, j, i] += self.accel_polynomial_model.predict(
                            PolynomialFeatures(degree=self.degree).fit_transform(np.array([[map_vel[i], map_accel[j]]]))
                        )[0]
                    if self.calc_base_map_error_performed:
                        self.accel_map_matrix_ensemble_NN[k, j, i] += self.base_accel_map.get_sim_actuation(map_vel[i], map_accel[j])
                for j in range(len(map_brake)):
                    self.brake_map_matrix_ensemble_NN[k, j, i] = self.brake_NN_models[k](
                        torch.tensor([map_vel[i], map_brake[j]], dtype=torch.float32,device=self.device)
                    ).item()
                    if self.polynomial_regression_performed:
                        self.brake_map_matrix_ensemble_NN[k, j, i] += self.brake_polynomial_model.predict(
                            PolynomialFeatures(degree=self.degree).fit_transform(np.array([[map_vel[i], map_brake[j]]]))
                        )[0]
                    if self.calc_base_map_error_performed:
                        self.brake_map_matrix_ensemble_NN[k, j, i] += self.base_brake_map.get_sim_actuation(map_vel[i], map_brake[j])
                self.accel_map_matrix_ensemble_NN[k,0] = 0.5 * (self.accel_map_matrix_ensemble_NN[k,0] + self.brake_map_matrix_ensemble_NN[k,0])
                self.brake_map_matrix_ensemble_NN[k,0] = self.accel_map_matrix_ensemble_NN[k,0]
        self.accel_map_matrix_NN = np.mean(self.accel_map_matrix_ensemble_NN, axis=0)
        self.brake_map_matrix_NN = np.mean(self.brake_map_matrix_ensemble_NN, axis=0)
        self.accel_map_matrix_std_NN = np.std(self.accel_map_matrix_ensemble_NN, axis=0)
        self.brake_map_matrix_std_NN = np.std(self.brake_map_matrix_ensemble_NN, axis=0)
    def save_accel_brake_map_ensemble_NN(self,map_vel=default_map_vel, map_accel=default_map_accel, map_brake=default_map_brake, save_dir=".",save_heat_map=False, true_map_dir=None):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.calc_accel_brake_map_ensemble_NN(map_vel, map_accel, map_brake)
        actuation_map_csv_writer.map_csv_writer(map_vel, map_accel, self.accel_map_matrix_NN, save_dir + "/accel_map.csv")
        actuation_map_csv_writer.map_csv_writer(map_vel, map_brake, self.brake_map_matrix_NN, save_dir + "/brake_map.csv")
        actuation_map_csv_writer.map_csv_writer(map_vel, map_accel, self.accel_map_matrix_std_NN, save_dir + "/accel_map_std.csv")
        actuation_map_csv_writer.map_csv_writer(map_vel, map_brake, self.brake_map_matrix_std_NN, save_dir + "/brake_map_std.csv")
        if save_heat_map:
            if true_map_dir is None:
                package_path_json = str(Path(__file__).parent.parent) + "/package_path.json"
                with open(package_path_json, "r") as file:
                    package_path = json.load(file)

                true_map_dir = (
                    package_path["path"] + "/autoware_vehicle_adaptor/actuation_cmd_maps/accel_brake_maps/default_parameter"
                )
            true_accel_map_path = true_map_dir + "/accel_map.csv"
            true_brake_map_path = true_map_dir + "/brake_map.csv"
            true_accel_map = actuation_map_2d.ActuationMap2D(true_accel_map_path)
            true_brake_map = actuation_map_2d.ActuationMap2D(true_brake_map_path)
            prediction_signed_error_accel = - self.accel_map_matrix_NN
            prediction_signed_error_brake = - self.brake_map_matrix_NN
            for i in range(len(map_vel)):
                for j in range(len(map_accel)):
                    prediction_signed_error_accel[j, i] += true_accel_map.get_sim_actuation(map_vel[i], map_accel[j])
                for j in range(len(map_brake)):
                    prediction_signed_error_brake[j, i] += true_brake_map.get_sim_actuation(map_vel[i], map_brake[j])
        fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(24, 15),tight_layout=True)
        fig.suptitle("calibration result")
        error_norm = TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5)
        std_norm = TwoSlopeNorm(vcenter=0, vmin=-0.2, vmax=0.2)
        sns.heatmap(prediction_signed_error_accel,
                    annot=True, cmap="coolwarm", 
                    xticklabels=map_vel, 
                    yticklabels=map_accel,
                    ax=axes[0,0],
                    linewidths=0.1,
                    linecolor="gray",
                    norm=error_norm,
                    )
        axes[0,0].set_xlabel("velocity [m/s]")
        axes[0,0].set_ylabel("accel")
        axes[0,0].set_title("Prediction error accel (true - prediction)")
        sns.heatmap(self.accel_map_matrix_std_NN,
                    annot=True, cmap="coolwarm", 
                    xticklabels=map_vel, 
                    yticklabels=map_accel,
                    ax=axes[0,1],
                    linewidths=0.1,
                    linecolor="gray",
                    norm=std_norm
                    )
        axes[0,1].set_xlabel("velocity [m/s]")
        axes[0,1].set_ylabel("accel")
        axes[0,1].set_title("Prediction std accel")
        sns.heatmap(prediction_signed_error_brake,
                    annot=True, cmap="coolwarm", 
                    xticklabels=map_vel, 
                    yticklabels=map_accel,
                    ax=axes[1,0],
                    linewidths=0.1,
                    linecolor="gray",
                    norm=error_norm,
                    )
        axes[1,0].set_xlabel("velocity [m/s]")
        axes[1,0].set_ylabel("brake")
        axes[1,0].set_title("Prediction error brake (true - prediction)")
        sns.heatmap(self.brake_map_matrix_std_NN,
                    annot=True, cmap="coolwarm", 
                    xticklabels=map_vel, 
                    yticklabels=map_brake,
                    ax=axes[1,1],
                    linewidths=0.1,
                    linecolor="gray",
                    norm=std_norm
                    )
        axes[1,1].set_xlabel("velocity [m/s]")
        axes[1,1].set_ylabel("brake")
        axes[1,1].set_title("Prediction std brake")
        plt.savefig(save_dir + "/calibration_result.png")
        plt.close()


class Calibrator(CalibratorByPolynomialRegression,CalibratorByGaussianProcessRegression,CalibratorByEnsembleNN):
    def __init__(self):
        super().__init__()