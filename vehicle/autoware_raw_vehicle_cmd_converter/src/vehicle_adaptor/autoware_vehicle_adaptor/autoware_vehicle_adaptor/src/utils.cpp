#include <Eigen/Core>
#include <Eigen/Dense>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include "utils.h"

namespace py = pybind11;
using namespace Proxima;
//Eigen::VectorXd tanh(const Eigen::VectorXd & v)
//{
//  return v.array().tanh();
//}
Eigen::VectorXd d_tanh(const Eigen::VectorXd & v)
{
  return 1 / (v.array().cosh() * v.array().cosh());
}
//Eigen::VectorXd sigmoid(const Eigen::VectorXd & v)
//{
//  return 0.5 * (0.5 * v).array().tanh() + 0.5;
//}
//Eigen::VectorXd relu(const Eigen::VectorXd & x)
//{
 // Eigen::VectorXd x_ = x;
//  for (int i = 0; i < x.size(); i++) {
//    if (x[i] < 0) {
//      x_[i] = 0;
//    }
//  }
//  return x_;
//}
Eigen::VectorXd d_relu(const Eigen::VectorXd & x)
{
  Eigen::VectorXd result = Eigen::VectorXd::Ones(x.size());
  for (int i = 0; i < x.size(); i++) {
    if (x[i] < 0) {
      result[i] = 0;
    }
  }
  return result;
}
Eigen::MatrixXd d_relu_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x)
{
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(m.rows(), m.cols());
  for (int i = 0; i < m.cols(); i++) {
    if (x[i] >= 0) {
      result.col(i) = m.col(i);
    }
  }
  return result;
}
Eigen::MatrixXd d_tanh_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x)
{
  Eigen::MatrixXd result = Eigen::MatrixXd(m.rows(), m.cols());
  for (int i = 0; i < m.cols(); i++) {
    result.col(i) = m.col(i) / (std::cosh(x[i]) * std::cosh(x[i]));
  }
  return result;
}
Eigen::VectorXd d_tanh_product_vec(const Eigen::VectorXd & v, const Eigen::VectorXd & x)
{
  Eigen::VectorXd result = Eigen::VectorXd(v.size());
  for (int i = 0; i < v.size(); i++) {
    result[i] = v[i] / (std::cosh(x[i]) * std::cosh(x[i]));
  }
  return result;
}
Eigen::MatrixXd d_sigmoid_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x)
{
  Eigen::MatrixXd result = Eigen::MatrixXd(m.rows(), m.cols());
  for (int i = 0; i < m.cols(); i++) {
    result.col(i) = 0.25 * m.col(i) / (std::cosh(0.5 * x[i]) * std::cosh(0.5 * x[i]));
  }
  return result;
}
Eigen::VectorXd d_sigmoid_product_vec(const Eigen::VectorXd & v, const Eigen::VectorXd & x)
{
  Eigen::VectorXd result = Eigen::VectorXd(v.size());
  for (int i = 0; i < v.size(); i++) {
    result[i] = 0.25 * v[i] / (std::cosh(0.5 * x[i]) * std::cosh(0.5 * x[i]));
  }
  return result;
}

Eigen::VectorXd rotate_data(Eigen::VectorXd states, double yaw)
{
  Eigen::VectorXd states_rotated = states;
  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  states_rotated[0] = states[0] * cos_yaw + states[1] * sin_yaw;
  states_rotated[1] = -states[1] * sin_yaw + states[0] * cos_yaw;
  return states_rotated;
}

Eigen::VectorXd vector_power(Eigen::VectorXd vec, int power)
{
  Eigen::VectorXd result = vec;
  for (int i = 0; i < power - 1; i++) {
    result = result.array() * vec.array();
  }
  return result;
}

double double_power(double val, int power)
{
  double result = val;
  for (int i = 0; i < power - 1; i++) {
    result = result * val;
  }
  return result;
}

double calc_table_value(double domain_value, std::vector<double> domain_table, std::vector<double> target_table)
{
  if (domain_value <= domain_table[0]) {
    return target_table[0];
  }
  if (domain_value >= domain_table[domain_table.size() - 1]) {
    return target_table[target_table.size() - 1];
  }
  for (int i = 0; i < int(domain_table.size()) - 1; i++) {
    if (domain_value >= domain_table[i] && domain_value <= domain_table[i + 1]) {
      return target_table[i] +
             (target_table[i + 1] - target_table[i]) /
               (domain_table[i + 1] - domain_table[i]) *
               (domain_value - domain_table[i]);
    }
  }
  return 0.0;
}

std::string get_param_dir_path()
{
  std::string build_path = BUILD_PATH;
  std::string param_dir_path = build_path + "/autoware_vehicle_adaptor/param";
  return param_dir_path;
}
/*
Eigen::VectorXd interpolate_eigen(Eigen::VectorXd y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new)
{
  Eigen::VectorXd y_new = Eigen::VectorXd(time_stamp_new.size());
  int lower_bound_index = 0;
  for (int i = 0; i < int(time_stamp_new.size()); i++) {
    if (time_stamp_new[i] >= time_stamp_obs[time_stamp_obs.size() - 1]) {
      y_new[i] = y[time_stamp_obs.size() - 1] + (y[time_stamp_obs.size() - 1] - y[time_stamp_obs.size() - 2]) / (time_stamp_obs[time_stamp_obs.size() - 1] - time_stamp_obs[time_stamp_obs.size() - 2]) * (time_stamp_new[i] - time_stamp_obs[time_stamp_obs.size() - 1]);
      continue;
    }
    for (int j = lower_bound_index; j < int(time_stamp_obs.size()) - 1; j++) {
      if (time_stamp_new[i] >= time_stamp_obs[j] && time_stamp_new[i] <= time_stamp_obs[j + 1]) {
        y_new[i] = y[j] + (y[j + 1] - y[j]) / (time_stamp_obs[j + 1] - time_stamp_obs[j]) * (time_stamp_new[i] - time_stamp_obs[j]);
        lower_bound_index = j;
        break;
      }
    }
  }
  return y_new;
}
std::vector<Eigen::VectorXd> interpolate_vector(std::vector<Eigen::VectorXd> y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new)
{
  std::vector<Eigen::VectorXd> y_new;
  int lower_bound_index = 0;
  for (int i = 0; i < int(time_stamp_new.size()); i++) {
    if (time_stamp_new[i] >= time_stamp_obs[time_stamp_obs.size() - 1]) {
      y_new.push_back(y[time_stamp_obs.size() - 1] + (y[time_stamp_obs.size() - 1] - y[time_stamp_obs.size() - 2]) / (time_stamp_obs[time_stamp_obs.size() - 1] - time_stamp_obs[time_stamp_obs.size() - 2]) * (time_stamp_new[i] - time_stamp_obs[time_stamp_obs.size() - 1]));
      continue;
    }
    for (int j = lower_bound_index; j < int(time_stamp_obs.size()) - 1; j++) {
      if (time_stamp_new[i] >= time_stamp_obs[j] && time_stamp_new[i] <= time_stamp_obs[j + 1]) {
        y_new.push_back(y[j] + (y[j + 1] - y[j]) / (time_stamp_obs[j + 1] - time_stamp_obs[j]) * (time_stamp_new[i] - time_stamp_obs[j]));
        lower_bound_index = j;
        break;
      }
    }
  }
  return y_new;
}
*/
Eigen::VectorXd states_vehicle_to_world(Eigen::VectorXd states_vehicle, double yaw)
{
  Eigen::VectorXd states_world = states_vehicle;
  states_world[0] = states_vehicle[0] * std::cos(yaw) - states_vehicle[1] * std::sin(yaw);
  states_world[1] = states_vehicle[0] * std::sin(yaw) + states_vehicle[1] * std::cos(yaw);
  return states_world;
}
Eigen::VectorXd states_world_to_vehicle(Eigen::VectorXd states_world, double yaw)
{
  Eigen::VectorXd states_vehicle = states_world;
  states_vehicle[0] = states_world[0] * std::cos(yaw) + states_world[1] * std::sin(yaw);
  states_vehicle[1] = -states_world[0] * std::sin(yaw) + states_world[1] * std::cos(yaw);
  return states_vehicle;
}
/*
Eigen::MatrixXd read_csv(std::string file_path)
{
  std::string build_path = BUILD_PATH;
  std::string file_path_abs = build_path + "/" + file_path;
  std::vector<std::vector<double>> csv_data;
  std::ifstream ifs(file_path_abs);
  if (!ifs) {
    std::cerr << "Failed to open file." << std::endl;
    return Eigen::MatrixXd(0, 0);
  }
  std::string line;
  while (std::getline(ifs, line)) {
    std::istringstream stream(line);
    std::string field;
    std::vector<double> fields;
    while (std::getline(stream, field, ',')) {
      fields.push_back(std::stod(field));
    }
    csv_data.push_back(fields);
  }
  ifs.close();
  Eigen::MatrixXd data = Eigen::MatrixXd(csv_data.size(), csv_data[0].size());
  for (int i = 0; i < int(csv_data.size()); i++) {
    for (int j = 0; j < int(csv_data[0].size()); j++) {
      data(i, j) = csv_data[i][j];
    }
  }
  return data;
}
*/
///////////////// Polynomial Regression ///////////////////////

  PolynomialRegression::PolynomialRegression() {}
  PolynomialRegression::~PolynomialRegression() {}
  void PolynomialRegression::set_params(int degree, int num_samples, std::vector<double> lambda)
  {
    degree_ = degree;
    num_samples_ = num_samples;
    lambda_ = lambda;
  }
  void PolynomialRegression::set_minimum_decay(double minimum_decay) { minimum_decay_ = minimum_decay; }
  void PolynomialRegression::set_ignore_intercept() { predict_and_ignore_intercept_ = true; }
  void PolynomialRegression::calc_coef_matrix()
  {
    if (!predict_and_ignore_intercept_) {
      Eigen::VectorXd time_vector = Eigen::VectorXd::LinSpaced(num_samples_, 0.0, num_samples_ - 1);
      Eigen::MatrixXd time_matrix = Eigen::MatrixXd::Ones(num_samples_, degree_ + 1);
    
      for (int i = 1; i < degree_; i++) {
        time_matrix.col(i) = time_matrix.col(i - 1).array() * time_vector.array();
      }
      Eigen::MatrixXd regularized_matrix = Eigen::MatrixXd::Identity(degree_ + 1, degree_ + 1);
      regularized_matrix(0, 0) = 0.0;
      for (int i = 0; i < degree_; i++) {
        regularized_matrix(i+1, i+1) = lambda_[i];
      }
      Eigen::VectorXd decay_vector = Eigen::VectorXd::LinSpaced(num_samples_, 1.0, minimum_decay_);
      Eigen::MatrixXd decay_matrix = decay_vector.asDiagonal();
      weight_matrix_ = time_matrix *
                    (time_matrix.transpose() * decay_matrix *time_matrix + regularized_matrix).inverse() *
                    time_matrix.transpose() * decay_matrix;
      coef_matrix_ = (time_matrix.transpose() * decay_matrix * time_matrix + regularized_matrix).inverse() *
                    time_matrix.transpose() * decay_matrix;
    }
    else {
      Eigen::VectorXd time_vector = Eigen::VectorXd::LinSpaced(num_samples_ -1 , - (num_samples_ - 1), - 1.0);
      Eigen::MatrixXd time_matrix = Eigen::MatrixXd::Zero(num_samples_ - 1 , degree_);

      time_matrix.col(0) = time_vector;

      for (int i = 1; i < degree_; i++) {
        time_matrix.col(i) = time_matrix.col(i - 1).array() * time_vector.array();
      }
      Eigen::VectorXd decay_vector = Eigen::VectorXd::LinSpaced(num_samples_ - 1, minimum_decay_, 1.0);
      Eigen::MatrixXd decay_matrix = decay_vector.asDiagonal();

      Eigen::MatrixXd regularized_matrix = Eigen::MatrixXd::Identity(degree_, degree_);
      for (int i = 0; i < degree_; i++) {
        regularized_matrix(i, i) = lambda_[i];
      }
      weight_matrix_ = time_matrix *
                    (time_matrix.transpose() * decay_matrix * time_matrix +
                      regularized_matrix).inverse() *
                    time_matrix.transpose() * decay_matrix;

      coef_matrix_ = (time_matrix.transpose() * decay_matrix * time_matrix +
                      regularized_matrix).inverse() *
                    time_matrix.transpose() * decay_matrix;
    } 

  }
  void PolynomialRegression::calc_prediction_matrix(int horizon_len)
  {
    Eigen::VectorXd time_vector = Eigen::VectorXd::LinSpaced(horizon_len, 1.0, horizon_len);
    Eigen::MatrixXd pre_prediction_matrix = Eigen::MatrixXd::Zero(horizon_len, degree_);
    pre_prediction_matrix.col(0) = time_vector;
    for (int i = 1; i < degree_; i++) {
      pre_prediction_matrix.col(i) = pre_prediction_matrix.col(i - 1).array() * time_vector.array();
    }
    prediction_matrix_ = pre_prediction_matrix * coef_matrix_;
  }
  Eigen::VectorXd PolynomialRegression::fit_transform(Eigen::VectorXd vec) { return weight_matrix_ * vec; }
  // Overloaded function for std::vector<Eigen::MatrixXd>
  std::vector<Eigen::MatrixXd> PolynomialRegression::fit_transform(const std::vector<Eigen::MatrixXd>& raw_data) {
      return fit_transform_impl(raw_data);
  }

  // Overloaded function for std::vector<Eigen::VectorXd>
  std::vector<Eigen::VectorXd> PolynomialRegression::fit_transform(const std::vector<Eigen::VectorXd>& raw_data) {
      return fit_transform_impl(raw_data);
  }
  Eigen::VectorXd PolynomialRegression::predict(Eigen::VectorXd vec)
  {
    Eigen::VectorXd vec_ = Eigen::VectorXd(vec.size() - 1);
    vec_ = vec.head(vec.size() - 1);
    vec_ = vec_.array() - vec[vec.size() - 1];
    Eigen::VectorXd prediction = prediction_matrix_ * vec_;
    prediction = prediction.array() + vec[vec.size() - 1];
    return prediction;
  }

///////////////// Polynomial Filter ///////////////////////
PolynomialFilter::PolynomialFilter() {
}
PolynomialFilter::~PolynomialFilter() {
}
void PolynomialFilter::set_params(int degree, int num_samples, double lambda, double minimum_decay)
{
  degree_ = degree;
  num_samples_ = num_samples;
  lambda_ = lambda;
  minimum_decay_ = minimum_decay;
}
double PolynomialFilter::fit_transform(double timestamp, double sample)
{
  timestamps_.push_back(timestamp);
  if (int(timestamps_.size()) > num_samples_) {
    timestamps_.erase(timestamps_.begin());
    samples_.erase(samples_.begin());
  }
  if (int(timestamps_.size()) < num_samples_) {
    samples_.push_back(sample);
    return sample;
  }
  Eigen::VectorXd time_vector = Eigen::VectorXd(num_samples_);
  for (int i = 0; i < num_samples_; i++) {
    time_vector[i] = timestamps_[i] - timestamps_[num_samples_ - 1];
  }
  Eigen::MatrixXd time_matrix = Eigen::MatrixXd::Ones(num_samples_, degree_ + 1);
  for (int i = 1; i < degree_; i++) {
    time_matrix.col(i) = time_matrix.col(i - 1).array() * time_vector.array();
  }
  Eigen::MatrixXd regularized_matrix = lambda_ * Eigen::MatrixXd::Identity(degree_ + 1, degree_ + 1);
  regularized_matrix(0, 0) = 0.0;
  Eigen::VectorXd decay_vector = Eigen::VectorXd::LinSpaced(num_samples_, minimum_decay_, 1.0);
  Eigen::MatrixXd decay_matrix = decay_vector.asDiagonal();
  Eigen::MatrixXd weight_matrix = time_matrix *
                                 (time_matrix.transpose() * decay_matrix * time_matrix + regularized_matrix).inverse() *
                                 time_matrix.transpose() * decay_matrix;
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(num_samples_);
  for (int i = 0; i < num_samples_ - 1; i++) {
    vec[i] = samples_[i];
  }
  vec[num_samples_ - 1] = sample;
  double sample_new = (weight_matrix * vec)[num_samples_ - 1];
  samples_.push_back(sample_new);
  return sample_new;
}
///////////////// SgFilter ///////////////////////

  SgFilter::SgFilter() {}
  SgFilter::~SgFilter() {}
  void SgFilter::set_params(int degree, int window_size)
  {
    degree_ = degree;
    window_size_ = window_size;
  }
  void SgFilter::calc_sg_filter_weight()
  {
    Eigen::VectorXd e_0 = Eigen::VectorXd::Zero(degree_+1);
    e_0[0] = 1.0;
    sg_vector_left_edge_ = std::vector<Eigen::VectorXd>(window_size_);
    sg_vector_right_edge_ = std::vector<Eigen::VectorXd>(window_size_);
    for (int i = 0; i<window_size_; i++){
      Eigen::VectorXd time_vector = Eigen::VectorXd::LinSpaced(window_size_ + i + 1, - i, window_size_);
      Eigen::MatrixXd time_matrix = Eigen::MatrixXd::Ones(window_size_ + i + 1, degree_ + 1);
      for (int j = 1; j < degree_+1; j++) {
        time_matrix.col(j) = time_matrix.col(j - 1).array() * time_vector.array();
      }
      sg_vector_left_edge_[i] = time_matrix * (time_matrix.transpose() * time_matrix).inverse() * e_0;
      time_vector = Eigen::VectorXd::LinSpaced(window_size_ + i + 1, - window_size_, i);
      time_matrix = Eigen::MatrixXd::Ones(window_size_ + i + 1, degree_ + 1);
      for (int j = 1; j < degree_+1; j++) {
        time_matrix.col(j) = time_matrix.col(j - 1).array() * time_vector.array();
      }
      sg_vector_right_edge_[i] = time_matrix * (time_matrix.transpose() * time_matrix).inverse() * e_0;
    }
    Eigen::VectorXd time_vector = Eigen::VectorXd::LinSpaced(2*window_size_ + 1, - window_size_, window_size_);
    Eigen::MatrixXd time_matrix = Eigen::MatrixXd::Ones(2*window_size_ + 1, degree_ + 1);
    for (int j = 1; j < degree_+1; j++) {
      time_matrix.col(j) = time_matrix.col(j - 1).array() * time_vector.array();
    }
    sg_vector_center_ = time_matrix * (time_matrix.transpose() * time_matrix).inverse() * e_0;
  }
  // Overloaded function for std::vector<Eigen::MatrixXd>
  std::vector<Eigen::MatrixXd> SgFilter::sg_filter(const std::vector<Eigen::MatrixXd>& raw_data) {
      return sg_filter_impl(raw_data);
  }

  // Overloaded function for std::vector<Eigen::VectorXd>
  std::vector<Eigen::VectorXd> SgFilter::sg_filter(const std::vector<Eigen::VectorXd>& raw_data) {
      return sg_filter_impl(raw_data);
  }

///////////////// FilterDiffNN ///////////////////////
  FilterDiffNN::FilterDiffNN() {}
  FilterDiffNN::~FilterDiffNN() {}
  void FilterDiffNN::set_sg_filter_params(int degree, int window_size, int state_size, int h_dim, int acc_queue_size, int steer_queue_size, int predict_step, double control_dt)
  {
    state_size_ = state_size;
    h_dim_ = h_dim;
    acc_queue_size_ = acc_queue_size;
    steer_queue_size_ = steer_queue_size;
    predict_step_ = predict_step;
    control_dt_ = control_dt;
    sg_filter_.set_params(degree, window_size);
    sg_filter_.calc_sg_filter_weight();
  }
  void FilterDiffNN::fit_transform_for_NN_diff(
    std::vector<Eigen::MatrixXd> A, std::vector<Eigen::MatrixXd> B, std::vector<Eigen::MatrixXd> C,std::vector<Eigen::MatrixXd> & dF_d_states,
    std::vector<Eigen::MatrixXd> & dF_d_inputs)
  {
    int num_samples = A.size();
    dF_d_states = std::vector<Eigen::MatrixXd>(num_samples);
    dF_d_inputs = std::vector<Eigen::MatrixXd>(num_samples);
    std::vector<Eigen::MatrixXd> dF_d_state_with_input_history;
    dF_d_state_with_input_history = sg_filter_.sg_filter(C);
    for (int i = 0; i < num_samples; i++) {
      Eigen::MatrixXd dF_d_state_temp = Eigen::MatrixXd::Zero(2*h_dim_ + state_size_, 2*h_dim_ + state_size_ + acc_queue_size_ + steer_queue_size_);

      Eigen::MatrixXd dF_d_input_temp = Eigen::MatrixXd::Zero(2*h_dim_ + state_size_, 2);
      Eigen::MatrixXd dF_d_state_with_input_history_tmp = dF_d_state_with_input_history[i];


      dF_d_state_temp.block(2*h_dim_, 2*h_dim_, state_size_, state_size_ + acc_queue_size_ + steer_queue_size_) = A[i];

      dF_d_input_temp.block(2*h_dim_, 0, state_size_, 2) = B[i];

      dF_d_state_temp.leftCols(2*h_dim_ + state_size_ + acc_queue_size_) += dF_d_state_with_input_history_tmp.leftCols(2*h_dim_ + state_size_ + acc_queue_size_);

      dF_d_state_temp.rightCols(steer_queue_size_) += dF_d_state_with_input_history_tmp.middleCols(2*h_dim_ + state_size_ + acc_queue_size_ + predict_step_, steer_queue_size_);
      for (int j=0; j<predict_step_; j++){
        dF_d_state_temp.col(2*h_dim_ + state_size_ + acc_queue_size_ -1) += dF_d_state_with_input_history_tmp.col(2*h_dim_ + state_size_ + acc_queue_size_ + j);
        dF_d_state_temp.col(2*h_dim_ + state_size_ + acc_queue_size_ + steer_queue_size_ -1) += dF_d_state_with_input_history_tmp.col(2*h_dim_ + state_size_ + acc_queue_size_ + predict_step_ + steer_queue_size_ + j);
        dF_d_input_temp.col(0) += (j+1) * dF_d_state_with_input_history_tmp.col(2*h_dim_ + state_size_ + acc_queue_size_ + j)*control_dt_;
        dF_d_input_temp.col(1) += (j+1) * dF_d_state_with_input_history_tmp.col(2*h_dim_ + state_size_ + acc_queue_size_ + predict_step_ + steer_queue_size_ + j)*control_dt_;
      }
      dF_d_states[i] = dF_d_state_temp;
      dF_d_inputs[i] = dF_d_input_temp;
    }
  }
///////////////// ButterworthFilter ///////////////////////

  ButterworthFilter::ButterworthFilter() {
    set_params();
  }
  ButterworthFilter::~ButterworthFilter() {}
  void ButterworthFilter::set_params()
  {
    YAML::Node butterworth_coef_node = YAML::LoadFile(get_param_dir_path() + "/butterworth_coef.yaml");
    order_ = butterworth_coef_node["Butterworth"]["order"].as<int>();
    a_ = butterworth_coef_node["Butterworth"]["a"].as<std::vector<double>>();
    b_ = butterworth_coef_node["Butterworth"]["b"].as<std::vector<double>>();
    initialized_ = false;
  }
  Eigen::VectorXd ButterworthFilter::apply(Eigen::VectorXd input_value){
    if (!initialized_){
      x_ = std::vector<Eigen::VectorXd>(order_, input_value);
      y_ = std::vector<Eigen::VectorXd>(order_, input_value);
      initialized_ = true;
    }
    Eigen::VectorXd output_value = b_[0] * input_value;
    for (int i = 0; i < order_; i++) {
      output_value += b_[order_ - i] * x_[i] - a_[order_ - i] * y_[i];
    }
    x_.erase(x_.begin());
    x_.push_back(input_value);
    y_.erase(y_.begin());
    y_.push_back(output_value);
    return output_value;
  }
///////////////// LinearRegressionCompensation ///////////////////////

  LinearRegressionCompensation::LinearRegressionCompensation() {
    YAML::Node optimization_param_node = YAML::LoadFile(get_param_dir_path() + "/optimization_param.yaml");
    lambda_ = optimization_param_node["optimization_parameter"]["compensation"]["lambda"].as<double>();
    lambda_bias_ = optimization_param_node["optimization_parameter"]["compensation"]["lambda_bias"].as<double>();
    decay_ = optimization_param_node["optimization_parameter"]["compensation"]["decay"].as<double>();
    fit_yaw_ = optimization_param_node["optimization_parameter"]["compensation"]["fit_yaw"].as<bool>();
    use_compensator_ = optimization_param_node["optimization_parameter"]["compensation"]["use_compensator"].as<bool>();
    max_yaw_compensation_ = optimization_param_node["optimization_parameter"]["compensation"]["max_yaw_compensation"].as<double>();
    max_acc_compensation_ = optimization_param_node["optimization_parameter"]["compensation"]["max_acc_compensation"].as<double>();
    max_steer_compensation_ = optimization_param_node["optimization_parameter"]["compensation"]["max_steer_compensation"].as<double>();
    initialize();
  }
  LinearRegressionCompensation::~LinearRegressionCompensation() {}
  void LinearRegressionCompensation::initialize(){
    if (fit_yaw_){
      XXT_ = Eigen::MatrixXd::Zero(6, 6);
      YXT_ = Eigen::MatrixXd::Zero(3, 6);
      regression_matrix_ = Eigen::MatrixXd::Zero(3, 6);
    }
    else{
      XXT_ = Eigen::MatrixXd::Zero(6, 6);
      YXT_ = Eigen::MatrixXd::Zero(2, 6);
      regression_matrix_ = Eigen::MatrixXd::Zero(2, 6);
    }
  }
  void LinearRegressionCompensation::update_input_queue(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat){
    Eigen::VectorXd input = Eigen::VectorXd(6);
    input[0] = 1.0;
    input[1] = states[vel_index_];
    input[2] = states[acc_index_];
    input[3] = states[steer_index_];
    input[4] = acc_input_history_concat.mean();
    input[5] = steer_input_history_concat.mean();
    input_queue_.push_back(input);
  }
  void LinearRegressionCompensation::update_regression_matrix(Eigen::VectorXd error_vector){
    int num_samples = input_queue_.size();
    Eigen::VectorXd error_vector_extracted;
    if (fit_yaw_){
      error_vector_extracted = Eigen::VectorXd::Zero(3);
      error_vector_extracted[0] = error_vector[yaw_index_];
      error_vector_extracted[1] = error_vector[acc_index_];
      error_vector_extracted[2] = error_vector[steer_index_];
    }
    else{
      error_vector_extracted = Eigen::VectorXd::Zero(2);
      error_vector_extracted[0] = error_vector[acc_index_];
      error_vector_extracted[1] = error_vector[steer_index_];
    }
    Eigen::MatrixXd XXT_new = Eigen::MatrixXd::Zero(XXT_.rows(), XXT_.cols());
    Eigen::MatrixXd YXT_new = Eigen::MatrixXd::Zero(YXT_.rows(), YXT_.cols());
    for (int i = 0; i < num_samples; i++){
      XXT_new += input_queue_[i] * input_queue_[i].transpose();
      YXT_new += (error_vector_extracted + regression_matrix_ * input_queue_[i]) * input_queue_[i].transpose();
    }
    XXT_ = decay_ * XXT_ + (1.0 - decay_) / num_samples * XXT_new;
    YXT_ = decay_ * YXT_ + (1.0 - decay_) / num_samples * YXT_new;
    Eigen::MatrixXd regularized_matrix = lambda_ * Eigen::MatrixXd::Identity(XXT_.rows(), XXT_.cols());
    regularized_matrix(0, 0) = lambda_bias_;
    XXT_inv_ = (XXT_ + regularized_matrix).inverse();
    regression_matrix_ = YXT_ * XXT_inv_;
    input_queue_.clear();
  }
  Eigen::VectorXd LinearRegressionCompensation::predict(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat){
    if (!use_compensator_){
      return Eigen::VectorXd::Zero(state_size_);
    }
    Eigen::VectorXd input = Eigen::VectorXd(6);
    input[0] = 1.0;
    input[1] = states[vel_index_];
    input[2] = states[acc_index_];
    input[3] = states[steer_index_];
    input[4] = acc_input_history_concat.mean();
    input[5] = steer_input_history_concat.mean();
    // Eigen::MatrixXd XXT_inv = (XXT_ + lambda_ * Eigen::MatrixXd::Identity(XXT_.rows(), XXT_.cols())).inverse();
    Eigen::VectorXd prediction_extracted = regression_matrix_ * input;
    Eigen::VectorXd prediction = Eigen::VectorXd::Zero(state_size_);
    if (fit_yaw_){
      prediction[yaw_index_] += std::clamp(prediction_extracted[0], -max_yaw_compensation_, max_yaw_compensation_);
      prediction[acc_index_] += std::clamp(prediction_extracted[1], -max_acc_compensation_, max_acc_compensation_);
      prediction[steer_index_] += std::clamp(prediction_extracted[2], -max_steer_compensation_, max_steer_compensation_);
    }
    else{
      prediction[acc_index_] += std::clamp(prediction_extracted[0], -max_acc_compensation_, max_acc_compensation_);
      prediction[steer_index_] += std::clamp(prediction_extracted[1], -max_steer_compensation_, max_steer_compensation_);
    }
    return prediction;
  }
  Eigen::MatrixXd LinearRegressionCompensation::Predict(Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat){
    if (!use_compensator_){
      return Eigen::MatrixXd::Zero(state_size_, States.cols());
    }
    Eigen::MatrixXd Input = Eigen::MatrixXd::Zero(6,States.cols());
    Input.row(0) = Eigen::VectorXd::Ones(States.cols());
    Input.row(1) = States.row(vel_index_);
    Input.row(2) = States.row(acc_index_);
    Input.row(3) = States.row(steer_index_);
    Input.row(4) = Acc_input_history_concat.colwise().mean();
    Input.row(5) = Steer_input_history_concat.colwise().mean();
    Eigen::MatrixXd Prediction = Eigen::MatrixXd::Zero(state_size_, States.cols());
    //Eigen::MatrixXd XXT_inv = (XXT_ + lambda_ * Eigen::MatrixXd::Identity(XXT_.rows(), XXT_.cols())).inverse();
    Eigen::MatrixXd Prediction_extracted = regression_matrix_ * Input;
    if (fit_yaw_){
      Prediction.row(yaw_index_) += Prediction_extracted.row(0).unaryExpr([this](double val) {
            return std::clamp(val, - this->max_yaw_compensation_, this->max_yaw_compensation_);
        });
      Prediction.row(acc_index_) += Prediction_extracted.row(1).unaryExpr([this](double val) {
            return std::clamp(val, - this->max_acc_compensation_, this->max_acc_compensation_);
        });
      Prediction.row(steer_index_) += Prediction_extracted.row(2).unaryExpr([this](double val) {
            return std::clamp(val, - this->max_steer_compensation_, this->max_steer_compensation_);
        });
    }
    else{
      Prediction.row(acc_index_) += Prediction_extracted.row(0).unaryExpr([this](double val) {
            return std::clamp(val, - this->max_acc_compensation_, this->max_acc_compensation_);
        });
      Prediction.row(steer_index_) += Prediction_extracted.row(1).unaryExpr([this](double val) {
            return std::clamp(val, - this->max_steer_compensation_, this->max_steer_compensation_);
        });
    } 
    return Prediction;
  }
  Eigen::VectorXd LinearRegressionCompensation::get_bias(){
    Eigen::VectorXd bias = Eigen::VectorXd::Zero(state_size_);
    if (fit_yaw_){
      bias[yaw_index_] = regression_matrix_(0, 0);
      bias[acc_index_] = regression_matrix_(1, 0);
      bias[steer_index_] = regression_matrix_(2, 0);
    }
    else{
      bias[acc_index_] = regression_matrix_(0, 0);
      bias[steer_index_] = regression_matrix_(1, 0);
    }
    return bias;
  }

///////////////// NominalDynamics ///////////////////////

  NominalDynamics::NominalDynamics() {}
  NominalDynamics::~NominalDynamics() {}
  void NominalDynamics::set_params(
    double wheel_base, double acc_time_delay, double steer_time_delay, double acc_time_constant,
    double steer_time_constant, int acc_queue_size, int steer_queue_size, double control_dt,
    int predict_step)
  {
    wheel_base_ = wheel_base;
    acc_time_delay_ = acc_time_delay;
    steer_time_delay_ = steer_time_delay;
    acc_time_constant_ = acc_time_constant;
    steer_time_constant_ = steer_time_constant;
    acc_queue_size_ = acc_queue_size;
    steer_queue_size_ = steer_queue_size;

    predict_step_ = predict_step;
    acc_input_end_index_ = 6 + acc_queue_size_ - 1;
    steer_input_start_index_ = 6 + acc_queue_size_;
    steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
    control_dt_ = control_dt;
    predict_dt_ = control_dt_ * predict_step_;

    acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
    steer_delay_step_ =
      std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);
  }
  void NominalDynamics::set_steer_dead_band(double steer_dead_band) { steer_dead_band_ = steer_dead_band; }
  Eigen::VectorXd NominalDynamics::F_nominal(Eigen::VectorXd states, Eigen::VectorXd inputs)
  {
    double vel = states[vel_index_];
    double yaw = states[yaw_index_];
    double acc = states[acc_index_];
    double steer = states[steer_index_];
    double steer_diff = inputs[1] - steer;
    if (std::abs(steer_diff) < steer_dead_band_) {
      steer_diff = 0.0;
    }
    else if (steer_diff > 0) {
      steer_diff -= steer_dead_band_;
    }
    else {
      steer_diff += steer_dead_band_;
    }
    Eigen::VectorXd states_dot = Eigen::VectorXd(state_size_);
    states_dot[x_index_] = vel * std::cos(yaw);
    states_dot[y_index_] = vel * std::sin(yaw);
    states_dot[vel_index_] = acc;
    states_dot[yaw_index_] = vel * std::tan(steer) / wheel_base_;
    states_dot[acc_index_] = (inputs[0] - acc) / acc_time_constant_;
    states_dot[steer_index_] = steer_diff / steer_time_constant_;
    return states + states_dot * control_dt_;
  }
  Eigen::MatrixXd NominalDynamics::F_nominal_with_diff(Eigen::VectorXd states, Eigen::VectorXd inputs)
  {
    Eigen::VectorXd state_next = F_nominal(states, inputs);
    double vel = states[vel_index_];
    double yaw = states[yaw_index_];
    double steer = states[steer_index_];
    Eigen::MatrixXd dF_d_states = Eigen::MatrixXd::Identity(state_size_, state_size_);
    Eigen::MatrixXd dF_d_inputs = Eigen::MatrixXd::Zero(state_size_, inputs.size());
    dF_d_states(x_index_, vel_index_) = std::cos(yaw) * control_dt_;
    dF_d_states(x_index_, yaw_index_) = -vel * std::sin(yaw) * control_dt_;
    dF_d_states(y_index_, vel_index_) = std::sin(yaw) * control_dt_;
    dF_d_states(y_index_, yaw_index_) = vel * std::cos(yaw) * control_dt_;
    dF_d_states(vel_index_, acc_index_) = control_dt_;
    dF_d_states(yaw_index_, vel_index_) = std::tan(steer) / wheel_base_ * control_dt_;
    dF_d_states(yaw_index_, steer_index_) =
      vel / (wheel_base_ * std::cos(steer) * std::cos(steer)) * control_dt_;
    dF_d_states(acc_index_, acc_index_) -= 1 / acc_time_constant_ * control_dt_;
    dF_d_states(steer_index_, steer_index_) -= 1 / steer_time_constant_ * control_dt_;
    dF_d_inputs(acc_index_, 0) = 1 / acc_time_constant_ * control_dt_;
    dF_d_inputs(steer_index_, 1) = 1 / steer_time_constant_ * control_dt_;
    Eigen::MatrixXd result = Eigen::MatrixXd(state_size_, 1 + state_size_ + inputs.size());
    result.col(0) = state_next;
    result.block(0, 1, state_size_, state_size_) = dF_d_states;
    result.block(0, 1 + state_size_, state_size_, inputs.size()) = dF_d_inputs;
    return result;
  }
  Eigen::VectorXd NominalDynamics::F_nominal_predict(Eigen::VectorXd states, Eigen::VectorXd Inputs)
  {
    Eigen::VectorXd states_predict = states;
    for (int i = 0; i < predict_step_; i++) {
      states_predict = F_nominal(states_predict, Inputs.segment(2 * i, 2));
    }
    return states_predict;
  }
  Eigen::VectorXd NominalDynamics::F_nominal_predict_with_diff(
    Eigen::VectorXd states, Eigen::VectorXd Inputs, Eigen::MatrixXd & dF_d_states,
    Eigen::MatrixXd & dF_d_inputs)
  {
    Eigen::VectorXd states_predict = states;
    dF_d_states = Eigen::MatrixXd::Identity(state_size_, state_size_);
    dF_d_inputs = Eigen::MatrixXd::Zero(state_size_, predict_step_ * 2);

    for (int i = 0; i < predict_step_; i++) {
      Eigen::MatrixXd dF_d_states_temp;
      Eigen::MatrixXd dF_d_inputs_temp;
      Eigen::MatrixXd F_nominal_with_diff_temp =
        F_nominal_with_diff(states_predict, Inputs.segment(2 * i, 2));
      dF_d_states_temp = F_nominal_with_diff_temp.block(0, 1, state_size_, state_size_);
      dF_d_inputs_temp = F_nominal_with_diff_temp.block(0, 1 + state_size_, state_size_, 2);
      dF_d_states = dF_d_states_temp * dF_d_states;
      if (i > 0) {
        dF_d_inputs.block(0, 0, state_size_, 2 * i) =
          dF_d_states_temp * dF_d_inputs.block(0, 0, state_size_, 2 * i);
      }
      dF_d_inputs.block(0, 2 * i, state_size_, 2) = dF_d_inputs_temp;
      states_predict = F_nominal_with_diff_temp.col(0);
    }
    return states_predict;
  }
  Eigen::MatrixXd NominalDynamics::F_nominal_for_candidates(Eigen::MatrixXd States, Eigen::MatrixXd Inputs)
  {
    Eigen::MatrixXd States_dot = Eigen::MatrixXd(States.rows(), States.cols());
    States_dot.row(x_index_) =
      States.row(vel_index_).array() * States.row(yaw_index_).array().cos();
    States_dot.row(y_index_) =
      States.row(vel_index_).array() * States.row(yaw_index_).array().sin();
    States_dot.row(vel_index_) = States.row(acc_index_);
    States_dot.row(yaw_index_) =
      States.row(vel_index_).array() * States.row(steer_index_).array().tan() / wheel_base_;
    States_dot.row(acc_index_) =
      (Inputs.row(0).array() - States.row(acc_index_).array()) / acc_time_constant_;
    States_dot.row(steer_index_) =
      (Inputs.row(1).array() - States.row(steer_index_).array()) / steer_time_constant_;
    return States + States_dot * control_dt_;
  }
  Eigen::MatrixXd NominalDynamics::F_nominal_predict_for_candidates(Eigen::MatrixXd States, Eigen::MatrixXd Inputs)
  {
    Eigen::MatrixXd States_predict = States;
    for (int i = 0; i < predict_step_; i++) {
      States_predict = F_nominal_for_candidates(States_predict, Inputs.middleRows(2 * i, 2));
    }
    return States_predict;
  }
  Eigen::VectorXd NominalDynamics::F_with_input_history(
    const Eigen::VectorXd states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat)
  {
    Eigen::VectorXd Inputs = Eigen::VectorXd(2 * predict_step_);
    for (int i = 0; i < predict_step_; i++) {
      Inputs[2 * i] = acc_input_history_concat[acc_queue_size_ + i - acc_delay_step_];
      Inputs[2 * i + 1] = steer_input_history_concat[steer_queue_size_ + i - steer_delay_step_];
    }
    return F_nominal_predict(states, Inputs);
  }
  Eigen::VectorXd NominalDynamics::F_with_input_history(
    const Eigen::VectorXd states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat, const Eigen::Vector2d & d_inputs)
  {
    Eigen::VectorXd Inputs = Eigen::VectorXd(2 * predict_step_);
    acc_input_history_concat = Eigen::VectorXd(acc_queue_size_ + predict_step_);
    steer_input_history_concat = Eigen::VectorXd(steer_queue_size_ + predict_step_);
    acc_input_history_concat.head(acc_queue_size_) = acc_input_history;
    steer_input_history_concat.head(steer_queue_size_) = steer_input_history;
    for (int i = 0; i < predict_step_; i++) {
      acc_input_history_concat[acc_queue_size_ + i] =
        acc_input_history_concat[acc_queue_size_ + i - 1] + d_inputs[0] * control_dt_;
      steer_input_history_concat[steer_queue_size_ + i] =
        steer_input_history_concat[steer_queue_size_ + i - 1] + d_inputs[1] * control_dt_;
      Inputs[2 * i] = acc_input_history_concat[acc_queue_size_ + i - acc_delay_step_];
      Inputs[2 * i + 1] = steer_input_history_concat[steer_queue_size_ + i - steer_delay_step_];
    }
    acc_input_history = acc_input_history_concat.tail(acc_queue_size_);
    steer_input_history = steer_input_history_concat.tail(steer_queue_size_);
    return F_nominal_predict(states, Inputs);
  }
  Eigen::VectorXd NominalDynamics::F_with_input_history_and_diff(
    const Eigen::VectorXd states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat, const Eigen::Vector2d & d_inputs,
    Eigen::MatrixXd & A, Eigen::MatrixXd & B)
  {
    A = Eigen::MatrixXd::Zero(state_size_, state_size_ + acc_queue_size_ + steer_queue_size_);
    B = Eigen::MatrixXd::Zero(state_size_, 2);
    Eigen::VectorXd Inputs = Eigen::VectorXd(2 * predict_step_);
    acc_input_history_concat = Eigen::VectorXd(acc_queue_size_ + predict_step_);
    steer_input_history_concat = Eigen::VectorXd(steer_queue_size_ + predict_step_);
    acc_input_history_concat.head(acc_queue_size_) = acc_input_history;
    steer_input_history_concat.head(steer_queue_size_) = steer_input_history;
    Eigen::MatrixXd dF_d_states, dF_d_inputs;
    for (int i = 0; i < predict_step_; i++) {
      acc_input_history_concat[acc_queue_size_ + i] =
        acc_input_history_concat[acc_queue_size_ + i - 1] + d_inputs[0] * control_dt_;
      steer_input_history_concat[steer_queue_size_ + i] =
        steer_input_history_concat[steer_queue_size_ + i - 1] + d_inputs[1] * control_dt_;
      Inputs[2 * i] = acc_input_history_concat[acc_queue_size_ + i - acc_delay_step_];
      Inputs[2 * i + 1] = steer_input_history_concat[steer_queue_size_ + i - steer_delay_step_];
    }
    acc_input_history = acc_input_history_concat.tail(acc_queue_size_);
    steer_input_history = steer_input_history_concat.tail(steer_queue_size_);
    Eigen::VectorXd states_predict =
      F_nominal_predict_with_diff(states, Inputs, dF_d_states, dF_d_inputs);
    A.block(0, 0, state_size_, state_size_) = dF_d_states;
    for (int i = 0; i < predict_step_; i++) {
      A.col(state_size_ + acc_queue_size_ - std::max(acc_delay_step_-i,1)) +=
        dF_d_inputs.col(2 * i);
      A.col(
        state_size_ + acc_queue_size_ + steer_queue_size_ - std::max(steer_delay_step_ - i, 1)) +=
        dF_d_inputs.col(2 * i + 1); 
      if (i - acc_delay_step_ >= 0) {
        B.col(0) += (i - acc_delay_step_ + 1) * dF_d_inputs.col(2 * i) * control_dt_;
      }
      if (i - steer_delay_step_ >= 0) {
        B.col(1) += (i - steer_delay_step_ + 1) * dF_d_inputs.col(2 * i + 1) * control_dt_;
      }
    }
    return states_predict;
  }
  Eigen::MatrixXd NominalDynamics::F_with_input_history_for_candidates(
    const Eigen::MatrixXd & States, Eigen::MatrixXd & Acc_input_history,
    Eigen::MatrixXd & Steer_input_history, Eigen::MatrixXd & Acc_input_history_concat,
    Eigen::MatrixXd & Steer_input_history_concat, const Eigen::MatrixXd & D_inputs)
  {
    Acc_input_history_concat = Eigen::MatrixXd(acc_queue_size_ + predict_step_, States.cols());
    Steer_input_history_concat = Eigen::MatrixXd(steer_queue_size_ + predict_step_, States.cols());
    Acc_input_history_concat.topRows(acc_queue_size_) = Acc_input_history;
    Steer_input_history_concat.topRows(steer_queue_size_) = Steer_input_history;
    
    Eigen::MatrixXd Inputs = Eigen::MatrixXd(2 * predict_step_, States.cols());
   
    for (int i = 0; i < predict_step_; i++) {
      Acc_input_history_concat.row(acc_queue_size_ + i) =
        Acc_input_history_concat.row(acc_queue_size_ + i - 1) + D_inputs.row(0) * control_dt_;
  
      Steer_input_history_concat.row(steer_queue_size_ + i) =
        Steer_input_history_concat.row(steer_queue_size_ + i - 1) +
        D_inputs.row(1) * control_dt_;
    
      Inputs.row(2 * i) = Acc_input_history_concat.row(acc_queue_size_ + i - acc_delay_step_);
     
      Inputs.row(2 * i + 1) =
        Steer_input_history_concat.row(steer_queue_size_ + i - steer_delay_step_);
    }

    Acc_input_history = Acc_input_history_concat.bottomRows(acc_queue_size_);
    Steer_input_history = Steer_input_history_concat.bottomRows(steer_queue_size_);
    return F_nominal_predict_for_candidates(States, Inputs);
  }

///////////////// TransformModelToEigen ///////////////////////

  TransformModelToEigen::TransformModelToEigen() {}
  TransformModelToEigen::~TransformModelToEigen() {}
  void TransformModelToEigen::set_params(
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    double vel_scaling, double vel_bias)
  {
    weight_acc_layer_1_.push_back(weight_acc_layer_1);
    weight_steer_layer_1_.push_back(weight_steer_layer_1);
    weight_acc_layer_2_.push_back(weight_acc_layer_2);
    weight_steer_layer_2_.push_back(weight_steer_layer_2);
    weight_lstm_ih_.push_back(weight_lstm_ih);
    weight_lstm_hh_.push_back(weight_lstm_hh);
    weight_complimentary_layer_.push_back(weight_complimentary_layer);
    weight_linear_relu_.push_back(weight_linear_relu);
    weight_final_layer_.push_back(weight_final_layer);
    bias_acc_layer_1_.push_back(bias_acc_layer_1);
    bias_steer_layer_1_.push_back(bias_steer_layer_1);
    bias_acc_layer_2_.push_back(bias_acc_layer_2);
    bias_steer_layer_2_.push_back(bias_steer_layer_2);
    bias_lstm_ih_.push_back(bias_lstm_ih);
    bias_lstm_hh_.push_back(bias_lstm_hh);
    bias_complimentary_layer_.push_back(bias_complimentary_layer);
    bias_linear_relu_.push_back(bias_linear_relu);
    bias_final_layer_.push_back(bias_final_layer);
    vel_scaling_.push_back(vel_scaling);
    vel_bias_.push_back(vel_bias);
    h_dim_ = weight_lstm_hh.cols();
    model_num_ += 1;
  }
  void TransformModelToEigen::clear_params()
  {
    weight_acc_layer_1_.clear();
    weight_steer_layer_1_.clear();
    weight_acc_layer_2_.clear();
    weight_steer_layer_2_.clear();
    weight_lstm_ih_.clear();
    weight_lstm_hh_.clear();
    weight_complimentary_layer_.clear();
    weight_linear_relu_.clear();
    weight_final_layer_.clear();
    bias_acc_layer_1_.clear();
    bias_steer_layer_1_.clear();
    bias_acc_layer_2_.clear();
    bias_steer_layer_2_.clear();
    bias_lstm_ih_.clear();
    bias_lstm_hh_.clear();
    bias_complimentary_layer_.clear();
    bias_linear_relu_.clear();
    bias_final_layer_.clear();
    vel_scaling_.clear();
    vel_bias_.clear();
    model_num_ = 0;
  }
  void TransformModelToEigen::update_lstm(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
    Eigen::VectorXd & h_lstm_next, Eigen::VectorXd & c_lstm_next,
    std::vector<Eigen::VectorXd> & u_acc_layer_1, std::vector<Eigen::VectorXd> & u_steer_layer_1,
    std::vector<Eigen::VectorXd> & u_acc_layer_2, std::vector<Eigen::VectorXd> & u_steer_layer_2,
    std::vector<Eigen::VectorXd> & u_i_lstm, std::vector<Eigen::VectorXd> & u_f_lstm, std::vector<Eigen::VectorXd> & u_g_lstm,
    std::vector<Eigen::VectorXd> & u_o_lstm, std::vector<Eigen::VectorXd> & i_lstm, std::vector<Eigen::VectorXd> & f_lstm,
    std::vector<Eigen::VectorXd> & g_lstm, std::vector<Eigen::VectorXd> & o_lstm,
    std::vector<Eigen::VectorXd> & u_complimentary_layer, std::vector<Eigen::VectorXd> & h_complimentary_layer)
  {
    h_lstm_next = Eigen::VectorXd::Zero(h_dim_);
    c_lstm_next = Eigen::VectorXd::Zero(h_dim_);
    u_acc_layer_1.assign(model_num_, Eigen::VectorXd());
    u_steer_layer_1.assign(model_num_, Eigen::VectorXd());
    u_acc_layer_2.assign(model_num_, Eigen::VectorXd());
    u_steer_layer_2.assign(model_num_, Eigen::VectorXd());
    u_i_lstm.assign(model_num_, Eigen::VectorXd());
    u_f_lstm.assign(model_num_, Eigen::VectorXd());
    u_g_lstm.assign(model_num_, Eigen::VectorXd());
    u_o_lstm.assign(model_num_, Eigen::VectorXd());
    i_lstm.assign(model_num_, Eigen::VectorXd());
    f_lstm.assign(model_num_, Eigen::VectorXd());
    g_lstm.assign(model_num_, Eigen::VectorXd());
    o_lstm.assign(model_num_, Eigen::VectorXd());
    u_complimentary_layer.assign(model_num_, Eigen::VectorXd());
    h_complimentary_layer.assign(model_num_, Eigen::VectorXd());
    for (int i = 0; i < model_num_; i++){
      Eigen::VectorXd acc_sub(acc_queue_size_ + predict_step_ + 2);
      Eigen::VectorXd steer_sub(steer_queue_size_ + predict_step_ + 2);
      acc_sub << vel_scaling_[i] * (x[vel_index_] - vel_bias_[i]), x[acc_index_], x.segment(acc_input_start_index_, acc_queue_size_ + predict_step_);
      steer_sub << vel_scaling_[i] * (x[vel_index_] - vel_bias_[i]), x[steer_index_],
        x.segment(steer_input_start_index_, steer_queue_size_ + predict_step_);

      u_acc_layer_1[i] = weight_acc_layer_1_[i] * acc_sub + bias_acc_layer_1_[i];
      u_steer_layer_1[i] = weight_steer_layer_1_[i] * steer_sub + bias_steer_layer_1_[i];

      const Eigen::VectorXd acc_layer_1 = relu(u_acc_layer_1[i]);
      const Eigen::VectorXd steer_layer_1 = relu(u_steer_layer_1[i]);
      u_acc_layer_2[i] = weight_acc_layer_2_[i] * acc_layer_1 + bias_acc_layer_2_[i];
      u_steer_layer_2[i] = weight_steer_layer_2_[i] * steer_layer_1 + bias_steer_layer_2_[i];

      const Eigen::VectorXd acc_layer_2 = relu(u_acc_layer_2[i]);
      const Eigen::VectorXd steer_layer_2 = relu(u_steer_layer_2[i]);

      Eigen::VectorXd h1 = Eigen::VectorXd(1 + acc_layer_2.size() + steer_layer_2.size());
      h1 << vel_scaling_[i] * (x[vel_index_] - vel_bias_[i]), acc_layer_2, steer_layer_2;

      u_i_lstm[i] = weight_lstm_ih_[i].block(0, 0, h_dim_, h1.size()) * h1 +
                  bias_lstm_ih_[i].segment(0, h_dim_) +
                  weight_lstm_hh_[i].block(0, 0, h_dim_, h_dim_) * h_lstm +
                  bias_lstm_hh_[i].segment(0, h_dim_);

      u_f_lstm[i] =
        weight_lstm_ih_[i].block(1 * h_dim_, 0, h_dim_, h1.size()) * h1 +
        bias_lstm_ih_[i].segment(h_dim_, h_dim_) +
        weight_lstm_hh_[i].block(1 * h_dim_, 0, h_dim_, h_dim_) * h_lstm +
        bias_lstm_hh_[i].segment(h_dim_, h_dim_);
      u_g_lstm[i] =
        weight_lstm_ih_[i].block(2 * h_dim_, 0, h_dim_, h1.size()) * h1 +
        bias_lstm_ih_[i].segment(2 * h_dim_, h_dim_) +
        weight_lstm_hh_[i].block(2 * h_dim_, 0, h_dim_, h_dim_) * h_lstm +
        bias_lstm_hh_[i].segment(2 * h_dim_, h_dim_);
      u_o_lstm[i] =
        weight_lstm_ih_[i].block(3 * h_dim_, 0, h_dim_, h1.size()) * h1 +
        bias_lstm_ih_[i].segment(3 * h_dim_, h_dim_) +
        weight_lstm_hh_[i].block(3 * h_dim_, 0, h_dim_, h_dim_) * h_lstm +
        bias_lstm_hh_[i].segment(3 * h_dim_, h_dim_);

      i_lstm[i] = sigmoid(u_i_lstm[i]);
      f_lstm[i] = sigmoid(u_f_lstm[i]);
      g_lstm[i] = tanh(u_g_lstm[i]);
      o_lstm[i] = sigmoid(u_o_lstm[i]);

      Eigen::VectorXd tmp_c_lstm_next = (f_lstm[i].array() * c_lstm.array() + i_lstm[i].array() * g_lstm[i].array());
      c_lstm_next += tmp_c_lstm_next / model_num_;
      h_lstm_next += (o_lstm[i].array() * tanh(tmp_c_lstm_next).array() / model_num_).matrix();
      u_complimentary_layer[i] = weight_complimentary_layer_[i] * h1 + bias_complimentary_layer_[i];
      h_complimentary_layer[i] = relu(u_complimentary_layer[i]);
    }
  }
  void TransformModelToEigen::error_prediction(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
    Eigen::VectorXd & h_lstm_next, Eigen::VectorXd & c_lstm_next,
    std::vector<Eigen::VectorXd> & u_acc_layer_1, std::vector<Eigen::VectorXd> & u_steer_layer_1,
    std::vector<Eigen::VectorXd> & u_acc_layer_2, std::vector<Eigen::VectorXd> & u_steer_layer_2,
    std::vector<Eigen::VectorXd> & u_i_lstm, std::vector<Eigen::VectorXd> & u_f_lstm, std::vector<Eigen::VectorXd> & u_g_lstm,
    std::vector<Eigen::VectorXd> & u_o_lstm, std::vector<Eigen::VectorXd> & i_lstm, std::vector<Eigen::VectorXd> & f_lstm,
    std::vector<Eigen::VectorXd> & g_lstm, std::vector<Eigen::VectorXd> & o_lstm, std::vector<Eigen::VectorXd> & u_complimentary_layer,
    std::vector<Eigen::VectorXd> & h_complimentary_layer,
    std::vector<Eigen::VectorXd> & u_linear_relu, Eigen::VectorXd & y)
  {
    update_lstm(
      x, h_lstm, c_lstm, h_lstm_next, c_lstm_next, u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2,
      u_i_lstm, u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm, u_complimentary_layer, h_complimentary_layer);
    u_linear_relu.assign(model_num_, Eigen::VectorXd());
    y = Eigen::VectorXd::Zero(bias_final_layer_[0].size());
    for (int i = 0; i < model_num_; i++){
      Eigen::VectorXd h_lstm_output = Eigen::VectorXd(h_dim_ + h_complimentary_layer[i].size());
      h_lstm_output << h_lstm_next, h_complimentary_layer[i];
      u_linear_relu[i] = weight_linear_relu_[i] * h_lstm_output + bias_linear_relu_[i];
      Eigen::VectorXd linear_relu = relu(u_linear_relu[i]);
      y += (weight_final_layer_[i] * linear_relu + bias_final_layer_[i]) / model_num_;
    }

  }
  void TransformModelToEigen::error_prediction(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
    Eigen::VectorXd & h_lstm_next, Eigen::VectorXd & c_lstm_next, Eigen::VectorXd & y)
  {
    std::vector<Eigen::VectorXd> u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2, 
      u_i_lstm, u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm,
      u_complimentary_layer, h_complimentary_layer, u_linear_relu;
    error_prediction(
      x, h_lstm, c_lstm, h_lstm_next, c_lstm_next,
      u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2, u_i_lstm,
      u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm,
      u_complimentary_layer, h_complimentary_layer, u_linear_relu, y);
  }
  void TransformModelToEigen::error_prediction_with_diff(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
    Eigen::VectorXd & h_lstm_next, Eigen::VectorXd & c_lstm_next, Eigen::VectorXd & y,
    Eigen::MatrixXd & dy_dx, Eigen::MatrixXd & dy_dhc, Eigen::MatrixXd & dhc_dhc,
    Eigen::MatrixXd & dhc_dx)
  {
    std::vector<Eigen::VectorXd> u_acc_layer_1, u_steer_layer_1, u_acc_layer_2,
      u_steer_layer_2, u_i_lstm, u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm,
      o_lstm, u_complimentary_layer, h_complimentary_layer, u_linear_relu;
    error_prediction(
      x, h_lstm, c_lstm, h_lstm_next, c_lstm_next,
      u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2, u_i_lstm,
      u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm,
      u_complimentary_layer, h_complimentary_layer, u_linear_relu, y);

    dy_dx = Eigen::MatrixXd::Zero(y.size(), x.size());
    dy_dhc = Eigen::MatrixXd::Zero(y.size(), 2 * h_dim_);
    dhc_dhc = Eigen::MatrixXd::Zero(2 * h_dim_, 2 * h_dim_);
    dhc_dx = Eigen::MatrixXd::Zero(2 * h_dim_, x.size());

    for (int i = 0; i< model_num_; i++){
      Eigen::MatrixXd dy_d_linear_relu = weight_final_layer_[i];
      Eigen::MatrixXd dy_d_lstm_output =
        d_relu_product(dy_d_linear_relu, u_linear_relu[i]) * weight_linear_relu_[i];

      Eigen::MatrixXd dy_d_lstm =
        dy_d_lstm_output.block(0, 0, dy_d_lstm_output.rows(), h_dim_);
      Eigen::MatrixXd dy_d_complimentary_layer = dy_d_lstm_output.block(
        0, h_dim_, dy_d_lstm_output.rows(), bias_complimentary_layer_[i].size());

      Eigen::MatrixXd dy_d_o_lstm = dy_d_lstm * tanh(c_lstm_next).asDiagonal();


      Eigen::MatrixXd dy_dc_lstm_next = d_tanh_product(dy_d_lstm * o_lstm[i].asDiagonal(), c_lstm_next);

      int h1_size = 1 + u_acc_layer_2[i].size() + u_steer_layer_2[i].size();
      Eigen::MatrixXd dy_dh1 =
        d_sigmoid_product(dy_d_o_lstm, u_o_lstm[i]) *
        weight_lstm_ih_[i].block(3 * h_dim_, 0, h_dim_, h1_size);

      dy_dh1 += d_sigmoid_product(dy_dc_lstm_next * c_lstm_next.asDiagonal(), u_f_lstm[i]) *
                weight_lstm_ih_[i].block(1 * h_dim_, 0, h_dim_, h1_size);

      dy_dh1 += d_tanh_product(dy_dc_lstm_next * i_lstm[i].asDiagonal(), u_g_lstm[i]) *
                weight_lstm_ih_[i].block(2 * h_dim_, 0, h_dim_, h1_size);
      dy_dh1 += d_sigmoid_product(dy_dc_lstm_next * g_lstm[i].asDiagonal(), u_i_lstm[i]) *
                weight_lstm_ih_[i].block(0, 0, h_dim_, h1_size);

      dy_dh1 += d_relu_product(dy_d_complimentary_layer, u_complimentary_layer[i]) *
                weight_complimentary_layer_[i];
      Eigen::MatrixXd dy_d_acc_layer_1 =
        d_relu_product(dy_dh1.block(0, 1, y.size(), u_acc_layer_2[i].size()), u_acc_layer_2[i]) *
        weight_acc_layer_2_[i];
      Eigen::MatrixXd dy_d_steer_layer_1 =
        d_relu_product(
          dy_dh1.block(0, 1 + u_acc_layer_2[i].size(), y.size(), u_steer_layer_2[i].size()),
          u_steer_layer_2[i]) *
        weight_steer_layer_2_[i];

      Eigen::MatrixXd dy_d_acc_sub =
        d_relu_product(dy_d_acc_layer_1, u_acc_layer_1[i]) * weight_acc_layer_1_[i];
      Eigen::MatrixXd dy_d_steer_sub =
        d_relu_product(dy_d_steer_layer_1, u_steer_layer_1[i]) * weight_steer_layer_1_[i];

      dy_dx.col(vel_index_) += vel_scaling_[i] * (dy_dh1.col(0) + dy_d_acc_sub.col(0) + dy_d_steer_sub.col(0)) / model_num_;
      dy_dx.col(acc_index_) += dy_d_acc_sub.col(1) / model_num_;
      dy_dx.col(steer_index_) += dy_d_steer_sub.col(1) / model_num_;
      dy_dx.block(0, acc_input_start_index_, y.size(), acc_queue_size_ + predict_step_) +=
        dy_d_acc_sub.block(0, 2, y.size(), acc_queue_size_ + predict_step_) / model_num_;
      dy_dx.block(0, steer_input_start_index_, y.size(), steer_queue_size_ + predict_step_) +=
        dy_d_steer_sub.block(0, 2, y.size(), steer_queue_size_ + predict_step_) / model_num_;

      // calc dy_dhc,  dhc_dhc, dhc_dx
      const Eigen::VectorXd dc_du_f = d_sigmoid_product_vec(c_lstm, u_f_lstm[i]);
      const Eigen::VectorXd dc_du_g = d_tanh_product_vec(i_lstm[i], u_g_lstm[i]);
      const Eigen::VectorXd dc_du_i = d_sigmoid_product_vec(g_lstm[i], u_i_lstm[i]);
      const Eigen::VectorXd dh_dc_next = d_tanh_product_vec(o_lstm[i], c_lstm_next);
      const Eigen::VectorXd dh_du_o = d_sigmoid_product_vec(tanh(c_lstm_next), u_o_lstm[i]);

      const Eigen::MatrixXd dc_dc = f_lstm[i].asDiagonal();
      const Eigen::MatrixXd dy_dc = dy_dc_lstm_next * dc_dc;

      Eigen::MatrixXd dc_dh =
        dc_du_f.asDiagonal() *
        weight_lstm_hh_[i].block(h_dim_, 0, h_dim_, h_dim_);
      dc_dh += dc_du_g.asDiagonal() *
              weight_lstm_hh_[i].block(2 * h_dim_, 0, h_dim_, h_dim_);
      dc_dh += dc_du_i.asDiagonal() * weight_lstm_hh_[i].block(0, 0, h_dim_, h_dim_);
      const Eigen::VectorXd dh_dc = dh_dc_next.array() * f_lstm[i].array();

      Eigen::MatrixXd dh_dh =
        dh_du_o.asDiagonal() *
        weight_lstm_hh_[i].block(3 * h_dim_, 0, h_dim_, h_dim_);

      dh_dh += dh_dc_next.asDiagonal() * dc_dh;

      const Eigen::MatrixXd dy_dh = dy_d_lstm * dh_dh;
      Eigen::MatrixXd dc_dh1 =
        dc_du_f.asDiagonal() * weight_lstm_ih_[i].block(h_dim_, 0, h_dim_, h1_size);
      dc_dh1 += dc_du_g.asDiagonal() *
                weight_lstm_ih_[i].block(2 * h_dim_, 0, h_dim_, h1_size);
      dc_dh1 += dc_du_i.asDiagonal() * weight_lstm_ih_[i].block(0, 0, h_dim_, h1_size);

      Eigen::MatrixXd dh_dh1 =
        dh_du_o.asDiagonal() *
        weight_lstm_ih_[i].block(3 * h_dim_, 0, h_dim_, h1_size);
      dh_dh1 += dh_dc_next.asDiagonal() * dc_dh1;

      Eigen::MatrixXd dh_d_acc_layer_1 =
        d_relu_product(dh_dh1.block(0, 1, h_dim_, u_acc_layer_2[i].size()), u_acc_layer_2[i]) *
        weight_acc_layer_2_[i];
      Eigen::MatrixXd dh_d_steer_layer_1 =
        d_relu_product(
          dh_dh1.block(0, 1 + u_acc_layer_2[i].size(), h_dim_, u_steer_layer_2[i].size()),
          u_steer_layer_2[i]) *
        weight_steer_layer_2_[i];
      Eigen::MatrixXd dh_d_acc_sub =
        d_relu_product(dh_d_acc_layer_1, u_acc_layer_1[i]) * weight_acc_layer_1_[i];
      Eigen::MatrixXd dh_d_steer_sub =
        d_relu_product(dh_d_steer_layer_1, u_steer_layer_1[i]) * weight_steer_layer_1_[i];

      Eigen::MatrixXd dh_dx = Eigen::MatrixXd::Zero(h_dim_, x.size());
      dh_dx.col(vel_index_) += vel_scaling_[i] * (dh_dh1.col(0) + dh_d_acc_sub.col(0) + dh_d_steer_sub.col(0));
      dh_dx.col(acc_index_) += dh_d_acc_sub.col(1);
      dh_dx.col(steer_index_) += dh_d_steer_sub.col(1);
      dh_dx.block(0, acc_input_start_index_, h_dim_, acc_queue_size_ + predict_step_) +=
        dh_d_acc_sub.block(0, 2, h_dim_, acc_queue_size_ + predict_step_);
      dh_dx.block(0, steer_input_start_index_, h_dim_, steer_queue_size_ + predict_step_) +=
        dh_d_steer_sub.block(0, 2, h_dim_, steer_queue_size_ + predict_step_);

      Eigen::MatrixXd dc_d_acc_layer_1 =
        d_relu_product(dc_dh1.block(0, 1, h_dim_, u_acc_layer_2[i].size()), u_acc_layer_2[i]) *
        weight_acc_layer_2_[i];
      Eigen::MatrixXd dc_d_steer_layer_1 =
        d_relu_product(
          dc_dh1.block(0, 1 + u_acc_layer_2[i].size(), h_dim_, u_steer_layer_2[i].size()),
          u_steer_layer_2[i]) *
        weight_steer_layer_2_[i];
      Eigen::MatrixXd dc_d_acc_sub =
        d_relu_product(dc_d_acc_layer_1, u_acc_layer_1[i]) * weight_acc_layer_1_[i];
      Eigen::MatrixXd dc_d_steer_sub =
        d_relu_product(dc_d_steer_layer_1, u_steer_layer_1[i]) * weight_steer_layer_1_[i];

      Eigen::MatrixXd dc_dx = Eigen::MatrixXd::Zero(h_dim_, x.size());
      dc_dx.col(vel_index_) += vel_scaling_[i] * (dc_dh1.col(0) + dc_d_acc_sub.col(0) + dc_d_steer_sub.col(0));
      dc_dx.col(acc_index_) += dc_d_acc_sub.col(1);
      dc_dx.col(steer_index_) += dc_d_steer_sub.col(1);
      dc_dx.block(0, acc_input_start_index_, h_dim_, acc_queue_size_ + predict_step_) +=
        dc_d_acc_sub.block(0, 2, h_dim_, acc_queue_size_ + predict_step_);
      dc_dx.block(0, steer_input_start_index_, h_dim_, steer_queue_size_ + predict_step_) +=
        dc_d_steer_sub.block(0, 2, h_dim_, steer_queue_size_ + predict_step_);

      dy_dhc.block(0, 0, y.size(), h_dim_) += dy_dh / model_num_;
      dy_dhc.block(0, h_dim_, y.size(), h_dim_) += dy_dc / model_num_;
      dhc_dhc.block(0, 0, h_dim_, h_dim_) += dh_dh / model_num_;
      dhc_dhc.block(h_dim_, 0, h_dim_, h_dim_) += dc_dh / model_num_;
      dhc_dhc.block(0, h_dim_, h_dim_, h_dim_) += (dh_dc / model_num_).asDiagonal();
      dhc_dhc.block(h_dim_, h_dim_, h_dim_, h_dim_) += dc_dc / model_num_;
      dhc_dx.block(0, 0, h_dim_, x.size()) += dh_dx / model_num_;
      dhc_dx.block(h_dim_, 0, h_dim_, x.size()) += dc_dx / model_num_;
    }
  }
///////////////// TrainedDynamics ///////////////////////
  TrainedDynamics::TrainedDynamics() {
    YAML::Node optimization_param_node = YAML::LoadFile(get_param_dir_path() + "/optimization_param.yaml");
    minimum_steer_diff_ = optimization_param_node["optimization_parameter"]["steer_diff"]["minimum_steer_diff"].as<double>();

    YAML::Node trained_model_param_node = YAML::LoadFile(get_param_dir_path() + "/trained_model_param.yaml");
    bool add_position_to_prediction = trained_model_param_node["trained_model_parameter"]["setting"]["add_position_to_prediction"].as<bool>();

    bool add_vel_to_prediction = trained_model_param_node["trained_model_parameter"]["setting"]["add_vel_to_prediction"].as<bool>();
   
    bool add_yaw_to_prediction = trained_model_param_node["trained_model_parameter"]["setting"]["add_yaw_to_prediction"].as<bool>();
    if (add_vel_to_prediction && add_yaw_to_prediction) {
      state_component_predicted_ = {"vel", "yaw", "acc", "steer"};
    }
    else if (add_vel_to_prediction) {
      state_component_predicted_ = {"vel", "acc", "steer"};
    }
    else if (add_yaw_to_prediction) {
      state_component_predicted_ = {"yaw", "acc", "steer"};
    }
    else {
      state_component_predicted_ = {"acc", "steer"};
    }
    if (add_position_to_prediction) {
      state_component_predicted_.insert(state_component_predicted_.begin(), "y");
      state_component_predicted_.insert(state_component_predicted_.begin(), "x");
    }
  }
  TrainedDynamics::~TrainedDynamics() {}
  void TrainedDynamics::set_vehicle_params(
    double wheel_base, double acc_time_delay, double steer_time_delay, double acc_time_constant,
    double steer_time_constant, int acc_queue_size, int steer_queue_size, double control_dt,
    int predict_step)
  {
    wheel_base_ = wheel_base;
    acc_time_delay_ = acc_time_delay;
    steer_time_delay_ = steer_time_delay;
    acc_time_constant_ = acc_time_constant;
    steer_time_constant_ = steer_time_constant;
    acc_queue_size_ = acc_queue_size;
    steer_queue_size_ = steer_queue_size;

    predict_step_ = predict_step;
    acc_input_end_index_ = 6 + acc_queue_size_ - 1;
    steer_input_start_index_ = 6 + acc_queue_size_;
    steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
    control_dt_ = control_dt;
    predict_dt_ = control_dt_ * predict_step_;

    acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
    steer_delay_step_ =
      std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);
    nominal_dynamics_.set_params(
      wheel_base, acc_time_delay, steer_time_delay, acc_time_constant, steer_time_constant,
      acc_queue_size, steer_queue_size, control_dt, predict_step);
  }
  void TrainedDynamics::set_NN_params(
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias)
  {
    transform_model_to_eigen_.set_params(
      weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2,
      weight_lstm_ih, weight_lstm_hh, weight_complimentary_layer, weight_linear_relu,
      weight_final_layer, bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2,
      bias_steer_layer_2, bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer,
      bias_linear_relu, bias_final_layer, vel_scaling, vel_bias);
      h_dim_ = weight_lstm_hh.cols();
      state_component_predicted_index_ = std::vector<int>(state_component_predicted_.size());
      for (int i = 0; i < int(state_component_predicted_.size()); i++) {
        for(int j = 0; j < int(all_state_name_.size()); j++){
          if(state_component_predicted_[i] == all_state_name_[j]){
            state_component_predicted_index_[i] = j;
          }
        }
      }
  }
  void TrainedDynamics::clear_NN_params()
  {
    transform_model_to_eigen_.clear_params();
  }
  void TrainedDynamics::set_sg_filter_params(int degree, int window_size)
  {
    filter_diff_NN_.set_sg_filter_params(degree, window_size, state_dim_,h_dim_,
                   acc_queue_size_, steer_queue_size_, predict_step_, control_dt_);
  }
  void TrainedDynamics::update_lstm_states(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm)
  {
    Eigen::VectorXd NN_input = Eigen::VectorXd::Zero(3 + acc_queue_size_ + steer_queue_size_+2*predict_step_);
    NN_input << states[vel_index_], states[acc_index_], states[steer_index_],
      acc_input_history_concat, steer_input_history_concat;
    Eigen::VectorXd h_lstm_next, c_lstm_next;
    std::vector<Eigen::VectorXd> u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2;
    std::vector<Eigen::VectorXd> u_i_lstm, u_f_lstm, u_g_lstm, u_o_lstm;
    std::vector<Eigen::VectorXd> i_lstm, f_lstm, g_lstm, o_lstm;
    std::vector<Eigen::VectorXd> u_complimentary_layer, h_complimentary_layer;

    transform_model_to_eigen_.update_lstm(
      NN_input, h_lstm, c_lstm, h_lstm_next, c_lstm_next,
      u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2, u_i_lstm,
      u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm,
      u_complimentary_layer, h_complimentary_layer);

    h_lstm = h_lstm_next;
    c_lstm = c_lstm_next;
  }
  void TrainedDynamics::initialize_compensation()
  {
    linear_regression_compensation_.initialize();
  }
  void TrainedDynamics::update_input_queue_for_compensation(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat){
    linear_regression_compensation_.update_input_queue(states, acc_input_history_concat, steer_input_history_concat);
  }
  void TrainedDynamics::update_regression_matrix_for_compensation(Eigen::VectorXd error_vector){
    linear_regression_compensation_.update_regression_matrix(error_vector);
  }
  Eigen::VectorXd TrainedDynamics::prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat
  ){
    return linear_regression_compensation_.predict(states, acc_input_history_concat, steer_input_history_concat);
  }
  Eigen::MatrixXd TrainedDynamics::Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat
  ){
    return linear_regression_compensation_.Predict(States, Acc_input_history_concat, Steer_input_history_concat);
  }
  Eigen::VectorXd TrainedDynamics::get_compensation_bias(){
    return linear_regression_compensation_.get_bias();
  }
  Eigen::VectorXd TrainedDynamics::F_with_model_for_calc_controller_prediction_error(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon)
  {
    Eigen::VectorXd states_next = nominal_dynamics_.F_with_input_history(
      states, acc_input_history_concat, steer_input_history_concat);

    Eigen::VectorXd NN_input = Eigen::VectorXd::Zero(3 + acc_queue_size_ + steer_queue_size_+2*predict_step_ );
    NN_input << states[vel_index_], states[acc_index_], states[steer_index_],
      acc_input_history_concat, steer_input_history_concat;
    
    Eigen::VectorXd h_lstm_next, c_lstm_next, NN_output;
    transform_model_to_eigen_.error_prediction(
      NN_input, h_lstm, c_lstm, h_lstm_next, c_lstm_next, NN_output);
    if (horizon == 0) {
      previous_error = error_decay_rate_* previous_error +
                       (1 - error_decay_rate_) * NN_output;
    } else {
      previous_error =
        double_power(error_decay_rate_, predict_step_) * previous_error +
        (1.0 - double_power(error_decay_rate_, predict_step_)) * NN_output;
    }
    double yaw = states[yaw_index_];
    for (int i = 0; i< int(state_component_predicted_index_.size()); i++) {
      if (state_component_predicted_index_[i] == x_index_) {
        states_next[x_index_] += previous_error[i] * predict_dt_ * std::cos(yaw) - previous_error[i+1] * predict_dt_ * std::sin(yaw);
      } else if (state_component_predicted_index_[i] == y_index_) {
        states_next[y_index_] += previous_error[i-1] * predict_dt_ * std::sin(yaw) + previous_error[i] * predict_dt_ * std::cos(yaw);
      } else {
        states_next[state_component_predicted_index_[i]] += previous_error[i] * predict_dt_;
      }
    }
    h_lstm = h_lstm_next;
    c_lstm = c_lstm_next;

    return states_next;
  }
  Eigen::VectorXd TrainedDynamics::F_with_model_for_compensation(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon)
  {
    Eigen::VectorXd states_next = nominal_dynamics_.F_with_input_history(
      states, acc_input_history_concat, steer_input_history_concat);

    Eigen::VectorXd NN_input = Eigen::VectorXd::Zero(3 + acc_queue_size_ + steer_queue_size_+2*predict_step_ );
    NN_input << states[vel_index_], states[acc_index_], states[steer_index_],
      acc_input_history_concat, steer_input_history_concat;
    
    Eigen::VectorXd h_lstm_next, c_lstm_next, NN_output;
    transform_model_to_eigen_.error_prediction(
      NN_input, h_lstm, c_lstm, h_lstm_next, c_lstm_next, NN_output);
    if (horizon == 0) {
      previous_error = NN_output;
    } else {
      previous_error =
        double_power(error_decay_rate_, predict_step_) * previous_error +
        (1.0 - double_power(error_decay_rate_, predict_step_)) * NN_output;
    }
    double yaw = states[yaw_index_];
    for (int i = 0; i< int(state_component_predicted_index_.size()); i++) {
      if (state_component_predicted_index_[i] == x_index_) {
        states_next[x_index_] += previous_error[i] * predict_dt_ * std::cos(yaw) - previous_error[i+1] * predict_dt_ * std::sin(yaw);
      } else if (state_component_predicted_index_[i] == y_index_) {
        states_next[y_index_] += previous_error[i-1] * predict_dt_ * std::sin(yaw) + previous_error[i] * predict_dt_ * std::cos(yaw);
      } else {
        states_next[state_component_predicted_index_[i]] += previous_error[i] * predict_dt_;
      }
    }
    h_lstm = h_lstm_next;
    c_lstm = c_lstm_next;

    return states_next;
  }
  Eigen::VectorXd TrainedDynamics::F_with_model(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat,
    const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon)
  {
    Eigen::VectorXd states_next = nominal_dynamics_.F_with_input_history(
      states, acc_input_history, steer_input_history, acc_input_history_concat,
      steer_input_history_concat, d_inputs);

    Eigen::VectorXd NN_input = Eigen::VectorXd::Zero(3 + acc_queue_size_ + steer_queue_size_+2*predict_step_ );
    NN_input << states[vel_index_], states[acc_index_], states[steer_index_],
      acc_input_history_concat, steer_input_history_concat;
    
    Eigen::VectorXd h_lstm_next, c_lstm_next, NN_output;
    transform_model_to_eigen_.error_prediction(
      NN_input, h_lstm, c_lstm, h_lstm_next, c_lstm_next, NN_output);
    if (horizon == 0) {
      previous_error = error_decay_rate_* previous_error +
                       (1 - error_decay_rate_) * NN_output;
    } else {
      previous_error =
        double_power(error_decay_rate_, predict_step_) * previous_error +
        (1.0 - double_power(error_decay_rate_, predict_step_)) * NN_output;
    }
    double yaw = states[yaw_index_];
    for (int i = 0; i< int(state_component_predicted_index_.size()); i++) {
      if (state_component_predicted_index_[i] == x_index_) {
        states_next[x_index_] += previous_error[i] * predict_dt_ * std::cos(yaw) - previous_error[i+1] * predict_dt_ * std::sin(yaw);
      } else if (state_component_predicted_index_[i] == y_index_) {
        states_next[y_index_] += previous_error[i-1] * predict_dt_ * std::sin(yaw) + previous_error[i] * predict_dt_ * std::cos(yaw);
      } else {
        states_next[state_component_predicted_index_[i]] += previous_error[i] * predict_dt_;
      }
    }
    Eigen::VectorXd compensation = prediction_for_compensation(states, acc_input_history_concat, steer_input_history_concat);
    states_next += compensation;
    h_lstm = h_lstm_next;
    c_lstm = c_lstm_next;

    return states_next;
  }
  Eigen::VectorXd TrainedDynamics::F_with_model(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon)
  {
    Eigen::VectorXd acc_input_history_concat, steer_input_history_concat;
    Eigen::VectorXd states_next = F_with_model(
      states, acc_input_history, steer_input_history, acc_input_history_concat,
      steer_input_history_concat, d_inputs, h_lstm, c_lstm, previous_error,
      horizon);
    return states_next;
  }  
  Eigen::VectorXd TrainedDynamics::F_with_model_diff(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat,
    const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon,
    Eigen::MatrixXd & A, Eigen::MatrixXd & B, Eigen::MatrixXd & C)
  {
    Eigen::VectorXd states_next = nominal_dynamics_.F_with_input_history_and_diff(
      states, acc_input_history, steer_input_history, acc_input_history_concat,
      steer_input_history_concat, d_inputs, A, B);
    Eigen::VectorXd NN_input =
      Eigen::VectorXd::Zero(3 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_);
    NN_input << states[vel_index_], states[acc_index_], states[steer_index_],
      acc_input_history_concat, steer_input_history_concat;
    Eigen::VectorXd h_lstm_next, c_lstm_next, NN_output;
    Eigen::MatrixXd dF_d_states_with_history, dF_dhc, dhc_dhc, dhc_d_states_with_history;
    transform_model_to_eigen_.error_prediction_with_diff(
      NN_input, h_lstm, c_lstm, h_lstm_next, c_lstm_next, NN_output, dF_d_states_with_history,
      dF_dhc, dhc_dhc, dhc_d_states_with_history);
    int h_dim = h_lstm.size();
    double yaw = states[yaw_index_];
    C = Eigen::MatrixXd::Zero(
      2*h_dim + states.size(),
      2*h_dim + states.size() + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_);
    for (int i = 0; i< int(state_component_predicted_index_.size()); i++) {
      if (state_component_predicted_index_[i] == x_index_) {
        C(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 2) += dF_d_states_with_history(i, 0) * predict_dt_ * std::cos(yaw);
        C(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 2) -= dF_d_states_with_history(i + 1, 0) * predict_dt_ * std::sin(yaw);
        C.block(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) += dF_d_states_with_history.block(i, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) * predict_dt_ * std::cos(yaw);
        C.block(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) -= dF_d_states_with_history.block(i + 1, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) * predict_dt_ * std::sin(yaw);
        C.block(2 * h_dim + state_component_predicted_index_[i], 0, 1, 2 * h_dim) += dF_dhc.row(i) * predict_dt_ * std::cos(yaw);
        C.block(2 * h_dim + state_component_predicted_index_[i], 0, 1, 2 * h_dim) -= dF_dhc.row(i + 1) * predict_dt_ * std::sin(yaw);
      } else if (state_component_predicted_index_[i] == y_index_) {
        C(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 2) += dF_d_states_with_history(i - 1, 0) * predict_dt_ * std::sin(yaw);
        C(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 2) += dF_d_states_with_history(i, 0) * predict_dt_ * std::cos(yaw);
        C.block(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) += dF_d_states_with_history.block(i - 1, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) * predict_dt_ * std::sin(yaw);
        C.block(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) += dF_d_states_with_history.block(i, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) * predict_dt_ * std::cos(yaw);
        C.block(2 * h_dim + state_component_predicted_index_[i], 0, 1, 2 * h_dim) += dF_dhc.row(i - 1) * predict_dt_ * std::sin(yaw);
        C.block(2 * h_dim + state_component_predicted_index_[i], 0, 1, 2 * h_dim) += dF_dhc.row(i) * predict_dt_ * std::cos(yaw);
      } else {
        C(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim+2) += dF_d_states_with_history(i, 0) * predict_dt_;
        C.block(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) += dF_d_states_with_history.block(i, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_) * predict_dt_;
        C.block(2 * h_dim + state_component_predicted_index_[i], 0, 1, 2 * h_dim) += dF_dhc.row(i) * predict_dt_;
      }
    }
    // C.block(2 * h_dim+4, 2 * h_dim+2, states.size()-4, 1) = dF_d_states_with_history.col(0) * predict_dt_;
    // C.block(2*h_dim+4, 2*h_dim+3, states.size()-4, states.size() -3-1+acc_queue_size_+steer_queue_size_+2*predict_step_) = dF_d_states_with_history.rightCols(states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_) * predict_dt_;
    
    C.block(0, 0, 2 * h_dim, 2 * h_dim) = dhc_dhc;

    C.block(0, 2 * h_dim + 2, 2 * h_dim, 1) = dhc_d_states_with_history.col(0);
    C.block(0, 2 * h_dim + 4, 2 * h_dim, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_) = dhc_d_states_with_history.rightCols(states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_);

    //C.block(2 * h_dim + 4, 0, states.size() -4, 2 * h_dim) = dF_dhc;

    Eigen::VectorXd d_steer_d_steer_input =  C.block(2 * h_dim + steer_index_, 2 * h_dim + steer_input_start_index_ + predict_step_, 1, steer_queue_size_ + predict_step_).transpose();
    d_steer_d_steer_input.head(steer_queue_size_) += A.block(steer_index_, steer_input_start_index_, 1, steer_queue_size_).transpose();
    double steer_diff = d_steer_d_steer_input.sum();

    if (steer_diff < minimum_steer_diff_) {
      int max_d_steer_index;
      d_steer_d_steer_input.array().maxCoeff(&max_d_steer_index);
      max_d_steer_index = std::min(max_d_steer_index, steer_queue_size_);
      for (int i = 0; i< predict_step_; i++) {
        C(2 * h_dim + steer_index_, 2 * h_dim + steer_input_start_index_ + predict_step_ + max_d_steer_index + i) += (minimum_steer_diff_ - steer_diff)/predict_step_;
      }
    }
    if (horizon == 0) {
      previous_error = error_decay_rate_ * previous_error +
                       (1 - error_decay_rate_) * NN_output;
    } else {
      previous_error =
        double_power(error_decay_rate_, predict_step_) * previous_error +
        (1 - double_power(error_decay_rate_, predict_step_)) * NN_output;
    }

    h_lstm = h_lstm_next;
    c_lstm = c_lstm_next;
    for (int i = 0; i < int(state_component_predicted_index_.size()); i++) {
      if (state_component_predicted_index_[i] == x_index_) {
        states_next[x_index_] += previous_error[i] * predict_dt_ * std::cos(yaw) - previous_error[i + 1] * predict_dt_ * std::sin(yaw);
        C(2 * h_dim + state_component_predicted_index_[i],2 * h_dim + yaw_index_) += - previous_error[i] * predict_dt_ * std::sin(yaw) - previous_error[i + 1] * predict_dt_ * std::cos(yaw);
      } else if (state_component_predicted_index_[i] == y_index_) {
        states_next[y_index_] += previous_error[i - 1] * predict_dt_ * std::sin(yaw) + previous_error[i] * predict_dt_ * std::cos(yaw);
        C(2 * h_dim + state_component_predicted_index_[i], 2 * h_dim + yaw_index_) += previous_error[i - 1] * predict_dt_ * std::cos(yaw) - previous_error[i] * predict_dt_ * std::sin(yaw);
      } else {
        states_next[state_component_predicted_index_[i]] += previous_error[i] * predict_dt_;
      }
    }
    Eigen::VectorXd compensation = prediction_for_compensation(states, acc_input_history_concat, steer_input_history_concat);
    states_next += compensation;
    return states_next;
  }
  Eigen::VectorXd TrainedDynamics::F_with_model_diff(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon,
    Eigen::MatrixXd & A, Eigen::MatrixXd & B, Eigen::MatrixXd & C)
  {
    Eigen::VectorXd acc_input_history_concat, steer_input_history_concat;
    Eigen::VectorXd states_next = F_with_model_diff(
      states, acc_input_history, steer_input_history, acc_input_history_concat,
      steer_input_history_concat, d_inputs, h_lstm, c_lstm, previous_error,
      horizon, A, B, C);
    return states_next;
  }
  Eigen::MatrixXd TrainedDynamics::F_with_model_for_candidates(
    const Eigen::MatrixXd & States, Eigen::MatrixXd & Acc_input_history,
    Eigen::MatrixXd & Steer_input_history, 
    Eigen::MatrixXd & Acc_input_history_concat, Eigen::MatrixXd & Steer_input_history_concat,
    const Eigen::MatrixXd & D_inputs,
    Eigen::MatrixXd & H_lstm, Eigen::MatrixXd & C_lstm, Eigen::MatrixXd & Previous_error, const int horizon)
  {
    Eigen::MatrixXd States_next = nominal_dynamics_.F_with_input_history_for_candidates(
      States, Acc_input_history, Steer_input_history, Acc_input_history_concat,
      Steer_input_history_concat, D_inputs);
    Eigen::MatrixXd NN_input =
      Eigen::MatrixXd::Zero(3 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_, States.cols());
    NN_input.row(0)=States.row(vel_index_);
    NN_input.row(1) = States.row(acc_index_);
    NN_input.row(2) = States.row(steer_index_),
    NN_input.middleRows(3, acc_queue_size_+predict_step_) = Acc_input_history_concat;
    NN_input.middleRows(3+acc_queue_size_+predict_step_, steer_queue_size_+predict_step_) = Steer_input_history_concat;
    
    for (int i = 0; i < States.rows(); i++) {
      Eigen::VectorXd h_lstm = H_lstm.col(i);
      Eigen::VectorXd c_lstm = C_lstm.col(i);
      Eigen::VectorXd NN_output;
      Eigen::VectorXd h_lstm_next, c_lstm_next;
      transform_model_to_eigen_.error_prediction(
        NN_input.col(i), h_lstm, c_lstm, h_lstm_next, c_lstm_next,
        NN_output);
      if (horizon == 0) {
        Previous_error.col(i) = error_decay_rate_ * Previous_error.col(i) +
                                (1.0 - error_decay_rate_) * NN_output;
      } else {
        Previous_error.col(i) =
          double_power(error_decay_rate_, predict_step_) * Previous_error.col(i) +
          (1.0 - double_power(error_decay_rate_, predict_step_)) * NN_output;
      }
      H_lstm.col(i) = h_lstm_next;
      C_lstm.col(i) = c_lstm_next;
    }
    Eigen::RowVectorXd yaw = States.row(yaw_index_);
    for (int i = 0; i< int(state_component_predicted_index_.size()); i++) {
      if (state_component_predicted_index_[i] == x_index_) {
        States_next.row(x_index_) += (Previous_error.row(i).array() * predict_dt_ * yaw.array().cos() - Previous_error.row(i+1).array() * predict_dt_ * yaw.array().sin()).matrix() ;
        //continue;
      }
      else if (state_component_predicted_index_[i] == y_index_) {
        States_next.row(y_index_) += (Previous_error.row(i-1).array() * predict_dt_ * yaw.array().sin() + Previous_error.row(i).array() * predict_dt_ * yaw.array().cos()).matrix() ;
        //continue;
      }
      else {
        States_next.row(state_component_predicted_index_[i]) += Previous_error.row(i) * predict_dt_;
      }
    }
    Eigen::MatrixXd Compensation = Prediction_for_compensation(States, Acc_input_history_concat, Steer_input_history_concat);
    States_next += Compensation;
    return States_next;
  }
  Eigen::MatrixXd TrainedDynamics::F_with_model_for_candidates(
    const Eigen::MatrixXd & States, Eigen::MatrixXd & Acc_input_history,
    Eigen::MatrixXd & Steer_input_history, const Eigen::MatrixXd & D_inputs,
    Eigen::MatrixXd & H_lstm, Eigen::MatrixXd & C_lstm, Eigen::MatrixXd & Previous_error, const int horizon)
  {
    Eigen::MatrixXd Acc_input_history_concat, Steer_input_history_concat;
    Eigen::MatrixXd States_next = F_with_model_for_candidates(
      States, Acc_input_history, Steer_input_history, Acc_input_history_concat,
      Steer_input_history_concat, D_inputs, H_lstm, C_lstm, Previous_error,
      horizon);
    return States_next;
  }
  void TrainedDynamics::calc_forward_trajectory_with_diff(Eigen::VectorXd states, Eigen::VectorXd acc_input_history,
                                         Eigen::VectorXd steer_input_history, std::vector<Eigen::VectorXd> d_inputs_schedule,
                                         const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
                                         const Eigen::VectorXd & previous_error, std::vector<Eigen::VectorXd> & states_prediction,
                                         std::vector<Eigen::MatrixXd> & dF_d_states, std::vector<Eigen::MatrixXd> & dF_d_inputs,
                                         std::vector<Eigen::Vector2d> & inputs_schedule)
  {
    int horizon_len = d_inputs_schedule.size();
    Eigen::VectorXd previous_error_tmp = previous_error;
    states_prediction = std::vector<Eigen::VectorXd>(horizon_len + 1);
    dF_d_states = std::vector<Eigen::MatrixXd>(horizon_len);
    dF_d_inputs = std::vector<Eigen::MatrixXd>(horizon_len);
    inputs_schedule = std::vector<Eigen::Vector2d>(horizon_len);
    std::vector<Eigen::MatrixXd> A_vec(horizon_len), B_vec(horizon_len), C_vec(horizon_len);
    states_prediction[0] = states;
    Eigen::VectorXd acc_input_history_tmp, steer_input_history_tmp, h_lstm_tmp, c_lstm_tmp;

    acc_input_history_tmp = acc_input_history;
    steer_input_history_tmp = steer_input_history;
    h_lstm_tmp = h_lstm;
    c_lstm_tmp = c_lstm;

    for (int i = 0; i < horizon_len; i++) {
      Eigen::Vector2d d_inputs = d_inputs_schedule[i];
      Eigen::MatrixXd A, B, C;
      states_prediction[i + 1] = F_with_model_diff(
        states_prediction[i], acc_input_history_tmp, steer_input_history_tmp,
        d_inputs, h_lstm_tmp, c_lstm_tmp,
        previous_error_tmp, i, A, B, C);
      inputs_schedule[i][0] = acc_input_history_tmp[acc_input_history_tmp.size() - predict_step_];
      inputs_schedule[i][1] = steer_input_history_tmp[steer_input_history_tmp.size() - predict_step_];

      A_vec[i] = A;
      B_vec[i] = B;
      C_vec[i] = C;
    }
    filter_diff_NN_.fit_transform_for_NN_diff(A_vec,B_vec,C_vec,dF_d_states,dF_d_inputs);
  }
///////////////// AdaptorILQR ///////////////////////
  
  AdaptorILQR::AdaptorILQR() {
    set_params();
  }
  AdaptorILQR::~AdaptorILQR() {}
  void AdaptorILQR::set_params()
  {
    YAML::Node optimization_param_node = YAML::LoadFile(get_param_dir_path() + "/optimization_param.yaml");
    bool add_position_to_ilqr = optimization_param_node["optimization_parameter"]["ilqr"]["add_position_to_ilqr"].as<bool>();
    bool add_yaw_to_ilqr = optimization_param_node["optimization_parameter"]["ilqr"]["add_yaw_to_ilqr"].as<bool>();
    if (add_yaw_to_ilqr) {
      state_component_ilqr_ = {"vel", "yaw", "acc", "steer"};
    }
    if (add_position_to_ilqr) {
      state_component_ilqr_.insert(state_component_ilqr_.begin(), "y");
      state_component_ilqr_.insert(state_component_ilqr_.begin(), "x");
    }
    acc_input_weight_vel_error_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_input_weight_vel_error_target_table"].as<std::vector<double>>();
    acc_input_weight_vel_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_input_weight_vel_error_domain_table"].as<std::vector<double>>();
    acc_input_weight_acc_error_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_input_weight_acc_error_target_table"].as<std::vector<double>>();
    acc_input_weight_acc_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_input_weight_acc_error_domain_table"].as<std::vector<double>>();
    steer_input_weight_lateral_error_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_input_weight_lateral_error_target_table"].as<std::vector<double>>();
    steer_input_weight_lateral_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_input_weight_lateral_error_domain_table"].as<std::vector<double>>();
    steer_input_weight_yaw_error_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_input_weight_yaw_error_target_table"].as<std::vector<double>>();
    steer_input_weight_yaw_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_input_weight_yaw_error_domain_table"].as<std::vector<double>>();
    steer_input_weight_steer_error_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_input_weight_steer_error_target_table"].as<std::vector<double>>();
    steer_input_weight_steer_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_input_weight_steer_error_domain_table"].as<std::vector<double>>();
  }
  void AdaptorILQR::set_states_cost(
    double x_cost, double y_cost, double vel_cost, double yaw_cost, double acc_cost, double steer_cost
  )
  {
    x_cost_ = x_cost;
    y_cost_ = y_cost;
    vel_cost_ = vel_cost;
    yaw_cost_ = yaw_cost;
    steer_cost_ = steer_cost;
    acc_cost_ = acc_cost;
  }
  void AdaptorILQR::set_inputs_cost(
     double acc_rate_cost, double steer_rate_cost
  )
  {
    steer_rate_cost_ = steer_rate_cost;
    acc_rate_cost_ = acc_rate_cost;
  }
  void AdaptorILQR::set_rate_cost(
    double acc_rate_rate_cost, double steer_rate_rate_cost
  )
  {
    steer_rate_rate_cost_ = steer_rate_rate_cost;
    acc_rate_rate_cost_ = acc_rate_rate_cost;
  }
  void AdaptorILQR::set_intermediate_cost(
    double x_intermediate_cost, double y_intermediate_cost, double vel_intermediate_cost, double yaw_intermediate_cost, double acc_intermediate_cost, double steer_intermediate_cost, int intermediate_cost_index
  )
  {
    x_intermediate_cost_ = x_intermediate_cost;
    y_intermediate_cost_ = y_intermediate_cost;
    vel_intermediate_cost_ = vel_intermediate_cost;
    yaw_intermediate_cost_ = yaw_intermediate_cost;
    steer_intermediate_cost_ = steer_intermediate_cost;
    acc_intermediate_cost_ = acc_intermediate_cost;
    intermediate_cost_index_ = intermediate_cost_index;
  }
  
  void AdaptorILQR::set_terminal_cost(
    double x_terminal_cost, double y_terminal_cost, double vel_terminal_cost, double yaw_terminal_cost, double acc_terminal_cost, double steer_terminal_cost
  )
  {
    x_terminal_cost_ = x_terminal_cost;
    y_terminal_cost_ = y_terminal_cost;
    vel_terminal_cost_ = vel_terminal_cost;
    yaw_terminal_cost_ = yaw_terminal_cost;
    steer_terminal_cost_ = steer_terminal_cost;
    acc_terminal_cost_ = acc_terminal_cost;
  }
  void AdaptorILQR::set_vehicle_params(
    double wheel_base, double acc_time_delay, double steer_time_delay, double acc_time_constant,
    double steer_time_constant, int acc_queue_size, int steer_queue_size, double control_dt,
    int predict_step)
  {
    wheel_base_ = wheel_base;
    acc_time_delay_ = acc_time_delay;
    steer_time_delay_ = steer_time_delay;
    acc_time_constant_ = acc_time_constant;
    steer_time_constant_ = steer_time_constant;
    acc_queue_size_ = acc_queue_size;
    steer_queue_size_ = steer_queue_size;

    predict_step_ = predict_step;
    acc_input_end_index_ = 6 + acc_queue_size_ - 1;
    steer_input_start_index_ = 6 + acc_queue_size_;
    steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
    control_dt_ = control_dt;
    predict_dt_ = control_dt_ * predict_step_;

    acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
    steer_delay_step_ =
      std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);
    trained_dynamics_.set_vehicle_params(
      wheel_base, acc_time_delay, steer_time_delay, acc_time_constant, steer_time_constant,
      acc_queue_size, steer_queue_size, control_dt, predict_step);
  }
  void AdaptorILQR::set_NN_params(
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias)
  {
    trained_dynamics_.set_NN_params(
      weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2,
      weight_lstm_ih, weight_lstm_hh, weight_complimentary_layer, weight_linear_relu,
      weight_final_layer, bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2,
      bias_steer_layer_2, bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer,
      bias_linear_relu, bias_final_layer, vel_scaling, vel_bias);
    h_dim_ = weight_lstm_hh.cols();
    num_state_component_ilqr_ = int(state_component_ilqr_.size());
    state_component_ilqr_index_ = std::vector<int>(num_state_component_ilqr_);
    for (int i = 0; i < int(state_component_ilqr_.size()); i++) {
      for(int j = 0; j < int(all_state_name_.size()); j++){
        if(state_component_ilqr_[i] == all_state_name_[j]){
          state_component_ilqr_index_[i] = j;
        }
      }
    }
  }
  void AdaptorILQR::clear_NN_params()
  {
    trained_dynamics_.clear_NN_params();
  }
  void AdaptorILQR::set_sg_filter_params(int degree, int horizon_len,int window_size)
  {
    horizon_len_ = horizon_len;
    trained_dynamics_.set_sg_filter_params(degree, window_size);
  }
  void AdaptorILQR::update_lstm_states(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm)
  {
    trained_dynamics_.update_lstm_states(
      states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
  }
  Eigen::VectorXd AdaptorILQR::F_with_model_for_compensation(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, int horizon)
  {
    return trained_dynamics_.F_with_model_for_compensation(
      states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm, previous_error, horizon);
  }
  void AdaptorILQR::initialize_compensation(){
    trained_dynamics_.initialize_compensation();
  }
  void AdaptorILQR::update_input_queue_for_compensation(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat){
    trained_dynamics_.update_input_queue_for_compensation(states, acc_input_history_concat, steer_input_history_concat);
  }
  void AdaptorILQR::update_regression_matrix_for_compensation(Eigen::VectorXd error_vector){
    trained_dynamics_.update_regression_matrix_for_compensation(error_vector);
  }
  Eigen::VectorXd AdaptorILQR::prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat
  ){
    return trained_dynamics_.prediction_for_compensation(states, acc_input_history_concat, steer_input_history_concat);
  }
  Eigen::MatrixXd AdaptorILQR::Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat
  ){
    return trained_dynamics_.Prediction_for_compensation(States, Acc_input_history_concat, Steer_input_history_concat);
  }
  Eigen::VectorXd AdaptorILQR::get_compensation_bias(){
    return trained_dynamics_.get_compensation_bias();
  }
  void AdaptorILQR::calc_forward_trajectory_with_cost(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history, Eigen::VectorXd steer_input_history,
    std::vector<Eigen::MatrixXd> D_inputs_schedule, const Eigen::VectorXd & h_lstm,
    const Eigen::VectorXd & c_lstm,
    const Eigen::VectorXd & previous_error, std::vector<Eigen::MatrixXd> & states_prediction,
    const std::vector<Eigen::VectorXd> & states_ref, const std::vector<Eigen::VectorXd> & d_input_ref, 
    std::vector<Eigen::Vector2d> inputs_ref, double acc_input_weight, double steer_input_weight,
    Eigen::VectorXd & Cost)
  {
    int sample_size = D_inputs_schedule[0].cols();
    Cost = Eigen::VectorXd::Zero(sample_size);
    states_prediction = std::vector<Eigen::MatrixXd>(horizon_len_+1);
    states_prediction[0] = states.replicate(1, sample_size);
    Eigen::MatrixXd Previous_error = previous_error.replicate(1, sample_size);
    Eigen::MatrixXd Acc_input_history = acc_input_history.replicate(1, sample_size);
    Eigen::MatrixXd Steer_input_history = steer_input_history.replicate(1, sample_size);
    Eigen::MatrixXd H_lstm = h_lstm.replicate(1, sample_size);
    Eigen::MatrixXd C_lstm = c_lstm.replicate(1, sample_size);
    for (int i = 0; i < horizon_len_; i++) {
      Eigen::MatrixXd D_inputs = D_inputs_schedule[i];
      double x_cost = x_cost_;
      double y_cost = y_cost_;
      double vel_cost = vel_cost_;
      double yaw_cost = yaw_cost_;
      double acc_cost = acc_cost_;
      double steer_cost = steer_cost_;
      if (i == intermediate_cost_index_) {
        x_cost = x_intermediate_cost_;
        y_cost = y_intermediate_cost_;
        vel_cost = vel_intermediate_cost_;
        yaw_cost = yaw_intermediate_cost_;
        acc_cost = acc_intermediate_cost_;
        steer_cost = steer_intermediate_cost_;
      }
      double yaw_target = states_ref[i][yaw_index_];
      Eigen::Matrix2d rotation_matrix = Eigen::Rotation2Dd(-yaw_target).toRotationMatrix();
      Eigen::Matrix2d xy_cost_matrix_sqrt = rotation_matrix.transpose() * Eigen::DiagonalMatrix<double, 2>(std::sqrt(x_cost), std::sqrt(y_cost)) * rotation_matrix;
      Eigen::MatrixXd xy_diff = states_prediction[i].block(x_index_, 0, 2, sample_size).colwise() - states_ref[i].segment(x_index_, 2);
      for (int j = 0; j < num_state_component_ilqr_; j++) {
        if (state_component_ilqr_[j] == "x") {
          Cost += 0.5 * (xy_cost_matrix_sqrt*xy_diff).colwise().squaredNorm();
        } else if (state_component_ilqr_[j] == "vel") {
          Cost += 0.5 * vel_cost * (states_prediction[i].row(vel_index_).array() - states_ref[i][vel_index_]).square().matrix();
        } else if (state_component_ilqr_[j] == "yaw") {
          Cost += 0.5 * yaw_cost * (states_prediction[i].row(yaw_index_).array() - states_ref[i][yaw_index_]).square().matrix();
        } else if (state_component_ilqr_[j] == "acc") {
          Cost += 0.5 * acc_cost * (states_prediction[i].row(acc_index_).array() - states_ref[i][acc_index_]).square().matrix();
        } else if (state_component_ilqr_[j] == "steer") {
          Cost += 0.5 * steer_cost * (states_prediction[i].row(steer_index_).array() - states_ref[i][steer_index_]).square().matrix();
        } 
      }

      //Cost += 0.5 * acc_cost * (states_prediction[i].row(acc_index_).array() - states_ref[i][acc_index_]).square().matrix();
      
      //Cost += 0.5 * steer_cost * (states_prediction[i].row(steer_index_).array() - states_ref[i][steer_index_]).square().matrix();
    
      Cost += 0.5 * acc_rate_cost_ * (D_inputs.row(0).array() - d_input_ref[i][0]).square().matrix();
     
      Cost += 0.5 * steer_rate_cost_ * (D_inputs.row(1).array() - d_input_ref[i][1]).square().matrix();
      if (i == 0) {
        double prev_acc_rate = (acc_input_history[acc_queue_size_ - 1] - acc_input_history[acc_queue_size_ - 2])/control_dt_;
        double prev_steer_rate = (steer_input_history[steer_queue_size_ - 1] - steer_input_history[steer_queue_size_ - 2])/control_dt_;
        Cost += 0.5 * acc_rate_rate_cost_ * (D_inputs.row(0).array() - prev_acc_rate).square().matrix();
        Cost += 0.5 * steer_rate_rate_cost_ * (D_inputs.row(1).array() - prev_steer_rate).square().matrix();
      }
      if (i > 0){
        Cost += 0.5 * acc_rate_rate_cost_ * (D_inputs.row(0).array() - D_inputs_schedule[i-1].row(0).array()).square().matrix();
        Cost += 0.5 * steer_rate_rate_cost_ * (D_inputs.row(1).array() - D_inputs_schedule[i-1].row(1).array()).square().matrix();
        Cost += 0.5 * acc_input_weight * (Acc_input_history.row(acc_queue_size_ - predict_step_).array() - inputs_ref[i - 1][0]).square().matrix();
        Cost += 0.5 * steer_input_weight * (Steer_input_history.row(steer_queue_size_ - predict_step_).array() - inputs_ref[i - 1][1]).square().matrix();
      }
      states_prediction[i + 1] = trained_dynamics_.F_with_model_for_candidates(
        states_prediction[i], Acc_input_history, Steer_input_history, 
        D_inputs, H_lstm, C_lstm, Previous_error, i);
    }
    double yaw_target = states_ref[horizon_len_][yaw_index_];
    Eigen::Matrix2d rotation_matrix = Eigen::Rotation2Dd(-yaw_target).toRotationMatrix();
    Eigen::Matrix2d xy_cost_matrix_sqrt = rotation_matrix.transpose() * Eigen::DiagonalMatrix<double, 2>(std::sqrt(x_terminal_cost_), std::sqrt(y_terminal_cost_)) * rotation_matrix;
    Eigen::MatrixXd xy_diff = states_prediction[horizon_len_].block(x_index_, 0, 2, sample_size).colwise() - states_ref[horizon_len_].segment(x_index_, 2);
    for (int j = 0; j < num_state_component_ilqr_; j++) {
      if (state_component_ilqr_[j] == "x") {
        Cost += 0.5 * (xy_cost_matrix_sqrt*xy_diff).colwise().squaredNorm();
      } else if (state_component_ilqr_[j] == "vel") {
        Cost += 0.5 * vel_terminal_cost_ * (states_prediction[horizon_len_].row(vel_index_).array() - states_ref[horizon_len_][vel_index_]).square().matrix();
      } else if (state_component_ilqr_[j] == "yaw") {
        Cost += 0.5 * yaw_terminal_cost_ * (states_prediction[horizon_len_].row(yaw_index_).array() - states_ref[horizon_len_][yaw_index_]).square().matrix();
      } else if (state_component_ilqr_[j] == "acc") {
        Cost += 0.5 * acc_terminal_cost_ * (states_prediction[horizon_len_].row(acc_index_).array() - states_ref[horizon_len_][acc_index_]).square().matrix();
      } else if (state_component_ilqr_[j] == "steer") {
        Cost += 0.5 * steer_terminal_cost_ * (states_prediction[horizon_len_].row(steer_index_).array() - states_ref[horizon_len_][steer_index_]).square().matrix();
      }
    }

    //Cost += 0.5 * acc_terminal_cost_ * (states_prediction[horizon_len_].row(acc_index_).array() - states_ref[horizon_len_][acc_index_]).square().matrix();

    //Cost += 0.5 * steer_terminal_cost_ * (states_prediction[horizon_len_].row(steer_index_).array() - states_ref[horizon_len_][steer_index_]).square().matrix();
  }
  void AdaptorILQR::calc_inputs_ref_info(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history,
    const Eigen::VectorXd & steer_input_history, const Eigen::VectorXd & h_lstm,
    const Eigen::VectorXd & c_lstm, 
    const Eigen::VectorXd & previous_error, const std::vector<Eigen::VectorXd> & states_ref,
    const Eigen::VectorXd & acc_controller_input_schedule, const Eigen::VectorXd & steer_controller_input_schedule,
    std::vector<Eigen::Vector2d> & inputs_ref, double & acc_input_weight, double & steer_input_weight)
  {
    Eigen::VectorXd acc_input_history_concat(acc_queue_size_ + predict_step_);
    Eigen::VectorXd steer_input_history_concat(steer_queue_size_ + predict_step_);
    acc_input_history_concat.head(acc_queue_size_) = acc_input_history;
    steer_input_history_concat.head(steer_queue_size_) = steer_input_history;
    acc_input_history_concat.tail(predict_step_) = acc_controller_input_schedule.head(predict_step_);
    steer_input_history_concat.tail(predict_step_) = steer_controller_input_schedule.head(predict_step_);
    std::vector<Eigen::VectorXd> states_prediction_by_controller_inputs(horizon_len_ + 1);
    states_prediction_by_controller_inputs[0] = states;
    Eigen::VectorXd previous_error_tmp = previous_error;
    Eigen::VectorXd h_lstm_tmp = h_lstm;
    Eigen::VectorXd c_lstm_tmp = c_lstm;
    inputs_ref = std::vector<Eigen::Vector2d>(horizon_len_);
    double max_vel_error = 0.0;
    double max_lateral_error = 0.0;
    double max_acc_error = 0.0;
    double max_steer_error = 0.0;
    double max_yaw_error = 0.0;
    for (int i = 0; i< horizon_len_; i++) {
      if (i > 0){
        acc_input_history_concat.head(acc_queue_size_) = acc_input_history_concat.tail(acc_queue_size_);
        steer_input_history_concat.head(steer_queue_size_) = steer_input_history_concat.tail(steer_queue_size_);
        acc_input_history_concat.tail(predict_step_) = acc_controller_input_schedule.segment(i*predict_step_, predict_step_);
        steer_input_history_concat.tail(predict_step_) = steer_controller_input_schedule.segment(i*predict_step_, predict_step_);        
      }
      inputs_ref[i][0] = acc_controller_input_schedule[i*predict_step_];
      inputs_ref[i][1] = steer_controller_input_schedule[i*predict_step_];
      states_prediction_by_controller_inputs[i + 1] = trained_dynamics_.F_with_model_for_calc_controller_prediction_error(
        states_prediction_by_controller_inputs[i], acc_input_history_concat, steer_input_history_concat,
        h_lstm_tmp, c_lstm_tmp, previous_error_tmp, i);
      double vel_error = std::abs(states_ref[i+1][vel_index_] - states_prediction_by_controller_inputs[i + 1][vel_index_]);
      double acc_error = std::abs(states_ref[i+1][acc_index_] - states_prediction_by_controller_inputs[i + 1][acc_index_]);
      double steer_error = std::abs(states_ref[i+1][steer_index_] - states_prediction_by_controller_inputs[i + 1][steer_index_]);
      double yaw = states_ref[i+1][yaw_index_];
      double lateral_error = std::abs(- std::sin(yaw) * (states_ref[i+1][x_index_] - states_prediction_by_controller_inputs[i + 1][x_index_])
                                 + std::cos(yaw) * (states_ref[i+1][y_index_] - states_prediction_by_controller_inputs[i + 1][y_index_]));
      double yaw_error = std::abs(states_ref[i+1][yaw_index_] - states_prediction_by_controller_inputs[i + 1][yaw_index_]);
      if (vel_error > max_vel_error) {
        max_vel_error = vel_error;
      }
      if (lateral_error > max_lateral_error) {
        max_lateral_error = lateral_error;
      }
      if (acc_error > max_acc_error) {
        max_acc_error = acc_error;
      }
      if (steer_error > max_steer_error) {
        max_steer_error = steer_error;
      }
      if (yaw_error > max_yaw_error) {
        max_yaw_error = yaw_error;
      }
    }
    acc_input_weight = std::min(
      calc_table_value(max_vel_error, acc_input_weight_vel_error_domain_table_, acc_input_weight_vel_error_target_table_),
      calc_table_value(max_acc_error, acc_input_weight_acc_error_domain_table_, acc_input_weight_acc_error_target_table_));
    steer_input_weight = std::min(
      {
      calc_table_value(max_lateral_error, steer_input_weight_lateral_error_domain_table_, steer_input_weight_lateral_error_target_table_),
      calc_table_value(max_steer_error, steer_input_weight_steer_error_domain_table_, steer_input_weight_steer_error_target_table_),
      calc_table_value(max_yaw_error, steer_input_weight_yaw_error_domain_table_, steer_input_weight_yaw_error_target_table_)
      });
  }
  Eigen::MatrixXd AdaptorILQR::extract_dF_d_state(Eigen::MatrixXd dF_d_state_with_history)
  {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(2*h_dim_ + num_state_component_ilqr_, 2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_);
    //(dF_d_state_with_history.rows() -3 , dF_d_state_with_history.cols()-3);
    result.block(0,0,2*h_dim_,2*h_dim_) = dF_d_state_with_history.block(0,0,2*h_dim_,2*h_dim_);
    for (int i = 0; i < num_state_component_ilqr_; i++) {
      result.block(2*h_dim_ + i, 0, 1, 2*h_dim_) = dF_d_state_with_history.block(2*h_dim_ + state_component_ilqr_index_[i], 0, 1, 2*h_dim_);
      result.block(0, 2*h_dim_ + i, 2*h_dim_, 1) = dF_d_state_with_history.block(0, 2*h_dim_ + state_component_ilqr_index_[i], 2*h_dim_, 1);
      result.block(2*h_dim_ +i, 2*h_dim_ + num_state_component_ilqr_, 1, acc_queue_size_ + steer_queue_size_) = dF_d_state_with_history.block(2*h_dim_ + state_component_ilqr_index_[i], 2*h_dim_ + state_size_, 1, acc_queue_size_ + steer_queue_size_);
      for (int j = 0; j < num_state_component_ilqr_; j++) {
        result(2*h_dim_ + i, 2*h_dim_ + j) = dF_d_state_with_history(2*h_dim_ + state_component_ilqr_index_[i], 2*h_dim_ + state_component_ilqr_index_[j]);
      }
    }
    //result.block(2*h_dim_,0,1,2*h_dim_) = dF_d_state_with_history.block(2*h_dim_+2,0,1,2*h_dim_);
    //result.block(2*h_dim_+1,0,dF_d_state_with_history.rows()-2*h_dim_ -3 -1,2*h_dim_) = dF_d_state_with_history.block(2*h_dim_+3+1,0,dF_d_state_with_history.rows()-2*h_dim_ -3 -1,2*h_dim_);
    
    //result.block(0,2*h_dim_,2*h_dim_,1) = dF_d_state_with_history.block(0,2*h_dim_+2,2*h_dim_,1);
    //result.block(2*h_dim_,2*h_dim_,1,1) = dF_d_state_with_history.block(2*h_dim_+2,2*h_dim_+2,1,1);
    //result.block(2*h_dim_+1,2*h_dim_,dF_d_state_with_history.rows()-2*h_dim_ -3 -1,1) = dF_d_state_with_history.block(2*h_dim_+3+1,2*h_dim_+2,dF_d_state_with_history.rows()-2*h_dim_ -3 -1,1);
    
    //result.block(0,2*h_dim_+1,2*h_dim_,dF_d_state_with_history.cols()-2*h_dim_-3-1) = dF_d_state_with_history.block(0,2*h_dim_+3+1,2*h_dim_,dF_d_state_with_history.cols()-2*h_dim_-3-1);
    //result.block(2*h_dim_,2*h_dim_+1,1,dF_d_state_with_history.cols()-2*h_dim_-3-1) = dF_d_state_with_history.block(2*h_dim_+2,2*h_dim_+3+1,1,dF_d_state_with_history.cols()-2*h_dim_-3-1);
    //result.block(2*h_dim_+1,2*h_dim_+1,dF_d_state_with_history.rows()-2*h_dim_ -3 -1,dF_d_state_with_history.cols()-2*h_dim_-3-1) = dF_d_state_with_history.block(2*h_dim_+3+1,2*h_dim_+3+1,dF_d_state_with_history.rows()-2*h_dim_ -3 -1,dF_d_state_with_history.cols()-2*h_dim_-3-1);
    
    return result;
  }
  Eigen::MatrixXd AdaptorILQR::extract_dF_d_input(Eigen::MatrixXd dF_d_input)
  {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(2*h_dim_ + num_state_component_ilqr_, 2);
    //(dF_d_input.rows() -3, 2);
    result.block(0,0,2*h_dim_,2) = dF_d_input.block(0,0,2*h_dim_,2);
    for (int i = 0; i < num_state_component_ilqr_; i++) {
      result.block(2*h_dim_ + i, 0, 1, 2) = dF_d_input.block(2*h_dim_ + state_component_ilqr_index_[i], 0, 1, 2);
    }

    //result.block(2*h_dim_,0,1,2) = dF_d_input.block(2*h_dim_+2,0,1,2);
    //result.block(2*h_dim_+1,0,dF_d_input.rows()-2*h_dim_ -3 -1,2) = dF_d_input.block(2*h_dim_+3+1,0,dF_d_input.rows()-2*h_dim_ -3 -1,2);
    return result;
  }

  Eigen::MatrixXd AdaptorILQR::right_action_by_state_diff_with_history(Eigen::MatrixXd Mat, Eigen::MatrixXd dF_d_state_with_history)
  {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Mat.rows(), dF_d_state_with_history.cols());
    result = Mat.leftCols(num_state_component_ilqr_ + 2*h_dim_)*dF_d_state_with_history;
    result.middleCols(num_state_component_ilqr_+2*h_dim_ + predict_step_, acc_queue_size_ - predict_step_) += Mat.middleCols(num_state_component_ilqr_ + 2*h_dim_, acc_queue_size_ - predict_step_);
    result.middleCols(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ + predict_step_, steer_queue_size_ - predict_step_) += Mat.middleCols(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_, steer_queue_size_ - predict_step_);
    for (int i = 0; i < predict_step_; i++) {
      result.col(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ - 1) += Mat.col(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ - predict_step_ + i);
      result.col(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_ - 1) += Mat.col(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_ - predict_step_ + i);
    }
    return result;
  }
  Eigen::MatrixXd AdaptorILQR::left_action_by_state_diff_with_history(Eigen::MatrixXd dF_d_state_with_history, Eigen::MatrixXd Mat)
  {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dF_d_state_with_history.cols(), Mat.cols());
    result = dF_d_state_with_history.transpose() * Mat.topRows(num_state_component_ilqr_ + 2*h_dim_);
    result.middleRows(num_state_component_ilqr_+2*h_dim_ + predict_step_, acc_queue_size_ - predict_step_) += Mat.middleRows(num_state_component_ilqr_ + 2*h_dim_, acc_queue_size_ - predict_step_);
    result.middleRows(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ + predict_step_, steer_queue_size_ - predict_step_) += Mat.middleRows(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_, steer_queue_size_ - predict_step_);
    for (int i = 0; i < predict_step_; i++) {
      result.row(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ - 1) += Mat.row(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ - predict_step_ + i);
      result.row(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_ - 1) += Mat.row(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_ - predict_step_ + i);
    }
    return result;
  }
  Eigen::MatrixXd AdaptorILQR::right_action_by_input_diff(Eigen::MatrixXd Mat, Eigen::MatrixXd dF_d_input)
  {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(Mat.rows(), dF_d_input.cols());
    result = Mat.leftCols(num_state_component_ilqr_ + 2*h_dim_)*dF_d_input;
    for (int i = 0; i < predict_step_; i++) {
      result.col(0) += (i+1) * Mat.col(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ - predict_step_ + i)*control_dt_;
      result.col(1) += (i+1) * Mat.col(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_ - predict_step_ + i)*control_dt_;
    }
    return result;
  }
  Eigen::MatrixXd AdaptorILQR::left_action_by_input_diff(Eigen::MatrixXd dF_d_input, Eigen::MatrixXd Mat)
  {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dF_d_input.cols(), Mat.cols());
    result = dF_d_input.transpose() * Mat.topRows(num_state_component_ilqr_ + 2*h_dim_);
    for (int i = 0; i < predict_step_; i++) {
      result.row(0) += (i+1) * Mat.row(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ - predict_step_ + i)*control_dt_;
      result.row(1) += (i+1) * Mat.row(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_ - predict_step_ + i)*control_dt_;
    }
    return result;
  }
  void AdaptorILQR::compute_ilqr_coefficients(
    const std::vector<Eigen::MatrixXd> & dF_d_states, const std::vector<Eigen::MatrixXd> & dF_d_inputs,
    const std::vector<Eigen::VectorXd> & states_prediction, const std::vector<Eigen::VectorXd> & d_inputs_schedule,
    const std::vector<Eigen::VectorXd> & states_ref,
    const std::vector<Eigen::VectorXd> & d_input_ref, const double prev_acc_rate, const double prev_steer_rate,
    std::vector<Eigen::Vector2d> inputs_ref, double acc_input_weight, double steer_input_weight, const std::vector<Eigen::Vector2d> & inputs_schedule,
    std::vector<Eigen::MatrixXd> & K, std::vector<Eigen::VectorXd> & k)
  {
    K = std::vector<Eigen::MatrixXd>(horizon_len_);
    k = std::vector<Eigen::VectorXd>(horizon_len_);
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_, num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ + steer_queue_size_);
    Eigen::VectorXd w = Eigen::VectorXd::Zero(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_);
    double yaw_target = states_ref[horizon_len_][yaw_index_];
    Eigen::Matrix2d rotation_matrix = Eigen::Rotation2Dd(-yaw_target).toRotationMatrix();
    Eigen::Matrix2d xy_cost_matrix = rotation_matrix.transpose() * Eigen::DiagonalMatrix<double, 2>(Eigen::Vector2d(x_terminal_cost_, y_terminal_cost_)) * rotation_matrix;
    Eigen::Vector2d xy_diff = Eigen::Vector2d(states_prediction[horizon_len_][x_index_] - states_ref[horizon_len_][x_index_], states_prediction[horizon_len_][y_index_] - states_ref[horizon_len_][y_index_]);
    for (int j = 0; j < num_state_component_ilqr_; j++) {
      if (state_component_ilqr_[j] == "x") {
        P.block(2*h_dim_ + j, 2*h_dim_ + j, 2, 2) = xy_cost_matrix;
        w.segment(2*h_dim_+j,2) = xy_cost_matrix * xy_diff;
      } else if (state_component_ilqr_[j] == "vel") {
        P(2*h_dim_ + j, 2*h_dim_ + j) = vel_terminal_cost_;
        w(2*h_dim_ + j) = vel_terminal_cost_ * (states_prediction[horizon_len_][vel_index_] - states_ref[horizon_len_][vel_index_]);
      } else if (state_component_ilqr_[j] == "yaw") {
        P(2*h_dim_ + j, 2*h_dim_ + j) = yaw_terminal_cost_;
        w(2*h_dim_ + j) = yaw_terminal_cost_ * (states_prediction[horizon_len_][yaw_index_] - states_ref[horizon_len_][yaw_index_]);
      } else if (state_component_ilqr_[j] == "acc") {
        P(2*h_dim_ + j, 2*h_dim_ + j) = acc_terminal_cost_;
        w(2*h_dim_ + j) = acc_terminal_cost_ * (states_prediction[horizon_len_][acc_index_] - states_ref[horizon_len_][acc_index_]);
      } else if (state_component_ilqr_[j] == "steer") {
        P(2*h_dim_ + j, 2*h_dim_ + j) = steer_terminal_cost_;
        w(2*h_dim_ + j) = steer_terminal_cost_ * (states_prediction[horizon_len_][steer_index_] - states_ref[horizon_len_][steer_index_]);
      } 
    }

    P(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_, 2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_) = acc_input_weight;
    P(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_, 2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_) = steer_input_weight;
    w(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_) = acc_input_weight * (inputs_schedule[horizon_len_-1][0] - inputs_ref[horizon_len_-1][0]);
    w(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_) = steer_input_weight * (inputs_schedule[horizon_len_-1][1] - inputs_ref[horizon_len_-1][1]);
    //P(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ - predict_step_, num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ - predict_step_) = acc_input_weight;
    //P(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_ - predict_step_, num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_ - predict_step_) = steer_input_weight;

    //P(2*h_dim_ + 1, 2*h_dim_ + 1) = acc_terminal_cost_;
    //P(2*h_dim_ + 2, 2*h_dim_ + 2) = steer_terminal_cost_;
    //w(2*h_dim_ + 1) = acc_terminal_cost_ * (states_prediction[horizon_len_][acc_index_] - states_ref[horizon_len_][acc_index_]);
    //w(2*h_dim_ + 2) = steer_terminal_cost_ * (states_prediction[horizon_len_][steer_index_] - states_ref[horizon_len_][steer_index_]);
    
    for (int i = horizon_len_ - 1; i >= 0; i--) {
      Eigen::MatrixXd Pi1Ai = right_action_by_state_diff_with_history(P, extract_dF_d_state(dF_d_states[i]));
      
      Eigen::MatrixXd G = left_action_by_input_diff(extract_dF_d_input(dF_d_inputs[i]), Pi1Ai);
   
      Eigen::MatrixXd H = left_action_by_input_diff(extract_dF_d_input(dF_d_inputs[i]), right_action_by_input_diff(P, extract_dF_d_input(dF_d_inputs[i])));
    
      H(0,0) += acc_rate_cost_;
      H(1,1) += steer_rate_cost_;
      if (i == 0) {
        H(0,0) += acc_rate_rate_cost_;
        H(1,1) += steer_rate_rate_cost_;
      }
      Eigen::MatrixXd invH = H.inverse();

      Eigen::VectorXd g_ = left_action_by_input_diff(extract_dF_d_input(dF_d_inputs[i]), w);
      g_(0) += acc_rate_cost_ * (d_inputs_schedule[i][0] - d_input_ref[i][0]);
      g_(1) += steer_rate_cost_ * (d_inputs_schedule[i][1] - d_input_ref[i][1]);
      if (i == 0) {
        g_(0) += acc_rate_rate_cost_ * (d_inputs_schedule[i][0] - prev_acc_rate);
        g_(1) += steer_rate_rate_cost_ * (d_inputs_schedule[i][1] - prev_steer_rate);
      }

      k[i] = invH * g_;
      K[i] = invH * G;
      P = left_action_by_state_diff_with_history(extract_dF_d_state(dF_d_states[i]), Pi1Ai) - G.transpose() * K[i];
      double x_cost = x_cost_;
      double y_cost = y_cost_;
      double vel_cost = vel_cost_;
      double yaw_cost = yaw_cost_;
      double acc_cost = acc_cost_;
      double steer_cost = steer_cost_;
      if (i == intermediate_cost_index_) {
        x_cost = x_intermediate_cost_;
        y_cost = y_intermediate_cost_;
        vel_cost = vel_intermediate_cost_;
        yaw_cost = yaw_intermediate_cost_;
        acc_cost = acc_intermediate_cost_;
        steer_cost = steer_intermediate_cost_;
      }
      w = left_action_by_state_diff_with_history(extract_dF_d_state(dF_d_states[i]), w);
      w -= G.transpose() * k[i];
      yaw_target = states_ref[i][yaw_index_];
      rotation_matrix = Eigen::Rotation2Dd(-yaw_target).toRotationMatrix();
      xy_cost_matrix = rotation_matrix.transpose() * Eigen::DiagonalMatrix<double, 2>(Eigen::Vector2d(x_cost, y_cost)) * rotation_matrix;
      xy_diff = Eigen::Vector2d(states_prediction[i][x_index_] - states_ref[i][x_index_], states_prediction[i][y_index_] - states_ref[i][y_index_]);
      for (int j = 0; j < num_state_component_ilqr_; j++){
        if (state_component_ilqr_[j] == "x") {
          P.block(2*h_dim_ + j, 2*h_dim_ + j, 2, 2) += xy_cost_matrix;
          w.segment(2*h_dim_ + j, 2) += xy_cost_matrix * xy_diff;
        } else if (state_component_ilqr_[j] == "vel") {
          P(2*h_dim_ + j, 2*h_dim_ + j) += vel_cost;
          w(2*h_dim_ + j) += vel_cost * (states_prediction[i][vel_index_] - states_ref[i][vel_index_]);
        } else if (state_component_ilqr_[j] == "yaw") {
          P(2*h_dim_ + j, 2*h_dim_ + j) += yaw_cost;
          w(2*h_dim_ + j) += yaw_cost * (states_prediction[i][yaw_index_] - states_ref[i][yaw_index_]);
        } else if (state_component_ilqr_[j] == "acc") {
          P(2*h_dim_ + j, 2*h_dim_ + j) += acc_cost;
          w(2*h_dim_ + j) += acc_cost * (states_prediction[i][acc_index_] - states_ref[i][acc_index_]);
        } else if (state_component_ilqr_[j] == "steer") {
          P(2*h_dim_ + j, 2*h_dim_ + j) += steer_cost;
          w(2*h_dim_ + j) += steer_cost * (states_prediction[i][steer_index_] - states_ref[i][steer_index_]);
        }
      }
      if (i>0){
        P(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_, 2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_) += acc_input_weight;
        P(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_, 2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_) += steer_input_weight;
        w(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_) += acc_input_weight * (inputs_schedule[i-1][0] - inputs_ref[i-1][0]);
        w(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_) += steer_input_weight * (inputs_schedule[i-1][1] - inputs_ref[i-1][1]);
      }
      //P(2*h_dim_ + 1, 2*h_dim_ + 1) += acc_cost;
      //P(2*h_dim_ + 2, 2*h_dim_ + 2) += steer_cost;
      //w(2*h_dim_ + 1) += acc_cost * (states_prediction[i][acc_index_] - states_ref[i][acc_index_]);
      //w(2*h_dim_ + 2) += steer_cost * (states_prediction[i][steer_index_] - states_ref[i][steer_index_]);
    }
  }
  std::vector<Eigen::MatrixXd> AdaptorILQR::calc_line_search_candidates(std::vector<Eigen::MatrixXd> K, std::vector<Eigen::VectorXd> k, std::vector<Eigen::MatrixXd> dF_d_states, std::vector<Eigen::MatrixXd> dF_d_inputs,  std::vector<Eigen::VectorXd> d_inputs_schedule, Eigen::VectorXd ls_points)
  {
    std::vector<Eigen::MatrixXd> D_inputs_schedule(horizon_len_);
    Eigen::MatrixXd D_states = Eigen::MatrixXd::Zero(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_, ls_points.size());
    for (int i = 0; i < horizon_len_; i++) {
      Eigen::MatrixXd D_D_inputs = - k[i]*ls_points.transpose() - K[i]*D_states;
      D_inputs_schedule[i] = D_D_inputs + d_inputs_schedule[i].replicate(1, ls_points.size());
      Eigen::MatrixXd D_states_temp = Eigen::MatrixXd::Zero(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_, ls_points.size());
      D_states_temp.topRows(num_state_component_ilqr_+2*h_dim_) += extract_dF_d_state(dF_d_states[i])* D_states + extract_dF_d_input(dF_d_inputs[i]) * D_D_inputs;
      D_states_temp.middleRows(num_state_component_ilqr_+2*h_dim_, acc_queue_size_ - predict_step_) += D_states.middleRows(num_state_component_ilqr_ + 2*h_dim_ + predict_step_, acc_queue_size_ - predict_step_);
      
      D_states_temp.middleRows(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_, steer_queue_size_ - predict_step_) += D_states.middleRows(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + predict_step_, steer_queue_size_ - predict_step_);
      
      for (int j = 0; j < predict_step_; j++) {
        D_states_temp.row(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ - predict_step_ + j) += D_states.row(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ - 1);
        
        D_states_temp.row(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ + steer_queue_size_ - predict_step_ + j) += D_states.row(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ + steer_queue_size_ - 1);
        
        D_states_temp.row(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ - predict_step_ + j) += (j+1)*control_dt_*D_D_inputs.row(0); 
      
        D_states_temp.row(num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ + steer_queue_size_ - predict_step_ + j) += (j+1)*control_dt_*D_D_inputs.row(1);
      }
      D_states = D_states_temp;
    }
    return D_inputs_schedule;
  }
  void AdaptorILQR::compute_optimal_control(Eigen::VectorXd states, Eigen::VectorXd acc_input_history,
                              Eigen::VectorXd steer_input_history, std::vector<Eigen::VectorXd> & d_inputs_schedule,
                              const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
                              const std::vector<Eigen::VectorXd> & states_ref,
                              const std::vector<Eigen::VectorXd> & d_input_ref, Eigen::VectorXd & previous_error,
                              std::vector<Eigen::VectorXd> & states_prediction,
                              const Eigen::VectorXd & acc_controller_input_schedule, const Eigen::VectorXd & steer_controller_input_schedule,
                              double & acc_input, double & steer_input)
  {
    std::vector<Eigen::Vector2d> inputs_ref;
    double acc_input_weight, steer_input_weight;
    calc_inputs_ref_info(
      states, acc_input_history, steer_input_history, h_lstm, c_lstm, previous_error, states_ref,
      acc_controller_input_schedule, steer_controller_input_schedule,
      inputs_ref, acc_input_weight, steer_input_weight);
    std::vector<Eigen::MatrixXd> dF_d_states, dF_d_inputs;
    std::vector<Eigen::Vector2d> inputs_schedule;
    trained_dynamics_.calc_forward_trajectory_with_diff(states, acc_input_history, steer_input_history, d_inputs_schedule,
                      h_lstm, c_lstm, previous_error, states_prediction, dF_d_states, dF_d_inputs, inputs_schedule);
    std::vector<Eigen::MatrixXd> K;
    std::vector<Eigen::VectorXd> k;
    double prev_acc_rate = (acc_input_history[acc_queue_size_ - 1] - acc_input_history[acc_queue_size_ - 2])/control_dt_;
    double prev_steer_rate = (steer_input_history[steer_queue_size_ - 1] - steer_input_history[steer_queue_size_ - 2])/control_dt_;
    compute_ilqr_coefficients(dF_d_states, dF_d_inputs, states_prediction, d_inputs_schedule, states_ref, d_input_ref, prev_acc_rate, prev_steer_rate,
    inputs_ref, acc_input_weight, steer_input_weight, inputs_schedule,K, k);
    Eigen::VectorXd ls_points = Eigen::VectorXd::LinSpaced(11, 0.0, 1.0);

    std::vector<Eigen::MatrixXd> D_inputs_schedule = calc_line_search_candidates(K, k, dF_d_states, dF_d_inputs, d_inputs_schedule, ls_points);
    Eigen::VectorXd Cost;
    std::vector<Eigen::MatrixXd> States_prediction;
    calc_forward_trajectory_with_cost(states, acc_input_history, steer_input_history, D_inputs_schedule, h_lstm, c_lstm, previous_error, States_prediction, states_ref, d_input_ref,
      inputs_ref, acc_input_weight, steer_input_weight, Cost);
    int min_index;
    Cost.minCoeff(&min_index);
    
    for (int i = 0; i < horizon_len_; i++) {
      d_inputs_schedule[i] = D_inputs_schedule[i].col(min_index);
      states_prediction[i+1] = States_prediction[i+1].col(min_index);
    }
    Eigen::VectorXd acc_input_history_copy = acc_input_history;
    Eigen::VectorXd steer_input_history_copy = steer_input_history;
    Eigen::VectorXd h_lstm_copy = h_lstm;
    Eigen::VectorXd c_lstm_copy = c_lstm;
    
    // update previous error
    trained_dynamics_.F_with_model(states, acc_input_history_copy, steer_input_history_copy, d_inputs_schedule[0], h_lstm_copy, c_lstm_copy, previous_error, 0);
    
    // get result
    acc_input = acc_input_history[acc_queue_size_ - 1] + d_inputs_schedule[0][0] * control_dt_;
    steer_input = steer_input_history[steer_queue_size_ - 1] + d_inputs_schedule[0][1] * control_dt_;
  
  }
  
///////////////// VehicleAdaptor ///////////////////////

  VehicleAdaptor::VehicleAdaptor(){
    set_params();
  }
  VehicleAdaptor::~VehicleAdaptor() {}
  void VehicleAdaptor::set_params()
  {
    adaptor_ilqr_.set_params();
    std::string param_dir_path = get_param_dir_path();
    YAML::Node nominal_param_node = YAML::LoadFile(param_dir_path + "/nominal_param.yaml");
    wheel_base_ = nominal_param_node["nominal_parameter"]["vehicle_info"]["wheel_base"].as<double>();
    acc_time_delay_ = nominal_param_node["nominal_parameter"]["acceleration"]["acc_time_delay"].as<double>();
    acc_time_constant_ = nominal_param_node["nominal_parameter"]["acceleration"]["acc_time_constant"].as<double>();
    steer_time_delay_ = nominal_param_node["nominal_parameter"]["steering"]["steer_time_delay"].as<double>();
    steer_time_constant_ = nominal_param_node["nominal_parameter"]["steering"]["steer_time_constant"].as<double>();


    YAML::Node optimization_param_node = YAML::LoadFile(param_dir_path + "/optimization_param.yaml"); 
    steer_dead_band_ = optimization_param_node["optimization_parameter"]["steering"]["steer_dead_band"].as<double>();
    x_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["x_cost"].as<double>();
    y_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["y_cost"].as<double>();
    vel_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["vel_cost"].as<double>();
    yaw_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["yaw_cost"].as<double>();
    acc_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_cost"].as<double>();
    steer_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_cost"].as<double>();
    
    acc_rate_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_rate_cost"].as<double>();
    steer_rate_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_rate_cost"].as<double>();
    steer_rate_cost_ /= (wheel_base_ * wheel_base_);

    double acc_rate_rate_cost = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_rate_rate_cost"].as<double>();
    double steer_rate_rate_cost = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_rate_rate_cost"].as<double>();
    acc_rate_rate_cost /= (wheel_base_ * wheel_base_);

  

    x_terminal_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["x_terminal_cost"].as<double>();
    y_terminal_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["y_terminal_cost"].as<double>();
    vel_terminal_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["vel_terminal_cost"].as<double>();
    yaw_terminal_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["yaw_terminal_cost"].as<double>();
    acc_terminal_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_terminal_cost"].as<double>();
    steer_terminal_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_terminal_cost"].as<double>();
    
    acc_rate_input_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_rate_input_table"].as<std::vector<double>>();
    steer_rate_input_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_rate_input_table"].as<std::vector<double>>();
    acc_rate_cost_coef_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_rate_cost_coef_table"].as<std::vector<double>>();
    steer_rate_cost_coef_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_rate_cost_coef_table"].as<std::vector<double>>();

    x_intermediate_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["x_intermediate_cost"].as<double>();
    y_intermediate_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["y_intermediate_cost"].as<double>();
    vel_intermediate_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["vel_intermediate_cost"].as<double>();
    yaw_intermediate_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["yaw_intermediate_cost"].as<double>();
    acc_intermediate_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_intermediate_cost"].as<double>();
    steer_intermediate_cost_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_intermediate_cost"].as<double>();
    intermediate_cost_index_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["intermediate_cost_index_predict_by_polynomial_regression"].as<int>();

    control_dt_ = optimization_param_node["optimization_parameter"]["setting"]["control_dt"].as<double>();
    predict_step_ = optimization_param_node["optimization_parameter"]["setting"]["predict_step"].as<int>();
    predict_dt_ = control_dt_ * predict_step_;
    horizon_len_ = optimization_param_node["optimization_parameter"]["setting"]["horizon_len_predict_by_polynomial_regression"].as<int>();

    controller_acc_input_history_len_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["controller_acc_input_history_len"].as<int>();
    controller_steer_input_history_len_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["controller_steer_input_history_len"].as<int>();
    deg_controller_acc_input_history_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["deg_controller_acc_input_history"].as<int>();
    deg_controller_steer_input_history_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["deg_controller_steer_input_history"].as<int>();
    lam_controller_acc_input_history_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["lam_controller_acc_input_history"].as<std::vector<double>>();
    lam_controller_steer_input_history_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["lam_controller_steer_input_history"].as<std::vector<double>>();
    minimum_decay_controller_acc_input_history_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["minimum_decay_controller_acc_input_history"].as<double>();
    minimum_decay_controller_steer_input_history_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["minimum_decay_controller_steer_input_history"].as<double>();

    use_sg_for_d_inputs_schedule_ = optimization_param_node["optimization_parameter"]["sg_filter"]["use_sg_for_d_inputs_schedule"].as<bool>();
    sg_window_size_for_d_inputs_schedule_ = optimization_param_node["optimization_parameter"]["sg_filter"]["sg_window_size_for_d_inputs_schedule"].as<int>();
    sg_deg_for_d_inputs_schedule_ = optimization_param_node["optimization_parameter"]["sg_filter"]["sg_deg_for_d_inputs_schedule"].as<int>();
    sg_window_size_for_NN_diff_ = optimization_param_node["optimization_parameter"]["sg_filter"]["sg_window_size_for_NN_diff"].as<int>();
    sg_deg_for_NN_diff_ = optimization_param_node["optimization_parameter"]["sg_filter"]["sg_deg_for_NN_diff"].as<int>();

    YAML::Node trained_model_param_node = YAML::LoadFile(param_dir_path + "/trained_model_param.yaml");
    acc_queue_size_ = trained_model_param_node["trained_model_parameter"]["queue_size"]["acc_queue_size"].as<int>();
    steer_queue_size_ = trained_model_param_node["trained_model_parameter"]["queue_size"]["steer_queue_size"].as<int>();
    update_lstm_len_ = trained_model_param_node["trained_model_parameter"]["lstm"]["update_lstm_len"].as<int>();
    nominal_dynamics_controller_.set_params(wheel_base_,acc_time_delay_,steer_time_delay_,acc_time_constant_,steer_time_constant_,acc_queue_size_,steer_queue_size_,control_dt_,predict_step_);
    nominal_dynamics_controller_.set_steer_dead_band(steer_dead_band_);

    
    compensation_lstm_len_ = optimization_param_node["optimization_parameter"]["compensation"]["compensation_lstm_len"].as<int>();

    mix_ratio_vel_target_table_ = optimization_param_node["optimization_parameter"]["mix_ratio"]["mix_ratio_vel_target_table"].as<std::vector<double>>();
    mix_ratio_vel_domain_table_ = optimization_param_node["optimization_parameter"]["mix_ratio"]["mix_ratio_vel_domain_table"].as<std::vector<double>>();
    mix_ratio_time_target_table_ = optimization_param_node["optimization_parameter"]["mix_ratio"]["mix_ratio_time_target_table"].as<std::vector<double>>();
    mix_ratio_time_domain_table_ = optimization_param_node["optimization_parameter"]["mix_ratio"]["mix_ratio_time_domain_table"].as<std::vector<double>>();

    adaptor_ilqr_.set_vehicle_params(
      wheel_base_, acc_time_delay_, steer_time_delay_, acc_time_constant_, steer_time_constant_,
      acc_queue_size_, steer_queue_size_, control_dt_, predict_step_);
    

    polynomial_reg_for_predict_acc_input_.set_params(
      deg_controller_acc_input_history_, controller_acc_input_history_len_, lam_controller_acc_input_history_);
    polynomial_reg_for_predict_acc_input_.set_minimum_decay(minimum_decay_controller_acc_input_history_);
    polynomial_reg_for_predict_acc_input_.set_ignore_intercept();
    polynomial_reg_for_predict_acc_input_.calc_coef_matrix();
    polynomial_reg_for_predict_acc_input_.calc_prediction_matrix(predict_step_*horizon_len_-1);
    polynomial_reg_for_predict_steer_input_.set_params(
      deg_controller_steer_input_history_, controller_steer_input_history_len_, lam_controller_steer_input_history_);
    polynomial_reg_for_predict_steer_input_.set_minimum_decay(minimum_decay_controller_steer_input_history_);
    polynomial_reg_for_predict_steer_input_.set_ignore_intercept();
    polynomial_reg_for_predict_steer_input_.calc_coef_matrix();
    polynomial_reg_for_predict_steer_input_.calc_prediction_matrix(predict_step_*horizon_len_-1);
    
 
    adaptor_ilqr_.set_states_cost(x_cost_,y_cost_,vel_cost_,yaw_cost_,acc_cost_,steer_cost_);
    adaptor_ilqr_.set_terminal_cost(x_terminal_cost_,y_terminal_cost_, vel_terminal_cost_,yaw_terminal_cost_,acc_terminal_cost_,steer_terminal_cost_);
    adaptor_ilqr_.set_intermediate_cost(x_intermediate_cost_,y_intermediate_cost_,vel_intermediate_cost_,yaw_intermediate_cost_,acc_intermediate_cost_,steer_intermediate_cost_,intermediate_cost_index_);
    adaptor_ilqr_.set_rate_cost(acc_rate_rate_cost,steer_rate_rate_cost);

    reflect_controller_d_input_ratio_ = optimization_param_node["optimization_parameter"]["ilqr"]["reflect_controller_d_input_ratio"].as<double>();
    use_controller_inputs_as_target_ = optimization_param_node["optimization_parameter"]["ilqr"]["use_controller_inputs_as_target"].as<bool>();
    int deg_for_acc_inputs_polynomial_filter = optimization_param_node["optimization_parameter"]["input_filter"]["deg_for_acc_inputs_polynomial_filter"].as<int>();
    int sample_num_for_acc_inputs_polynomial_filter = optimization_param_node["optimization_parameter"]["input_filter"]["sample_num_for_acc_inputs_polynomial_filter"].as<int>();
    double lam_for_acc_inputs_polynomial_filter = optimization_param_node["optimization_parameter"]["input_filter"]["lam_for_acc_inputs_polynomial_filter"].as<double>();
    double minimum_decay_for_acc_inputs_polynomial_filter = optimization_param_node["optimization_parameter"]["input_filter"]["minimum_decay_for_acc_inputs_polynomial_filter"].as<double>();
    int deg_for_steer_inputs_polynomial_filter = optimization_param_node["optimization_parameter"]["input_filter"]["deg_for_steer_inputs_polynomial_filter"].as<int>();
    int sample_num_for_steer_inputs_polynomial_filter = optimization_param_node["optimization_parameter"]["input_filter"]["sample_num_for_steer_inputs_polynomial_filter"].as<int>();
    double lam_for_steer_inputs_polynomial_filter = optimization_param_node["optimization_parameter"]["input_filter"]["lam_for_steer_inputs_polynomial_filter"].as<double>();
    double minimum_decay_for_steer_inputs_polynomial_filter = optimization_param_node["optimization_parameter"]["input_filter"]["minimum_decay_for_steer_inputs_polynomial_filter"].as<double>();
    polynomial_filter_for_acc_inputs_.set_params(deg_for_acc_inputs_polynomial_filter,
    sample_num_for_acc_inputs_polynomial_filter,
    lam_for_acc_inputs_polynomial_filter,
    minimum_decay_for_acc_inputs_polynomial_filter);
    polynomial_filter_for_steer_inputs_.set_params(deg_for_steer_inputs_polynomial_filter,
    sample_num_for_steer_inputs_polynomial_filter,
    lam_for_steer_inputs_polynomial_filter,
    minimum_decay_for_steer_inputs_polynomial_filter);
    input_filter_mode_ = optimization_param_node["optimization_parameter"]["input_filter"]["input_filter_mode"].as<std::string>();
    if (input_filter_mode_ == "butterworth"){
      butterworth_filter_.set_params();
    }
    use_controller_steer_input_schedule_ = optimization_param_node["optimization_parameter"]["autoware_alignment"]["use_controller_steer_input_schedule"].as<bool>();

    use_inputs_schedule_prediction_ = optimization_param_node["optimization_parameter"]["inputs_schedule_prediction_NN"]["use_inputs_schedule_prediction"].as<bool>();
    if (use_inputs_schedule_prediction_){
      inputs_schedule_prediction_.set_params(horizon_len_*predict_step_, horizon_len_ * predict_step_ -1 , control_dt_, "inputs_schedule_prediction_model");
    }
  }
  void VehicleAdaptor::set_NN_params(
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias)
  {
    adaptor_ilqr_.set_NN_params(
      weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2,
      weight_lstm_ih, weight_lstm_hh, weight_complimentary_layer, weight_linear_relu,
      weight_final_layer, bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2, bias_steer_layer_2,
      bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer,
      bias_linear_relu, bias_final_layer, vel_scaling, vel_bias);
    h_dim_ = weight_lstm_hh.cols();
    previous_error_ = Eigen::VectorXd::Zero(bias_final_layer.size());
  }
  void VehicleAdaptor::set_NN_params_from_csv(std::string csv_dir){
    Eigen::MatrixXd weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2, weight_lstm_ih, weight_lstm_hh, weight_complimentary_layer, weight_linear_relu, weight_final_layer;
    Eigen::VectorXd bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2, bias_steer_layer_2, bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer, bias_linear_relu, bias_final_layer;
    double vel_scaling, vel_bias;
    weight_acc_layer_1 = read_csv(csv_dir + "/weight_acc_layer_1.csv");
    weight_steer_layer_1 = read_csv(csv_dir + "/weight_steer_layer_1.csv");
    weight_acc_layer_2 = read_csv(csv_dir + "/weight_acc_layer_2.csv");
    weight_steer_layer_2 = read_csv(csv_dir + "/weight_steer_layer_2.csv");
    weight_lstm_ih = read_csv(csv_dir + "/weight_lstm_ih.csv");
    weight_lstm_hh = read_csv(csv_dir + "/weight_lstm_hh.csv");
    weight_complimentary_layer = read_csv(csv_dir + "/weight_complimentary_layer.csv");
    weight_linear_relu = read_csv(csv_dir + "/weight_linear_relu.csv");
    weight_final_layer = read_csv(csv_dir + "/weight_final_layer.csv");
    bias_acc_layer_1 = read_csv(csv_dir + "/bias_acc_layer_1.csv").col(0);
    bias_steer_layer_1 = read_csv(csv_dir + "/bias_steer_layer_1.csv").col(0);
    bias_acc_layer_2 = read_csv(csv_dir + "/bias_acc_layer_2.csv").col(0);
    bias_steer_layer_2 = read_csv(csv_dir + "/bias_steer_layer_2.csv").col(0);
    bias_lstm_ih = read_csv(csv_dir + "/bias_lstm_ih.csv").col(0);
    bias_lstm_hh = read_csv(csv_dir + "/bias_lstm_hh.csv").col(0);
    bias_complimentary_layer = read_csv(csv_dir + "/bias_complimentary_layer.csv").col(0);
    bias_linear_relu = read_csv(csv_dir + "/bias_linear_relu.csv").col(0);
    bias_final_layer = read_csv(csv_dir + "/bias_final_layer.csv").col(0);

    Eigen::MatrixXd vel_params = read_csv(csv_dir + "/vel_scale.csv");

    vel_scaling = vel_params(0,0);
    vel_bias = vel_params(1,0);
    set_NN_params(
      weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2,
      weight_lstm_ih, weight_lstm_hh, weight_complimentary_layer, weight_linear_relu,
      weight_final_layer, bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2, bias_steer_layer_2,
      bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer,
      bias_linear_relu, bias_final_layer, vel_scaling, vel_bias);
  }
  void VehicleAdaptor::clear_NN_params()
  {
    adaptor_ilqr_.clear_NN_params();
  }
  void VehicleAdaptor::set_controller_d_inputs_schedule(const Eigen::VectorXd & acc_controller_d_inputs_schedule, const Eigen::VectorXd & steer_controller_d_inputs_schedule)
  {
    acc_controller_d_inputs_schedule_ = acc_controller_d_inputs_schedule;
    steer_controller_d_inputs_schedule_ = steer_controller_d_inputs_schedule;
    if (!initialized_)
    { 
      set_params();
      states_ref_mode_ = "controller_d_inputs_schedule";
      YAML::Node optimization_param_node = YAML::LoadFile(get_param_dir_path() + "/optimization_param.yaml"); 
      horizon_len_ = optimization_param_node["optimization_parameter"]["setting"]["horizon_len_controller_d_inputs_schedule"].as<int>();
      horizon_len_ = std::min(horizon_len_, int(acc_controller_d_inputs_schedule.size()));
      horizon_len_ = std::min(horizon_len_, int(steer_controller_d_inputs_schedule.size()));
      intermediate_cost_index_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["intermediate_cost_index_controller_d_inputs_schedule"].as<int>();
      intermediate_cost_index_ = std::min(intermediate_cost_index_, horizon_len_-1);
      adaptor_ilqr_.set_intermediate_cost(x_intermediate_cost_,y_intermediate_cost_,vel_intermediate_cost_,yaw_intermediate_cost_,acc_intermediate_cost_,steer_intermediate_cost_,intermediate_cost_index_);
    }
  }
  void VehicleAdaptor::set_controller_d_steer_schedule(const Eigen::VectorXd & steer_controller_d_inputs_schedule)
  {
    steer_controller_d_inputs_schedule_ = steer_controller_d_inputs_schedule;
    if (!initialized_)
    {
      set_params();
      states_ref_mode_ = "controller_d_steer_schedule";
    }
  }
  void VehicleAdaptor::set_controller_steer_input_schedule(double timestamp, const std::vector<double> & steer_controller_input_schedule)
  {
    if (int(steer_controller_input_schedule.size()) < horizon_len_ + 1)
    {
      std::cerr << "steer_controller_input_schedule size is smaller than horizon_len" << std::endl;
      return;
    }
    Eigen::VectorXd steer_schedule(steer_controller_input_schedule.size());
    std::vector<double> timestamp_horizon(steer_controller_input_schedule.size());
    std::vector<double> timestamp_new(steer_controller_input_schedule.size() - 1);
    for (int i = 0; i < int(steer_controller_input_schedule.size()); i++)
    {
      steer_schedule[i] = steer_controller_input_schedule[i];
      timestamp_horizon[i] = i * predict_dt_;
      if (i < int(steer_controller_input_schedule.size()) - 1)
      {
        timestamp_new[i] = (i + 1) * predict_dt_ -  control_dt_;
      }
    }
    Eigen::VectorXd steer_controller_input_schedule_interpolated = interpolate_eigen(steer_schedule, timestamp_horizon, timestamp_new);
    Eigen::VectorXd steer_controller_d_inputs_schedule(steer_controller_input_schedule_interpolated.size());
    for (int i = 0; i < int(steer_controller_input_schedule_interpolated.size()) - 1; i++)
    {
      steer_controller_d_inputs_schedule[i + 1] = (steer_controller_input_schedule_interpolated[i+1] - steer_controller_input_schedule_interpolated[i]) / predict_dt_;
    }
    if (!initialized_)
    {
      steer_controller_d_inputs_schedule[0] = (steer_controller_input_schedule_interpolated[0] - steer_controller_input_schedule[0]) / (predict_dt_ - control_dt_);
    }
    else{
      double prev_timestamp = time_stamp_obs_[time_stamp_obs_.size()-1];
      double prev_steer_controller_input = steer_controller_input_history_obs_[steer_controller_input_history_obs_.size()-1];
      steer_controller_d_inputs_schedule[0] = (steer_controller_input_schedule_interpolated[0] - prev_steer_controller_input) / (predict_dt_ - control_dt_ + timestamp - prev_timestamp);
    }
    set_controller_d_steer_schedule(steer_controller_d_inputs_schedule);

  }
  void VehicleAdaptor::set_controller_prediction(const Eigen::VectorXd & x_controller_prediction, const Eigen::VectorXd & y_controller_prediction, const Eigen::VectorXd & vel_controller_prediction, const Eigen::VectorXd & yaw_controller_prediction, const Eigen::VectorXd & acc_controller_prediction, const Eigen::VectorXd & steer_controller_prediction)
  {
    x_controller_prediction_ = x_controller_prediction;
    y_controller_prediction_ = y_controller_prediction;
    vel_controller_prediction_ = vel_controller_prediction;
    yaw_controller_prediction_ = yaw_controller_prediction;
    acc_controller_prediction_ = acc_controller_prediction;
    steer_controller_prediction_ = steer_controller_prediction;
    if (!initialized_)
    {
      set_params();
      states_ref_mode_ = "controller_prediction";
      YAML::Node optimization_param_node = YAML::LoadFile(get_param_dir_path() + "/optimization_param.yaml"); 
      horizon_len_ = optimization_param_node["optimization_parameter"]["setting"]["horizon_len_controller_prediction"].as<int>();
      horizon_len_ = std::min(horizon_len_, int(acc_controller_prediction.size())-1);
      horizon_len_ = std::min(horizon_len_, int(steer_controller_prediction.size())-1);
      intermediate_cost_index_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["intermediate_cost_index_controller_prediction"].as<int>();
      intermediate_cost_index_ = std::min(intermediate_cost_index_, horizon_len_-1);
      adaptor_ilqr_.set_intermediate_cost(x_intermediate_cost_,y_intermediate_cost_,vel_intermediate_cost_,yaw_intermediate_cost_,acc_intermediate_cost_,steer_intermediate_cost_,intermediate_cost_index_);
    }
  }
  void VehicleAdaptor::set_controller_steer_prediction(const Eigen::VectorXd & steer_controller_prediction)
  {
    steer_controller_prediction_ = steer_controller_prediction;
    steer_controller_prediction_aided_ = true;
  }

  Eigen::VectorXd VehicleAdaptor::get_adjusted_inputs(
    double time_stamp, const Eigen::VectorXd & states, const double acc_controller_input, const double steer_controller_input)
  {
    if (!initialized_) {
      start_time_ = time_stamp;
      if (states_ref_mode_ == "predict_by_polynomial_regression"){
        set_params();
      }
      adaptor_ilqr_.initialize_compensation();

      adaptor_ilqr_.set_sg_filter_params(sg_deg_for_NN_diff_,horizon_len_,sg_window_size_for_NN_diff_);
      max_queue_size_ = std::max({acc_queue_size_ + 1, steer_queue_size_ + 1, controller_acc_input_history_len_, controller_steer_input_history_len_, predict_step_*update_lstm_len_+2});
      time_stamp_obs_ = std::vector<double>(max_queue_size_);
      for (int i = 0; i < max_queue_size_; i++)
      {
        time_stamp_obs_[i] = time_stamp - (max_queue_size_ - i - 1) * control_dt_;
      }


      acc_input_history_ = acc_controller_input * Eigen::VectorXd::Ones(acc_queue_size_);
      steer_input_history_ = steer_controller_input * Eigen::VectorXd::Ones(steer_queue_size_);
      acc_controller_input_history_ = acc_controller_input * Eigen::VectorXd::Ones(controller_acc_input_history_len_);
      steer_controller_input_history_ = steer_controller_input * Eigen::VectorXd::Ones(controller_steer_input_history_len_);
      d_inputs_schedule_ = std::vector<Eigen::VectorXd>(horizon_len_, Eigen::VectorXd::Zero(2));

      state_history_lstm_ = std::vector<Eigen::VectorXd>(predict_step_*update_lstm_len_, states);
      
      acc_input_history_lstm_ = std::vector<Eigen::VectorXd>(predict_step_*update_lstm_len_+1, acc_input_history_);
      steer_input_history_lstm_ = std::vector<Eigen::VectorXd>(predict_step_*update_lstm_len_+1, steer_input_history_);

      acc_input_history_obs_ = std::vector<double>(max_queue_size_-1, acc_controller_input);
      steer_input_history_obs_ = std::vector<double>(max_queue_size_-1, steer_controller_input);
      acc_controller_input_history_obs_ = std::vector<double>(max_queue_size_, acc_controller_input);
      steer_controller_input_history_obs_ = std::vector<double>(max_queue_size_, steer_controller_input);
      state_history_lstm_obs_ = std::vector<Eigen::VectorXd>(max_queue_size_-1, states);
      acc_input_history_lstm_obs_ = std::vector<Eigen::VectorXd>(max_queue_size_-1, acc_input_history_);
      steer_input_history_lstm_obs_ = std::vector<Eigen::VectorXd>(max_queue_size_-1, steer_input_history_);

      sg_window_size_for_d_inputs_schedule_ = std::min(sg_window_size_for_d_inputs_schedule_, int(horizon_len_/2)-1);
      sg_filter_for_d_inputs_schedule_.set_params(sg_deg_for_d_inputs_schedule_,sg_window_size_for_d_inputs_schedule_);
      sg_filter_for_d_inputs_schedule_.calc_sg_filter_weight();

      initialized_ = true;
    }
    else{//already initialized case

      std::vector<double> time_stamp_acc_input_history(acc_queue_size_);
      for (int i = 0; i < acc_queue_size_; i++)
      {
        time_stamp_acc_input_history[i] = time_stamp - (acc_queue_size_ - i) * control_dt_;
      }
      std::vector<double> time_stamp_steer_input_history(steer_queue_size_); 
      for (int i = 0; i < steer_queue_size_; i++)
      {
        time_stamp_steer_input_history[i] = time_stamp - (steer_queue_size_ - i) * control_dt_;
      }
      std::vector<double> time_stamp_controller_acc_input_history(controller_acc_input_history_len_);
      for (int i = 0; i < controller_acc_input_history_len_; i++)
      {
        time_stamp_controller_acc_input_history[i] = time_stamp - (controller_acc_input_history_len_ - i - 1) * control_dt_;
      }
      std::vector<double> time_stamp_controller_steer_input_history(controller_steer_input_history_len_);
      for (int i = 0; i < controller_steer_input_history_len_; i++)
      {
        time_stamp_controller_steer_input_history[i] = time_stamp - (controller_steer_input_history_len_ - i - 1) * control_dt_;
      }
      std::vector<double> time_stamp_state_history_lstm(predict_step_*update_lstm_len_);
      for (int i = 0; i < predict_step_*update_lstm_len_; i++)
      {
        time_stamp_state_history_lstm[i] = time_stamp - (predict_step_*update_lstm_len_ - i) * control_dt_;
      }
      std::vector<double> time_stamp_input_history_lstm(predict_step_*update_lstm_len_+1);
      for (int i = 0; i < predict_step_*update_lstm_len_ + 1; i++)
      {
        time_stamp_input_history_lstm[i] = time_stamp - (predict_step_*update_lstm_len_ + 1 - i) * control_dt_;
      }
      Eigen::Map<Eigen::VectorXd> acc_input_history_obs_eigen(acc_input_history_obs_.data(), acc_input_history_obs_.size());
      Eigen::Map<Eigen::VectorXd> steer_input_history_obs_eigen(steer_input_history_obs_.data(), steer_input_history_obs_.size());

      acc_input_history_ = interpolate_eigen(acc_input_history_obs_eigen, time_stamp_obs_, time_stamp_acc_input_history);
      steer_input_history_ = interpolate_eigen(steer_input_history_obs_eigen, time_stamp_obs_, time_stamp_steer_input_history);
      

      state_history_lstm_ = interpolate_vector(state_history_lstm_obs_, time_stamp_obs_, time_stamp_state_history_lstm);
      acc_input_history_lstm_ = interpolate_vector(acc_input_history_lstm_obs_, time_stamp_obs_, time_stamp_input_history_lstm);
      steer_input_history_lstm_ = interpolate_vector(steer_input_history_lstm_obs_, time_stamp_obs_, time_stamp_input_history_lstm);

      time_stamp_obs_.push_back(time_stamp);
      acc_controller_input_history_obs_.push_back(acc_controller_input);
      steer_controller_input_history_obs_.push_back(steer_controller_input);

      if (time_stamp - time_stamp_obs_[1] > max_queue_size_ * control_dt_)
      {
        time_stamp_obs_.erase(time_stamp_obs_.begin());
        acc_input_history_obs_.erase(acc_input_history_obs_.begin());
        steer_input_history_obs_.erase(steer_input_history_obs_.begin());
        acc_controller_input_history_obs_.erase(acc_controller_input_history_obs_.begin());
        steer_controller_input_history_obs_.erase(steer_controller_input_history_obs_.begin());
        state_history_lstm_obs_.erase(state_history_lstm_obs_.begin());
        acc_input_history_lstm_obs_.erase(acc_input_history_lstm_obs_.begin());
        steer_input_history_lstm_obs_.erase(steer_input_history_lstm_obs_.begin());
      }

      Eigen::Map<Eigen::VectorXd> acc_controller_input_history_obs_eigen(acc_controller_input_history_obs_.data(), acc_controller_input_history_obs_.size());
      Eigen::Map<Eigen::VectorXd> steer_controller_input_history_obs_eigen(steer_controller_input_history_obs_.data(), steer_controller_input_history_obs_.size());

      //acc_controller_input_history_.head(controller_acc_input_history_len_ - 1) = acc_controller_input_history_.tail(controller_acc_input_history_len_ - 1);
      //acc_controller_input_history_(controller_acc_input_history_len_ - 1) = acc_controller_input;
      acc_controller_input_history_ = interpolate_eigen(acc_controller_input_history_obs_eigen, time_stamp_obs_, time_stamp_controller_acc_input_history);

      //steer_controller_input_history_.head(controller_steer_input_history_len_ - 1) = steer_controller_input_history_.tail(controller_steer_input_history_len_ - 1);
      //steer_controller_input_history_(controller_steer_input_history_len_ - 1) = steer_controller_input;
      steer_controller_input_history_ = interpolate_eigen(steer_controller_input_history_obs_eigen, time_stamp_obs_, time_stamp_controller_steer_input_history);


      if (use_sg_for_d_inputs_schedule_){
        d_inputs_schedule_ = sg_filter_for_d_inputs_schedule_.sg_filter(d_inputs_schedule_);
      }
      //state_history_lstm_.erase(state_history_lstm_.begin());
      //state_history_lstm_.push_back(states);
    }

    // update lstm states //
    Eigen::VectorXd h_lstm = Eigen::VectorXd::Zero(h_dim_);
    Eigen::VectorXd c_lstm = Eigen::VectorXd::Zero(h_dim_);
    Eigen::VectorXd h_lstm_compensation = Eigen::VectorXd::Zero(h_dim_);
    Eigen::VectorXd c_lstm_compensation = Eigen::VectorXd::Zero(h_dim_);
    for (int i = 0; i<update_lstm_len_; i++) {

      Eigen::VectorXd states_tmp = state_history_lstm_[predict_step_*i];
      Eigen::VectorXd acc_input_history_concat = Eigen::VectorXd::Zero(acc_queue_size_ + predict_step_);
      Eigen::VectorXd steer_input_history_concat = Eigen::VectorXd::Zero(steer_queue_size_ + predict_step_);
      acc_input_history_concat.head(acc_queue_size_) = acc_input_history_lstm_[predict_step_*i];
      steer_input_history_concat.head(steer_queue_size_) = steer_input_history_lstm_[predict_step_*i];


      for (int j = 0; j < predict_step_; j++) {
        acc_input_history_concat[acc_queue_size_ + j] = acc_input_history_lstm_[predict_step_*i + j + 1][acc_queue_size_ - 1];
        steer_input_history_concat[steer_queue_size_ + j] = steer_input_history_lstm_[predict_step_*i + j + 1][steer_queue_size_ - 1];
      }
      if (i == update_lstm_len_ - compensation_lstm_len_){
        h_lstm_compensation = h_lstm;
        c_lstm_compensation = c_lstm;
      }
      adaptor_ilqr_.update_lstm_states(states_tmp, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
    }

///////// calculate states_ref /////////
    double acc_rate_cost = acc_rate_cost_;
    double steer_rate_cost = steer_rate_cost_;
    std::vector<Eigen::VectorXd> states_ref(horizon_len_ + 1);

    // calculate controller input history with schedule //
    Eigen::VectorXd acc_controller_input_history_with_schedule = Eigen::VectorXd::Zero(controller_acc_input_history_len_+predict_step_*horizon_len_-1);
    Eigen::VectorXd steer_controller_input_history_with_schedule = Eigen::VectorXd::Zero(controller_steer_input_history_len_+predict_step_*horizon_len_-1);
    acc_controller_input_history_with_schedule.head(controller_acc_input_history_len_) = acc_controller_input_history_;
    steer_controller_input_history_with_schedule.head(controller_steer_input_history_len_) = steer_controller_input_history_;

    Eigen::VectorXd acc_controller_inputs_prediction_by_NN(horizon_len_*predict_step_-1);
    Eigen::VectorXd steer_controller_inputs_prediction_by_NN(horizon_len_*predict_step_-1);
    Eigen::VectorXd inputs_schedule_predictor_states(5);
    inputs_schedule_predictor_states << states[vel_index_], states[acc_index_], states[steer_index_], acc_controller_input, steer_controller_input;
    if (use_inputs_schedule_prediction_)
    {
      std::vector<Eigen::VectorXd> controller_inputs_prediction_by_NN = inputs_schedule_prediction_.get_inputs_schedule_predicted(inputs_schedule_predictor_states, time_stamp);
      for (int i = 0; i < horizon_len_ * predict_step_ - 1; i++)
      {
        acc_controller_inputs_prediction_by_NN[i] = controller_inputs_prediction_by_NN[i][0];
        steer_controller_inputs_prediction_by_NN[i] = controller_inputs_prediction_by_NN[i][1];
      }
    }
    



    if (states_ref_mode_ == "controller_d_steer_schedule")
    {
      for (int i = 0; i<int(d_inputs_schedule_.size()); i++){
        d_inputs_schedule_[i][1] = (1 - reflect_controller_d_input_ratio_)*d_inputs_schedule_[i][1];
        d_inputs_schedule_[i][1] += reflect_controller_d_input_ratio_*steer_controller_d_inputs_schedule_[i];
      }
      Eigen::VectorXd acc_controller_input_history_diff = (acc_controller_input_history_.tail(acc_controller_input_history_.size() - 1) - acc_controller_input_history_.head(acc_controller_input_history_.size() - 1))/control_dt_;
      Eigen::VectorXd abs_acc_controller_input_history_diff = acc_controller_input_history_diff.cwiseAbs();
      Eigen::VectorXd abs_steer_controller_d_inputs_schedule = steer_controller_d_inputs_schedule_.head(horizon_len_).cwiseAbs();

      acc_rate_cost = calc_table_value(abs_acc_controller_input_history_diff.maxCoeff(), acc_rate_input_table_, acc_rate_cost_coef_table_)*acc_rate_cost_;
      steer_rate_cost = calc_table_value(abs_steer_controller_d_inputs_schedule.maxCoeff(), steer_rate_input_table_, steer_rate_cost_coef_table_)*steer_rate_cost_;

      Eigen::VectorXd acc_controller_input_prediction;
      if (use_inputs_schedule_prediction_)
      {
        acc_controller_input_prediction = acc_controller_inputs_prediction_by_NN; 
      }
      else{
        acc_controller_input_prediction = polynomial_reg_for_predict_acc_input_.predict(acc_controller_input_history_);
      }
      acc_controller_input_history_with_schedule.tail(predict_step_*horizon_len_ - 1) = acc_controller_input_prediction;
      for (int i = 0; i< horizon_len_; i++) {
        for (int j = 0; j< predict_step_; j++) {
          if (i != 0 || j != 0)
          {
            steer_controller_input_history_with_schedule[controller_steer_input_history_len_ - 1 + i*predict_step_ + j] = steer_controller_input_history_with_schedule[controller_steer_input_history_len_ - 1 + i*predict_step_ + j -1] + control_dt_ * steer_controller_d_inputs_schedule_[i];
          }
        }
      }
    }
    else if (states_ref_mode_ == "controller_d_inputs_schedule")
    {
      for (int i = 0; i<int(d_inputs_schedule_.size()); i++){
        d_inputs_schedule_[i] = (1 - reflect_controller_d_input_ratio_)*d_inputs_schedule_[i];
        d_inputs_schedule_[i][0] += reflect_controller_d_input_ratio_*acc_controller_d_inputs_schedule_[i];
        d_inputs_schedule_[i][1] += reflect_controller_d_input_ratio_*steer_controller_d_inputs_schedule_[i];
      }
      Eigen::VectorXd abs_acc_controller_d_inputs_schedule = acc_controller_d_inputs_schedule_.head(horizon_len_).cwiseAbs();
      Eigen::VectorXd abs_steer_controller_d_inputs_schedule = steer_controller_d_inputs_schedule_.head(horizon_len_).cwiseAbs();
      acc_rate_cost = calc_table_value(abs_acc_controller_d_inputs_schedule.maxCoeff(), acc_rate_input_table_, acc_rate_cost_coef_table_)*acc_rate_cost_;
      steer_rate_cost = calc_table_value(abs_steer_controller_d_inputs_schedule.maxCoeff(), steer_rate_input_table_, steer_rate_cost_coef_table_)*steer_rate_cost_;
      for (int i = 0; i< horizon_len_; i++) {
        for (int j = 0; j< predict_step_; j++) {
          if (i != 0 || j != 0)
          {
            acc_controller_input_history_with_schedule[controller_acc_input_history_len_ - 1 + i*predict_step_ + j] = acc_controller_input_history_with_schedule[controller_acc_input_history_len_ - 1 + i*predict_step_ + j -1] + control_dt_ * acc_controller_d_inputs_schedule_[i];
            steer_controller_input_history_with_schedule[controller_steer_input_history_len_ - 1 + i*predict_step_ + j] = steer_controller_input_history_with_schedule[controller_steer_input_history_len_ - 1 + i*predict_step_ + j -1] + control_dt_ * steer_controller_d_inputs_schedule_[i];
          }
        }
      }
    }
    else
    {
      if (states_ref_mode_ == "predict_by_polynomial_regression")
      {
        Eigen::VectorXd acc_controller_input_history_diff = (acc_controller_input_history_.tail(acc_controller_input_history_.size() - 1) - acc_controller_input_history_.head(acc_controller_input_history_.size() - 1))/control_dt_;
        Eigen::VectorXd steer_controller_input_history_diff = (steer_controller_input_history_.tail(steer_controller_input_history_.size() - 1) - steer_controller_input_history_.head(steer_controller_input_history_.size() - 1))/control_dt_;
        Eigen::VectorXd abs_acc_controller_input_history_diff = acc_controller_input_history_diff.cwiseAbs();
        Eigen::VectorXd abs_steer_controller_input_history_diff = steer_controller_input_history_diff.cwiseAbs();
        acc_rate_cost = calc_table_value(abs_acc_controller_input_history_diff.maxCoeff(), acc_rate_input_table_, acc_rate_cost_coef_table_)*acc_rate_cost_;
        steer_rate_cost = calc_table_value(abs_steer_controller_input_history_diff.maxCoeff(), steer_rate_input_table_, steer_rate_cost_coef_table_)*steer_rate_cost_;
      }
      else{
        Eigen::VectorXd acc_controller_prediction_diff = (acc_controller_prediction_.tail(acc_controller_prediction_.size() - 1) - acc_controller_prediction_.head(acc_controller_prediction_.size() - 1))/predict_dt_;
        Eigen::VectorXd steer_controller_prediction_diff = (steer_controller_prediction_.tail(steer_controller_prediction_.size() - 1) - steer_controller_prediction_.head(steer_controller_prediction_.size() - 1))/predict_dt_;
        Eigen::VectorXd abs_acc_controller_prediction_diff = acc_controller_prediction_diff.head(horizon_len_).cwiseAbs();
        Eigen::VectorXd abs_steer_controller_prediction_diff = steer_controller_prediction_diff.head(horizon_len_).cwiseAbs();
        acc_rate_cost = calc_table_value(abs_acc_controller_prediction_diff.maxCoeff(), acc_rate_input_table_, acc_rate_cost_coef_table_)*acc_rate_cost_;
        steer_rate_cost = calc_table_value(abs_steer_controller_prediction_diff.maxCoeff(), steer_rate_input_table_, steer_rate_cost_coef_table_)*steer_rate_cost_;
      }
      Eigen::VectorXd acc_controller_input_prediction, steer_controller_input_prediction;
      if (use_inputs_schedule_prediction_)
      {
        acc_controller_input_prediction = acc_controller_inputs_prediction_by_NN; 
        steer_controller_input_prediction = steer_controller_inputs_prediction_by_NN;
      }
      else{
        acc_controller_input_prediction = polynomial_reg_for_predict_acc_input_.predict(acc_controller_input_history_);
        steer_controller_input_prediction = polynomial_reg_for_predict_steer_input_.predict(steer_controller_input_history_);
      }
      acc_controller_input_history_with_schedule.tail(predict_step_*horizon_len_ - 1) = acc_controller_input_prediction;
      steer_controller_input_history_with_schedule.tail(predict_step_*horizon_len_ - 1) = steer_controller_input_prediction;
    }
      // calculate states_ref by controller inputs history with schedule //
    if (states_ref_mode_ == "predict_by_polynomial_regression" || states_ref_mode_ == "controller_d_inputs_schedule" || states_ref_mode_ == "controller_d_steer_schedule")
    {
      states_ref[0] = states;
      Eigen::VectorXd states_ref_tmp = states;
      for (int i = 0; i < horizon_len_; i++) {
        for (int j = 0; j < predict_step_; j++) {
          Eigen::Vector2d inputs_tmp;          
          inputs_tmp << acc_controller_input_history_with_schedule[controller_acc_input_history_len_ + i*predict_step_ + j - acc_delay_step_ - 1], steer_controller_input_history_with_schedule[controller_steer_input_history_len_ + i*predict_step_ + j - steer_delay_step_ - 1];
          states_ref_tmp = nominal_dynamics_controller_.F_nominal(states_ref_tmp, inputs_tmp);
          if (steer_controller_prediction_aided_){
            states_ref_tmp[steer_index_]
             = (predict_step_ - j - 1) * steer_controller_prediction_[i]/ predict_step_
                      + (j + 1) * steer_controller_prediction_[i + 1]/ predict_step_;
          }
        }
        states_ref[i + 1] = states_ref_tmp;
      }
    }
    if (states_ref_mode_ == "controller_prediction")
    {//calculate states_ref by controller prediction

      states_ref = std::vector<Eigen::VectorXd>(horizon_len_ + 1,states);
      for (int i = 0; i< horizon_len_+1; i++) {
        states_ref[i][x_index_] = x_controller_prediction_[i];
        states_ref[i][y_index_] = y_controller_prediction_[i];
        states_ref[i][vel_index_] = vel_controller_prediction_[i];
        states_ref[i][yaw_index_] = yaw_controller_prediction_[i];
        states_ref[i][acc_index_] =acc_controller_prediction_[i];
        states_ref[i][steer_index_] =steer_controller_prediction_[i];
      }
    }
    
    adaptor_ilqr_.set_inputs_cost(acc_rate_cost,
                              steer_rate_cost);

    std::vector<Eigen::VectorXd> d_input_ref(horizon_len_, Eigen::VectorXd::Zero(2));

    Eigen::VectorXd states_tmp_compensation = state_history_lstm_[predict_step_*(update_lstm_len_ - compensation_lstm_len_)];
    Eigen::VectorXd previous_error_tmp = Eigen::VectorXd::Zero(previous_error_.size());
    for (int i = 0; i<compensation_lstm_len_; i++) {
      Eigen::VectorXd acc_input_history_concat = Eigen::VectorXd::Zero(acc_queue_size_ + predict_step_);
      Eigen::VectorXd steer_input_history_concat = Eigen::VectorXd::Zero(steer_queue_size_ + predict_step_);
      acc_input_history_concat.head(acc_queue_size_) = acc_input_history_lstm_[predict_step_*(update_lstm_len_ - compensation_lstm_len_ + i)];
      steer_input_history_concat.head(steer_queue_size_) = steer_input_history_lstm_[predict_step_*(update_lstm_len_ - compensation_lstm_len_ + i)];
      for (int j = 0; j < predict_step_; j++) {
        acc_input_history_concat[acc_queue_size_ + j] = acc_input_history_lstm_[predict_step_*(update_lstm_len_ - compensation_lstm_len_ + i) + j + 1][acc_queue_size_ - 1];
        steer_input_history_concat[steer_queue_size_ + j] = steer_input_history_lstm_[predict_step_*(update_lstm_len_ - compensation_lstm_len_ + i) + j + 1][steer_queue_size_ - 1];
      }
      adaptor_ilqr_.update_input_queue_for_compensation(states_tmp_compensation,acc_input_history_concat, steer_input_history_concat);
      Eigen::VectorXd compensation = adaptor_ilqr_.prediction_for_compensation(states_tmp_compensation, acc_input_history_concat, steer_input_history_concat);
      states_tmp_compensation = adaptor_ilqr_.F_with_model_for_compensation(states_tmp_compensation, acc_input_history_concat, steer_input_history_concat, h_lstm_compensation, c_lstm_compensation, previous_error_tmp, i) + compensation;
    }
    Eigen::VectorXd compensation_error_vector = (states - states_tmp_compensation)/compensation_lstm_len_;
    adaptor_ilqr_.update_regression_matrix_for_compensation(compensation_error_vector);
    double acc_input, steer_input;
    if (states_ref_mode_ == "controller_d_inputs_schedule" and use_controller_inputs_as_target_){
      for (int i = 0; i<horizon_len_;i++){
        d_input_ref[i][0] = acc_controller_d_inputs_schedule_[i];
        d_input_ref[i][1] = steer_controller_d_inputs_schedule_[i];
      }
    }
    adaptor_ilqr_.compute_optimal_control(
      states, acc_input_history_, steer_input_history_, d_inputs_schedule_, h_lstm, c_lstm, states_ref, d_input_ref, previous_error_, states_prediction_, 
      acc_controller_input_history_with_schedule.tail(predict_step_*horizon_len_), steer_controller_input_history_with_schedule.tail(predict_step_*horizon_len_),
      acc_input, steer_input
      );

    if (input_filter_mode_ == "butterworth"){
      Eigen::Vector2d inputs_tmp = Eigen::Vector2d(acc_input, steer_input);
      inputs_tmp = butterworth_filter_.apply(inputs_tmp);
      acc_input = inputs_tmp[0];
      steer_input = inputs_tmp[1];
    }
    else if (input_filter_mode_ == "polynomial"){
      acc_input = polynomial_filter_for_acc_inputs_.fit_transform(time_stamp,acc_input);
      steer_input = polynomial_filter_for_steer_inputs_.fit_transform(time_stamp,steer_input);
    }
    double mix_ratio_vel = calc_table_value(states[vel_index_], mix_ratio_vel_domain_table_, mix_ratio_vel_target_table_);
    double mix_ratio_time = calc_table_value(time_stamp - start_time_, mix_ratio_time_domain_table_, mix_ratio_time_target_table_);
    double mix_ratio = mix_ratio_vel * mix_ratio_time;
    acc_input = (1 - mix_ratio) * acc_controller_input + mix_ratio * acc_input;
    steer_input = (1 - mix_ratio) * steer_controller_input + mix_ratio * steer_input;

    //acc_input_history_.head(acc_queue_size_ - 1) = acc_input_history_.tail(acc_queue_size_ - 1);
    //steer_input_history_.head(steer_queue_size_ - 1) = steer_input_history_.tail(steer_queue_size_ - 1);
    state_history_lstm_obs_.push_back(states);
    acc_input_history_obs_.push_back(acc_input);
    steer_input_history_obs_.push_back(steer_input);
    acc_input_history_lstm_obs_.push_back(acc_input_history_);
    steer_input_history_lstm_obs_.push_back(steer_input_history_);

    //acc_input_history_(acc_queue_size_ - 1) = acc_input;
    //steer_input_history_(steer_queue_size_ - 1) = steer_input;
    //acc_input_history_lstm_.erase(acc_input_history_lstm_.begin());
    //steer_input_history_lstm_.erase(steer_input_history_lstm_.begin());
    
    //acc_input_history_lstm_.push_back(acc_input_history_);
    //steer_input_history_lstm_.push_back(steer_input_history_);
    return Eigen::Vector2d(acc_input, steer_input);
  }
  Eigen::MatrixXd VehicleAdaptor::get_states_prediction()
  {
    Eigen::MatrixXd states_prediction_matrix(states_prediction_[0].size(), states_prediction_.size());
    for (int i = 0; i < int(states_prediction_.size()); i++)
    {
      states_prediction_matrix.col(i) = states_prediction_[i];
    }
    return states_prediction_matrix;
  }
  Eigen::MatrixXd VehicleAdaptor::get_d_inputs_schedule()
  {
    Eigen::MatrixXd d_inputs_schedule_matrix(d_inputs_schedule_[0].size(), d_inputs_schedule_.size());
    for (int i = 0; i < int(d_inputs_schedule_.size()); i++)
    {
      d_inputs_schedule_matrix.col(i) = d_inputs_schedule_[i];
    }
    return d_inputs_schedule_matrix;
  }
  void VehicleAdaptor::send_initialized_flag()
  {
    initialized_ = false;
    states_ref_mode_ = "predict_by_polynomial_regression";
  }

PYBIND11_MODULE(utils, m)
{
  m.def("rotate_data", &rotate_data);
  py::class_<NominalDynamics>(m, "NominalDynamics")
    .def(py::init())
    .def("set_params", &NominalDynamics::set_params)
    .def("F_nominal", &NominalDynamics::F_nominal)
    .def("F_nominal_with_diff", &NominalDynamics::F_nominal_with_diff)
    .def("F_nominal_predict", &NominalDynamics::F_nominal_predict)
    .def("F_nominal_predict_with_diff", &NominalDynamics::F_nominal_predict_with_diff);
  py::class_<VehicleAdaptor>(m, "VehicleAdaptor")
    .def(py::init())
    .def("set_NN_params", &VehicleAdaptor::set_NN_params)
    .def("set_NN_params_from_csv", &VehicleAdaptor::set_NN_params_from_csv)
    .def("clear_NN_params", &VehicleAdaptor::clear_NN_params)
    .def("set_controller_d_inputs_schedule", &VehicleAdaptor::set_controller_d_inputs_schedule)
    .def("set_controller_d_steer_schedule", &VehicleAdaptor::set_controller_d_steer_schedule)
    .def("set_controller_prediction", &VehicleAdaptor::set_controller_prediction)
    .def("set_controller_steer_prediction", &VehicleAdaptor::set_controller_steer_prediction)
    .def("get_adjusted_inputs", &VehicleAdaptor::get_adjusted_inputs)
    .def("get_states_prediction", &VehicleAdaptor::get_states_prediction)
    .def("get_d_inputs_schedule", &VehicleAdaptor::get_d_inputs_schedule)
    .def("send_initialized_flag", &VehicleAdaptor::send_initialized_flag);
}
