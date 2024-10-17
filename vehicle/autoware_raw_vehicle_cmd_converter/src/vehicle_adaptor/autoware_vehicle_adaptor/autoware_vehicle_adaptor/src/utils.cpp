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
/*
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
*/
///////////////// TrainedDynamics ///////////////////////
  TrainedDynamics::TrainedDynamics() {
    YAML::Node optimization_param_node = YAML::LoadFile(get_param_dir_path() + "/optimization_param.yaml");
    minimum_acc_diff_ = optimization_param_node["optimization_parameter"]["minimum_diff"]["minimum_acc_diff"].as<double>();
    minimum_steer_diff_ = optimization_param_node["optimization_parameter"]["minimum_diff"]["minimum_steer_diff"].as<double>();
    /*
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
    */
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
    const Eigen::MatrixXd & weight_acc_encoder_layer_1, const Eigen::MatrixXd & weight_steer_encoder_layer_1,
    const Eigen::MatrixXd & weight_acc_encoder_layer_2, const Eigen::MatrixXd & weight_steer_encoder_layer_2,
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_ih, const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_hh,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_encoder_layer_1, const Eigen::VectorXd & bias_steer_encoder_layer_1,
    const Eigen::VectorXd & bias_acc_encoder_layer_2, const Eigen::VectorXd & bias_steer_encoder_layer_2,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const std::vector<Eigen::VectorXd> & bias_lstm_encoder_ih, const std::vector<Eigen::VectorXd> & bias_lstm_encoder_hh,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted)
  {
    transform_model_to_eigen_.set_params(
      weight_acc_encoder_layer_1, weight_steer_encoder_layer_1, weight_acc_encoder_layer_2, weight_steer_encoder_layer_2,
      weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2,
      weight_lstm_encoder_ih, weight_lstm_encoder_hh, weight_lstm_ih, weight_lstm_hh,
      weight_complimentary_layer, weight_linear_relu, weight_final_layer,
      bias_acc_encoder_layer_1, bias_steer_encoder_layer_1, bias_acc_encoder_layer_2, bias_steer_encoder_layer_2,
      bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2, bias_steer_layer_2,
      bias_lstm_encoder_ih, bias_lstm_encoder_hh, bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer,
      bias_linear_relu, bias_final_layer, vel_scaling, vel_bias);
      h_dim_full_ = weight_lstm_encoder_ih[0].cols();
      h_dim_ = weight_lstm_hh.cols();
      //num_layers_encoder_ = weight_lstm_encoder_ih.size();
      state_component_predicted_ = state_component_predicted;
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
  Eigen::VectorXd TrainedDynamics::nominal_prediction(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat)
  {
    return nominal_dynamics_.F_with_input_history(
      states, acc_input_history_concat, steer_input_history_concat);
  }
  void TrainedDynamics::update_lstm_states(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat,
    std::vector<Eigen::VectorXd> & h_lstm, std::vector<Eigen::VectorXd> & c_lstm, const Eigen::Vector2d & acc_steer_error)
  {
    Eigen::VectorXd NN_input = Eigen::VectorXd::Zero(3 + acc_queue_size_ + steer_queue_size_+2*predict_step_);
    NN_input << states[vel_index_], states[acc_index_], states[steer_index_],
      acc_input_history_concat, steer_input_history_concat;
    std::vector<Eigen::VectorXd> h_lstm_next, c_lstm_next;
    transform_model_to_eigen_.update_lstm(
      NN_input, h_lstm, c_lstm, h_lstm_next, c_lstm_next, acc_steer_error);
    h_lstm = h_lstm_next;
    c_lstm = c_lstm_next;
  }
  void TrainedDynamics::initialize_compensation(int acc_queue_size, int steer_queue_size, int predict_step, int h_dim_full)
  {
    linear_regression_compensation_.initialize(acc_queue_size, steer_queue_size, predict_step, h_dim_full);
  }
  void TrainedDynamics::update_state_queue_for_compensation(Eigen::VectorXd states){
    linear_regression_compensation_.update_state_queue(states);
  }
  void TrainedDynamics::update_one_step_for_compensation(Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm, Eigen::VectorXd error_vector){
    linear_regression_compensation_.update_one_step(acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm, error_vector);
  }
  void TrainedDynamics::update_regression_matrix_for_compensation(){
    linear_regression_compensation_.update_regression_matrix();
  }
  void TrainedDynamics::save_state_queue_for_compensation(){
    linear_regression_compensation_.save_state_queue();
  }
  void TrainedDynamics::load_state_queue_for_compensation(){
    linear_regression_compensation_.load_state_queue();
  }
  void TrainedDynamics::initialize_for_candidates_compensation(int num_candidates){
    linear_regression_compensation_.initialize_for_candidates(num_candidates);
  }
  /*
  void TrainedDynamics::update_input_queue_for_compensation(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat){
    linear_regression_compensation_.update_input_queue(states, acc_input_history_concat, steer_input_history_concat);
  }
  void TrainedDynamics::update_regression_matrix_for_compensation(Eigen::VectorXd error_vector){
    linear_regression_compensation_.update_regression_matrix(error_vector);
  }
  */
  Eigen::VectorXd TrainedDynamics::prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm
  ){
    return linear_regression_compensation_.predict(states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
  }
  Eigen::MatrixXd TrainedDynamics::Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat, Eigen::MatrixXd H_lstm, Eigen::MatrixXd C_lstm
  ){
    return linear_regression_compensation_.Predict(States, Acc_input_history_concat, Steer_input_history_concat, H_lstm, C_lstm);
  }
  void TrainedDynamics::set_offline_data_set_for_compensation(
    Eigen::MatrixXd XXT, Eigen::MatrixXd YXT
  ){
    linear_regression_compensation_.set_offline_data_set(XXT, YXT);
  }
  void TrainedDynamics::unset_offline_data_set_for_compensation(){
    linear_regression_compensation_.unset_offline_data_set();
  }
  void TrainedDynamics::set_projection_matrix_for_compensation(Eigen::MatrixXd P)
  {
    linear_regression_compensation_.set_projection_matrix(P);
  }
  void TrainedDynamics::unset_projection_matrix_for_compensation()
  {
    linear_regression_compensation_.unset_projection_matrix();
  }
  /*
  Eigen::VectorXd TrainedDynamics::get_compensation_bias(){
    return linear_regression_compensation_.get_bias();
  }
  */
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
    Eigen::VectorXd compensation = prediction_for_compensation(states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
    states_next += compensation * predict_dt_;
    h_lstm = h_lstm_next;
    c_lstm = c_lstm_next;

    return states_next;
  }
  Eigen::VectorXd TrainedDynamics::F_with_model_without_compensation(
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
    Eigen::VectorXd compensation = prediction_for_compensation(states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
    states_next += compensation * predict_dt_;
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
    double yaw = states[yaw_index_];
    C = Eigen::MatrixXd::Zero(
      2*h_dim_ + states.size(),
      2*h_dim_ + states.size() + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_);
    for (int i = 0; i< int(state_component_predicted_index_.size()); i++) {
      if (state_component_predicted_index_[i] == x_index_) {
        C(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 2) += dF_d_states_with_history(i, 0) * predict_dt_ * std::cos(yaw);
        C(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 2) -= dF_d_states_with_history(i + 1, 0) * predict_dt_ * std::sin(yaw);
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) += dF_d_states_with_history.block(i, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) * predict_dt_ * std::cos(yaw);
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) -= dF_d_states_with_history.block(i + 1, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) * predict_dt_ * std::sin(yaw);
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 0, 1, 2 * h_dim_) += dF_dhc.row(i) * predict_dt_ * std::cos(yaw);
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 0, 1, 2 * h_dim_) -= dF_dhc.row(i + 1) * predict_dt_ * std::sin(yaw);
      } else if (state_component_predicted_index_[i] == y_index_) {
        C(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 2) += dF_d_states_with_history(i - 1, 0) * predict_dt_ * std::sin(yaw);
        C(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 2) += dF_d_states_with_history(i, 0) * predict_dt_ * std::cos(yaw);
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) += dF_d_states_with_history.block(i - 1, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) * predict_dt_ * std::sin(yaw);
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) += dF_d_states_with_history.block(i, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) * predict_dt_ * std::cos(yaw);
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 0, 1, 2 * h_dim_) += dF_dhc.row(i - 1) * predict_dt_ * std::sin(yaw);
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 0, 1, 2 * h_dim_) += dF_dhc.row(i) * predict_dt_ * std::cos(yaw);
      } else {
        C(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_+2) += dF_d_states_with_history(i, 0) * predict_dt_;
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + 4, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2*predict_step_) += dF_d_states_with_history.block(i, 1, 1, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_) * predict_dt_;
        C.block(2 * h_dim_ + state_component_predicted_index_[i], 0, 1, 2 * h_dim_) += dF_dhc.row(i) * predict_dt_;
      }
    }
    // C.block(2 * h_dim_+4, 2 * h_dim_+2, states.size()-4, 1) = dF_d_states_with_history.col(0) * predict_dt_;
    // C.block(2*h_dim_+4, 2*h_dim_+3, states.size()-4, states.size() -3-1+acc_queue_size_+steer_queue_size_+2*predict_step_) = dF_d_states_with_history.rightCols(states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_) * predict_dt_;
    
    C.block(0, 0, 2 * h_dim_, 2 * h_dim_) = dhc_dhc;

    C.block(0, 2 * h_dim_ + 2, 2 * h_dim_, 1) = dhc_d_states_with_history.col(0);
    C.block(0, 2 * h_dim_ + 4, 2 * h_dim_, states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_) = dhc_d_states_with_history.rightCols(states.size() - 3 - 1 + acc_queue_size_ + steer_queue_size_ + 2 * predict_step_);

    //C.block(2 * h_dim_ + 4, 0, states.size() -4, 2 * h_dim_) = dF_dhc;

    Eigen::VectorXd d_steer_d_steer_input =  C.block(2 * h_dim_ + steer_index_, 2 * h_dim_ + steer_input_start_index_ + predict_step_, 1, steer_queue_size_ + predict_step_).transpose();
    d_steer_d_steer_input.head(steer_queue_size_) += A.block(steer_index_, steer_input_start_index_, 1, steer_queue_size_).transpose();
    Eigen::VectorXd d_acc_d_acc_input = C.block(2 * h_dim_ + acc_index_, 2 * h_dim_ + states.size(), 1, acc_queue_size_ + predict_step_).transpose();
    d_acc_d_acc_input.head(acc_queue_size_) += A.block(acc_index_, states.size(), 1, acc_queue_size_).transpose();
    double steer_diff = d_steer_d_steer_input.sum();
    double acc_diff = d_acc_d_acc_input.sum();

    if (steer_diff < minimum_steer_diff_) {
      int max_d_steer_index;
      d_steer_d_steer_input.array().maxCoeff(&max_d_steer_index);
      max_d_steer_index = std::min(max_d_steer_index, steer_queue_size_);
      for (int i = 0; i< predict_step_; i++) {
        C(2 * h_dim_ + steer_index_, 2 * h_dim_ + steer_input_start_index_ + predict_step_ + max_d_steer_index + i) += (minimum_steer_diff_ - steer_diff)/predict_step_;
      }
    }
    if (acc_diff < minimum_acc_diff_) {
      int max_d_acc_index;
      d_acc_d_acc_input.array().maxCoeff(&max_d_acc_index);
      max_d_acc_index = std::min(max_d_acc_index, acc_queue_size_);
      for (int i = 0; i< predict_step_; i++) {
        C(2 * h_dim_ + acc_index_, 2 * h_dim_ + states.size() + acc_queue_size_ + predict_step_ + max_d_acc_index + i) += (minimum_acc_diff_ - acc_diff)/predict_step_;
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
    Eigen::VectorXd compensation = prediction_for_compensation(states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
    h_lstm = h_lstm_next;
    c_lstm = c_lstm_next;
    for (int i = 0; i < int(state_component_predicted_index_.size()); i++) {
      if (state_component_predicted_index_[i] == x_index_) {
        states_next[x_index_] += previous_error[i] * predict_dt_ * std::cos(yaw) - previous_error[i + 1] * predict_dt_ * std::sin(yaw);
        C(2 * h_dim_ + state_component_predicted_index_[i],2 * h_dim_ + yaw_index_) += - previous_error[i] * predict_dt_ * std::sin(yaw) - previous_error[i + 1] * predict_dt_ * std::cos(yaw);
      } else if (state_component_predicted_index_[i] == y_index_) {
        states_next[y_index_] += previous_error[i - 1] * predict_dt_ * std::sin(yaw) + previous_error[i] * predict_dt_ * std::cos(yaw);
        C(2 * h_dim_ + state_component_predicted_index_[i], 2 * h_dim_ + yaw_index_) += previous_error[i - 1] * predict_dt_ * std::cos(yaw) - previous_error[i] * predict_dt_ * std::sin(yaw);
      } else {
        states_next[state_component_predicted_index_[i]] += previous_error[i] * predict_dt_;
      }
    }
    states_next += compensation * predict_dt_;
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
    Eigen::MatrixXd Compensation = Prediction_for_compensation(States, Acc_input_history_concat, Steer_input_history_concat, H_lstm, C_lstm);
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
        NN_input.col(i), h_lstm, c_lstm, h_lstm_next, c_lstm_next, NN_output);
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
    States_next += Compensation * predict_dt_;
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
    load_state_queue_for_compensation();
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
    acc_input_weight_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["acc_input_weight_target_table"].as<std::vector<double>>();
    longitudinal_coef_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["longitudinal_coef_target_table"].as<std::vector<double>>();
    longitudinal_coef_by_vel_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["longitudinal_coef_by_vel_error_domain_table"].as<std::vector<double>>();
    longitudinal_coef_by_acc_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["longitudinal_coef_by_acc_error_domain_table"].as<std::vector<double>>();

    steer_input_weight_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["steer_input_weight_target_table"].as<std::vector<double>>();
    lateral_coef_target_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["lateral_coef_target_table"].as<std::vector<double>>();
    lateral_coef_by_lateral_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["lateral_coef_by_lateral_error_domain_table"].as<std::vector<double>>();
    lateral_coef_by_yaw_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["lateral_coef_by_yaw_error_domain_table"].as<std::vector<double>>();
    lateral_coef_by_steer_error_domain_table_ = optimization_param_node["optimization_parameter"]["weight_parameter"]["lateral_coef_by_steer_error_domain_table"].as<std::vector<double>>();
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
    const Eigen::MatrixXd & weight_acc_encoder_layer_1, const Eigen::MatrixXd & weight_steer_encoder_layer_1,
    const Eigen::MatrixXd & weight_acc_encoder_layer_2, const Eigen::MatrixXd & weight_steer_encoder_layer_2,
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_ih, const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_hh,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_encoder_layer_1, const Eigen::VectorXd & bias_steer_encoder_layer_1,
    const Eigen::VectorXd & bias_acc_encoder_layer_2, const Eigen::VectorXd & bias_steer_encoder_layer_2,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const std::vector<Eigen::VectorXd> & bias_lstm_encoder_ih, const std::vector<Eigen::VectorXd> & bias_lstm_encoder_hh,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted)
  {
    trained_dynamics_.set_NN_params(
      weight_acc_encoder_layer_1, weight_steer_encoder_layer_1, weight_acc_encoder_layer_2, weight_steer_encoder_layer_2,
      weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2,
      weight_lstm_encoder_ih, weight_lstm_encoder_hh, weight_lstm_ih, weight_lstm_hh,
      weight_complimentary_layer, weight_linear_relu, weight_final_layer,
      bias_acc_encoder_layer_1, bias_steer_encoder_layer_1, bias_acc_encoder_layer_2, bias_steer_encoder_layer_2,
      bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2, bias_steer_layer_2,
      bias_lstm_encoder_ih, bias_lstm_encoder_hh, bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer,
      bias_linear_relu, bias_final_layer, vel_scaling, vel_bias, state_component_predicted);
    h_dim_ = weight_lstm_hh.cols();
    //num_layers_encoder_ = weight_lstm_encoder_ih.size();
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
  Eigen::VectorXd AdaptorILQR::nominal_prediction(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat)
  {
    return trained_dynamics_.nominal_prediction(states, acc_input_history_concat, steer_input_history_concat);
  }
  void AdaptorILQR::update_lstm_states(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat,
    std::vector<Eigen::VectorXd> & h_lstm, std::vector<Eigen::VectorXd> & c_lstm, const Eigen::Vector2d & previous_error)
  {
    trained_dynamics_.update_lstm_states(
      states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm, previous_error);
  }
  Eigen::VectorXd AdaptorILQR::F_with_model_without_compensation(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, int horizon)
  {
    return trained_dynamics_.F_with_model_without_compensation(
      states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm, previous_error, horizon);
  }
  void AdaptorILQR::initialize_compensation(int acc_queue_size, int steer_queue_size, int predict_step, int h_dim_full){
    trained_dynamics_.initialize_compensation(acc_queue_size, steer_queue_size, predict_step, h_dim_full);
  }
  void AdaptorILQR::update_state_queue_for_compensation(Eigen::VectorXd states){
    trained_dynamics_.update_state_queue_for_compensation(states);
  }
  void AdaptorILQR::update_one_step_for_compensation(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm){
    trained_dynamics_.update_one_step_for_compensation(states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
  }
  void AdaptorILQR::update_regression_matrix_for_compensation(){
    trained_dynamics_.update_regression_matrix_for_compensation();
  }
  void AdaptorILQR::save_state_queue_for_compensation(){
    trained_dynamics_.save_state_queue_for_compensation();
  }
  void AdaptorILQR::load_state_queue_for_compensation(){
    trained_dynamics_.load_state_queue_for_compensation();
  }
  Eigen::VectorXd AdaptorILQR::prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm
  ){
    return trained_dynamics_.prediction_for_compensation(states, acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
  }
  void AdaptorILQR::set_offline_data_set_for_compensation(
    Eigen::MatrixXd XXT, Eigen::MatrixXd YXT
  ){
    trained_dynamics_.set_offline_data_set_for_compensation(XXT, YXT);
  }
  void AdaptorILQR::unset_offline_data_set_for_compensation(){
    trained_dynamics_.unset_offline_data_set_for_compensation();
  }
  void AdaptorILQR::set_projection_matrix_for_compensation(
    Eigen::MatrixXd P
  ){
    trained_dynamics_.set_projection_matrix_for_compensation(P);
  }
  void AdaptorILQR::unset_projection_matrix_for_compensation(){
    trained_dynamics_.unset_projection_matrix_for_compensation();
  }
  /*
  Eigen::MatrixXd AdaptorILQR::Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat, Eigen::MatrixXd H_lstm, Eigen::MatrixXd C_lstm
  ){
    return trained_dynamics_.Prediction_for_compensation(States, Acc_input_history_concat, Steer_input_history_concat, H_lstm, C_lstm);
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
  */
  void AdaptorILQR::calc_forward_trajectory_with_cost(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history, Eigen::VectorXd steer_input_history,
    std::vector<Eigen::MatrixXd> D_inputs_schedule, const Eigen::VectorXd & h_lstm,
    const Eigen::VectorXd & c_lstm,
    const Eigen::VectorXd & previous_error, std::vector<Eigen::MatrixXd> & states_prediction,
    const std::vector<Eigen::VectorXd> & states_ref, const std::vector<Eigen::VectorXd> & d_input_ref, 
    std::vector<Eigen::Vector2d> inputs_ref, double longitudinal_coef, double lateral_coef, double acc_input_weight, double steer_input_weight,
    Eigen::VectorXd & Cost)
  {
    int sample_size = D_inputs_schedule[0].cols();
    trained_dynamics_.initialize_for_candidates_compensation(sample_size);
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
      Eigen::Matrix2d xy_cost_matrix_sqrt = rotation_matrix.transpose() * Eigen::DiagonalMatrix<double, 2>(std::sqrt(longitudinal_coef * x_cost), std::sqrt(lateral_coef * y_cost)) * rotation_matrix;
      Eigen::MatrixXd xy_diff = states_prediction[i].block(x_index_, 0, 2, sample_size).colwise() - states_ref[i].segment(x_index_, 2);
      for (int j = 0; j < num_state_component_ilqr_; j++) {
        if (state_component_ilqr_[j] == "x") {
          Cost += 0.5 * (xy_cost_matrix_sqrt*xy_diff).colwise().squaredNorm();
        } else if (state_component_ilqr_[j] == "vel") {
          Cost += 0.5 * longitudinal_coef * vel_cost * (states_prediction[i].row(vel_index_).array() - states_ref[i][vel_index_]).square().matrix();
        } else if (state_component_ilqr_[j] == "yaw") {
          Cost += 0.5 * lateral_coef * yaw_cost * (states_prediction[i].row(yaw_index_).array() - states_ref[i][yaw_index_]).square().matrix();
        } else if (state_component_ilqr_[j] == "acc") {
          Cost += 0.5 * longitudinal_coef * acc_cost * (states_prediction[i].row(acc_index_).array() - states_ref[i][acc_index_]).square().matrix();
        } else if (state_component_ilqr_[j] == "steer") {
          Cost += 0.5 * lateral_coef * steer_cost * (states_prediction[i].row(steer_index_).array() - states_ref[i][steer_index_]).square().matrix();
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
    Eigen::Matrix2d xy_cost_matrix_sqrt = rotation_matrix.transpose() * Eigen::DiagonalMatrix<double, 2>(std::sqrt(longitudinal_coef * x_terminal_cost_), std::sqrt(lateral_coef * y_terminal_cost_)) * rotation_matrix;
    Eigen::MatrixXd xy_diff = states_prediction[horizon_len_].block(x_index_, 0, 2, sample_size).colwise() - states_ref[horizon_len_].segment(x_index_, 2);
    for (int j = 0; j < num_state_component_ilqr_; j++) {
      if (state_component_ilqr_[j] == "x") {
        Cost += 0.5 * (xy_cost_matrix_sqrt*xy_diff).colwise().squaredNorm();
      } else if (state_component_ilqr_[j] == "vel") {
        Cost += 0.5 * longitudinal_coef * vel_terminal_cost_ * (states_prediction[horizon_len_].row(vel_index_).array() - states_ref[horizon_len_][vel_index_]).square().matrix();
      } else if (state_component_ilqr_[j] == "yaw") {
        Cost += 0.5 *lateral_coef * yaw_terminal_cost_ * (states_prediction[horizon_len_].row(yaw_index_).array() - states_ref[horizon_len_][yaw_index_]).square().matrix();
      } else if (state_component_ilqr_[j] == "acc") {
        Cost += 0.5 * longitudinal_coef * acc_terminal_cost_ * (states_prediction[horizon_len_].row(acc_index_).array() - states_ref[horizon_len_][acc_index_]).square().matrix();
      } else if (state_component_ilqr_[j] == "steer") {
        Cost += 0.5 * lateral_coef * steer_terminal_cost_ * (states_prediction[horizon_len_].row(steer_index_).array() - states_ref[horizon_len_][steer_index_]).square().matrix();
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
    std::vector<Eigen::Vector2d> & inputs_ref, double & longitudinal_coef, double & lateral_coef, double & acc_input_weight, double & steer_input_weight)
  {
    trained_dynamics_.load_state_queue_for_compensation();
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
    longitudinal_coef = std::max(
      calc_table_value(max_vel_error, longitudinal_coef_by_vel_error_domain_table_, longitudinal_coef_target_table_),
      calc_table_value(max_acc_error, longitudinal_coef_by_acc_error_domain_table_, longitudinal_coef_target_table_));
    lateral_coef = std::max(
      {
      calc_table_value(max_lateral_error, lateral_coef_by_lateral_error_domain_table_, lateral_coef_target_table_),
      calc_table_value(max_steer_error, lateral_coef_by_steer_error_domain_table_, lateral_coef_target_table_),
      calc_table_value(max_yaw_error, lateral_coef_by_yaw_error_domain_table_, lateral_coef_target_table_)
      });
    acc_input_weight = std::min(
      calc_table_value(max_vel_error, longitudinal_coef_by_vel_error_domain_table_, acc_input_weight_target_table_),
      calc_table_value(max_acc_error, longitudinal_coef_by_acc_error_domain_table_, acc_input_weight_target_table_));
    steer_input_weight = std::min(
      {
      calc_table_value(max_lateral_error, lateral_coef_by_lateral_error_domain_table_, steer_input_weight_target_table_),
      calc_table_value(max_steer_error, lateral_coef_by_steer_error_domain_table_, steer_input_weight_target_table_),
      calc_table_value(max_yaw_error, lateral_coef_by_yaw_error_domain_table_, steer_input_weight_target_table_)
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
    std::vector<Eigen::Vector2d> inputs_ref, double longitudinal_coef, double lateral_coef, double acc_input_weight, double steer_input_weight,
    const std::vector<Eigen::Vector2d> & inputs_schedule, std::vector<Eigen::MatrixXd> & K, std::vector<Eigen::VectorXd> & k)
  {
    K = std::vector<Eigen::MatrixXd>(horizon_len_);
    k = std::vector<Eigen::VectorXd>(horizon_len_);
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_, num_state_component_ilqr_+2*h_dim_ + acc_queue_size_ + steer_queue_size_);
    Eigen::VectorXd w = Eigen::VectorXd::Zero(num_state_component_ilqr_ + 2*h_dim_ + acc_queue_size_ + steer_queue_size_);
    double yaw_target = states_ref[horizon_len_][yaw_index_];
    Eigen::Matrix2d rotation_matrix = Eigen::Rotation2Dd(-yaw_target).toRotationMatrix();
    Eigen::Matrix2d xy_cost_matrix = rotation_matrix.transpose() * Eigen::DiagonalMatrix<double, 2>(Eigen::Vector2d(longitudinal_coef * x_terminal_cost_, lateral_coef * y_terminal_cost_)) * rotation_matrix;
    Eigen::Vector2d xy_diff = Eigen::Vector2d(states_prediction[horizon_len_][x_index_] - states_ref[horizon_len_][x_index_], states_prediction[horizon_len_][y_index_] - states_ref[horizon_len_][y_index_]);
    for (int j = 0; j < num_state_component_ilqr_; j++) {
      if (state_component_ilqr_[j] == "x") {
        P.block(2*h_dim_ + j, 2*h_dim_ + j, 2, 2) = xy_cost_matrix;
        w.segment(2*h_dim_+j,2) = xy_cost_matrix * xy_diff;
      } else if (state_component_ilqr_[j] == "vel") {
        P(2*h_dim_ + j, 2*h_dim_ + j) = longitudinal_coef * vel_terminal_cost_;
        w(2*h_dim_ + j) = longitudinal_coef * vel_terminal_cost_ * (states_prediction[horizon_len_][vel_index_] - states_ref[horizon_len_][vel_index_]);
      } else if (state_component_ilqr_[j] == "yaw") {
        P(2*h_dim_ + j, 2*h_dim_ + j) = lateral_coef * yaw_terminal_cost_;
        w(2*h_dim_ + j) = lateral_coef * yaw_terminal_cost_ * (states_prediction[horizon_len_][yaw_index_] - states_ref[horizon_len_][yaw_index_]);
      } else if (state_component_ilqr_[j] == "acc") {
        P(2*h_dim_ + j, 2*h_dim_ + j) = longitudinal_coef * acc_terminal_cost_;
        w(2*h_dim_ + j) = longitudinal_coef * acc_terminal_cost_ * (states_prediction[horizon_len_][acc_index_] - states_ref[horizon_len_][acc_index_]);
      } else if (state_component_ilqr_[j] == "steer") {
        P(2*h_dim_ + j, 2*h_dim_ + j) = lateral_coef * steer_terminal_cost_;
        w(2*h_dim_ + j) = lateral_coef * steer_terminal_cost_ * (states_prediction[horizon_len_][steer_index_] - states_ref[horizon_len_][steer_index_]);
      } 
    }

    P(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_, 2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_) = acc_input_weight;
    P(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_, 2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_) = steer_input_weight;
    w(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ - predict_step_) = acc_input_weight * (inputs_schedule[horizon_len_-1][0] - inputs_ref[horizon_len_-1][0]);
    w(2*h_dim_ + num_state_component_ilqr_ + acc_queue_size_ + steer_queue_size_ - predict_step_) = steer_input_weight * (inputs_schedule[horizon_len_-1][1] - inputs_ref[horizon_len_-1][1]);
    
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
      xy_cost_matrix = rotation_matrix.transpose() * Eigen::DiagonalMatrix<double, 2>(Eigen::Vector2d(longitudinal_coef * x_cost, lateral_coef * y_cost)) * rotation_matrix;
      xy_diff = Eigen::Vector2d(states_prediction[i][x_index_] - states_ref[i][x_index_], states_prediction[i][y_index_] - states_ref[i][y_index_]);
      for (int j = 0; j < num_state_component_ilqr_; j++){
        if (state_component_ilqr_[j] == "x") {
          P.block(2*h_dim_ + j, 2*h_dim_ + j, 2, 2) += xy_cost_matrix;
          w.segment(2*h_dim_ + j, 2) += xy_cost_matrix * xy_diff;
        } else if (state_component_ilqr_[j] == "vel") {
          P(2*h_dim_ + j, 2*h_dim_ + j) += longitudinal_coef * vel_cost;
          w(2*h_dim_ + j) += longitudinal_coef * vel_cost * (states_prediction[i][vel_index_] - states_ref[i][vel_index_]);
        } else if (state_component_ilqr_[j] == "yaw") {
          P(2*h_dim_ + j, 2*h_dim_ + j) += lateral_coef * yaw_cost;
          w(2*h_dim_ + j) += lateral_coef * yaw_cost * (states_prediction[i][yaw_index_] - states_ref[i][yaw_index_]);
        } else if (state_component_ilqr_[j] == "acc") {
          P(2*h_dim_ + j, 2*h_dim_ + j) += longitudinal_coef * acc_cost;
          w(2*h_dim_ + j) += longitudinal_coef * acc_cost * (states_prediction[i][acc_index_] - states_ref[i][acc_index_]);
        } else if (state_component_ilqr_[j] == "steer") {
          P(2*h_dim_ + j, 2*h_dim_ + j) += lateral_coef * steer_cost;
          w(2*h_dim_ + j) += lateral_coef * steer_cost * (states_prediction[i][steer_index_] - states_ref[i][steer_index_]);
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
    double longitudinal_coef, lateral_coef, acc_input_weight, steer_input_weight;
    calc_inputs_ref_info(
      states, acc_input_history, steer_input_history, h_lstm, c_lstm, previous_error, states_ref,
      acc_controller_input_schedule, steer_controller_input_schedule,
      inputs_ref, longitudinal_coef, lateral_coef, acc_input_weight, steer_input_weight);
    std::vector<Eigen::MatrixXd> dF_d_states, dF_d_inputs;
    std::vector<Eigen::Vector2d> inputs_schedule;
    trained_dynamics_.calc_forward_trajectory_with_diff(states, acc_input_history, steer_input_history, d_inputs_schedule,
                      h_lstm, c_lstm, previous_error, states_prediction, dF_d_states, dF_d_inputs, inputs_schedule);
    std::vector<Eigen::MatrixXd> K;
    std::vector<Eigen::VectorXd> k;
    double prev_acc_rate = (acc_input_history[acc_queue_size_ - 1] - acc_input_history[acc_queue_size_ - 2])/control_dt_;
    double prev_steer_rate = (steer_input_history[steer_queue_size_ - 1] - steer_input_history[steer_queue_size_ - 2])/control_dt_;
    compute_ilqr_coefficients(dF_d_states, dF_d_inputs, states_prediction, d_inputs_schedule, states_ref, d_input_ref, prev_acc_rate, prev_steer_rate,
    inputs_ref, longitudinal_coef, lateral_coef, acc_input_weight, steer_input_weight, inputs_schedule, K, k);
    Eigen::VectorXd ls_points = Eigen::VectorXd::LinSpaced(11, 0.0, 1.0);

    std::vector<Eigen::MatrixXd> D_inputs_schedule = calc_line_search_candidates(K, k, dF_d_states, dF_d_inputs, d_inputs_schedule, ls_points);
    Eigen::VectorXd Cost;
    std::vector<Eigen::MatrixXd> States_prediction;
    calc_forward_trajectory_with_cost(states, acc_input_history, steer_input_history, D_inputs_schedule, h_lstm, c_lstm, previous_error, States_prediction, states_ref, d_input_ref,
      inputs_ref, longitudinal_coef, lateral_coef, acc_input_weight, steer_input_weight, Cost);
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
    acc_polynomial_prediction_len_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["acc_polynomial_prediction_len"].as<int>();
    steer_polynomial_prediction_len_ = optimization_param_node["optimization_parameter"]["polynomial_regression"]["steer_polynomial_prediction_len"].as<int>();

    use_acc_linear_extrapolation_ = optimization_param_node["optimization_parameter"]["linear_extrapolation"]["use_acc_linear_extrapolation"].as<bool>();
    use_steer_linear_extrapolation_ = optimization_param_node["optimization_parameter"]["linear_extrapolation"]["use_steer_linear_extrapolation"].as<bool>();
    acc_linear_extrapolation_len_ = optimization_param_node["optimization_parameter"]["linear_extrapolation"]["acc_linear_extrapolation_len"].as<int>();
    steer_linear_extrapolation_len_ = optimization_param_node["optimization_parameter"]["linear_extrapolation"]["steer_linear_extrapolation_len"].as<int>();
    past_len_for_acc_linear_extrapolation_ = optimization_param_node["optimization_parameter"]["linear_extrapolation"]["past_len_for_acc_linear_extrapolation"].as<int>();
    past_len_for_steer_linear_extrapolation_ = optimization_param_node["optimization_parameter"]["linear_extrapolation"]["past_len_for_steer_linear_extrapolation"].as<int>();

    use_sg_for_d_inputs_schedule_ = optimization_param_node["optimization_parameter"]["sg_filter"]["use_sg_for_d_inputs_schedule"].as<bool>();
    sg_window_size_for_d_inputs_schedule_ = optimization_param_node["optimization_parameter"]["sg_filter"]["sg_window_size_for_d_inputs_schedule"].as<int>();
    sg_deg_for_d_inputs_schedule_ = optimization_param_node["optimization_parameter"]["sg_filter"]["sg_deg_for_d_inputs_schedule"].as<int>();
    sg_window_size_for_NN_diff_ = optimization_param_node["optimization_parameter"]["sg_filter"]["sg_window_size_for_NN_diff"].as<int>();
    sg_deg_for_NN_diff_ = optimization_param_node["optimization_parameter"]["sg_filter"]["sg_deg_for_NN_diff"].as<int>();

    YAML::Node trained_model_param_node = YAML::LoadFile(param_dir_path + "/trained_model_param.yaml");
    acc_queue_size_ = trained_model_param_node["trained_model_parameter"]["queue_size"]["acc_queue_size"].as<int>();
    steer_queue_size_ = trained_model_param_node["trained_model_parameter"]["queue_size"]["steer_queue_size"].as<int>();
    update_lstm_len_ = trained_model_param_node["trained_model_parameter"]["lstm"]["update_lstm_len"].as<int>();

    YAML::Node controller_param_node = YAML::LoadFile(param_dir_path + "/controller_param.yaml");

    double acc_time_delay_controller = controller_param_node["controller_parameter"]["acceleration"]["acc_time_delay"].as<double>();
    double acc_time_constant_controller = controller_param_node["controller_parameter"]["acceleration"]["acc_time_constant"].as<double>();
    double steer_time_delay_controller = controller_param_node["controller_parameter"]["steering"]["steer_time_delay"].as<double>();
    double steer_time_constant_controller = controller_param_node["controller_parameter"]["steering"]["steer_time_constant"].as<double>();
    acc_delay_step_controller_ = std::min(int(std::round(acc_time_delay_controller / control_dt_)), acc_queue_size_);
    steer_delay_step_controller_ = std::min(int(std::round(steer_time_delay_controller / control_dt_)), steer_queue_size_);

    nominal_dynamics_controller_.set_params(wheel_base_,acc_time_delay_controller,steer_time_delay_controller,acc_time_constant_controller,steer_time_constant_controller,acc_queue_size_,steer_queue_size_,control_dt_,predict_step_);
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
    use_vehicle_adaptor_ = optimization_param_node["optimization_parameter"]["autoware_alignment"]["use_vehicle_adaptor"].as<bool>();

    use_acc_input_schedule_prediction_ = optimization_param_node["optimization_parameter"]["inputs_schedule_prediction_NN"]["use_acc_input_schedule_prediction"].as<bool>();
    use_steer_input_schedule_prediction_ = optimization_param_node["optimization_parameter"]["inputs_schedule_prediction_NN"]["use_steer_input_schedule_prediction"].as<bool>();
    acc_input_schedule_prediction_len_ = optimization_param_node["optimization_parameter"]["inputs_schedule_prediction_NN"]["acc_input_schedule_prediction_len"].as<int>();
    steer_input_schedule_prediction_len_ = optimization_param_node["optimization_parameter"]["inputs_schedule_prediction_NN"]["steer_input_schedule_prediction_len"].as<int>();
    acc_input_schedule_prediction_len_ = std::min(acc_input_schedule_prediction_len_, horizon_len_ * predict_step_ -1);
    steer_input_schedule_prediction_len_ = std::min(steer_input_schedule_prediction_len_, horizon_len_ * predict_step_ -1);
    int controller_acc_input_history_len = optimization_param_node["optimization_parameter"]["inputs_schedule_prediction_NN"]["controller_acc_input_history_len"].as<int>();
    int controller_steer_input_history_len = optimization_param_node["optimization_parameter"]["inputs_schedule_prediction_NN"]["controller_steer_input_history_len"].as<int>();
    int adaptive_scale_index = optimization_param_node["optimization_parameter"]["inputs_schedule_prediction_NN"]["adaptive_scale_index"].as<int>();
    if (use_acc_input_schedule_prediction_ && !acc_input_schedule_prediction_initialized_){
      acc_input_schedule_prediction_.set_params(controller_acc_input_history_len, acc_input_schedule_prediction_len_ , control_dt_, "inputs_schedule_prediction_model/acc_schedule_predictor",adaptive_scale_index);
      acc_input_schedule_prediction_initialized_ = true;
    }
    if (use_steer_input_schedule_prediction_ && !steer_input_schedule_prediction_initialized_){
      steer_input_schedule_prediction_.set_params(controller_steer_input_history_len, steer_input_schedule_prediction_len_ , control_dt_, "inputs_schedule_prediction_model/steer_schedule_predictor",adaptive_scale_index);
      steer_input_schedule_prediction_initialized_ = true;
    }
  }
  void VehicleAdaptor::set_NN_params(
    const Eigen::MatrixXd & weight_acc_encoder_layer_1, const Eigen::MatrixXd & weight_steer_encoder_layer_1,
    const Eigen::MatrixXd & weight_acc_encoder_layer_2, const Eigen::MatrixXd & weight_steer_encoder_layer_2,
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_ih, const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_hh,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_encoder_layer_1, const Eigen::VectorXd & bias_steer_encoder_layer_1,
    const Eigen::VectorXd & bias_acc_encoder_layer_2, const Eigen::VectorXd & bias_steer_encoder_layer_2,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const std::vector<Eigen::VectorXd> & bias_lstm_encoder_ih, const std::vector<Eigen::VectorXd> & bias_lstm_encoder_hh,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted)
  {
    adaptor_ilqr_.set_NN_params(
      weight_acc_encoder_layer_1, weight_steer_encoder_layer_1, weight_acc_encoder_layer_2, weight_steer_encoder_layer_2,
      weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2,
      weight_lstm_encoder_ih, weight_lstm_encoder_hh, weight_lstm_ih, weight_lstm_hh,
      weight_complimentary_layer, weight_linear_relu, weight_final_layer,
      bias_acc_encoder_layer_1, bias_steer_encoder_layer_1, bias_acc_encoder_layer_2, bias_steer_encoder_layer_2,
      bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2, bias_steer_layer_2,
      bias_lstm_encoder_ih, bias_lstm_encoder_hh, bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer,
      bias_linear_relu, bias_final_layer, vel_scaling, vel_bias, state_component_predicted);
    h_dim_full_ = weight_lstm_encoder_hh[0].cols();
    h_dim_ = weight_lstm_hh.cols();
    num_layers_encoder_ = weight_lstm_encoder_ih.size();
    NN_prediction_target_dim_ = state_component_predicted.size();
  }
  void VehicleAdaptor::set_NN_params_from_csv(std::string csv_dir){
    // Eigen::MatrixXd weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2, weight_lstm_ih, weight_lstm_hh, weight_complimentary_layer, weight_linear_relu, weight_final_layer;
    // Eigen::VectorXd bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2, bias_steer_layer_2, bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer, bias_linear_relu, bias_final_layer;
    double vel_scaling, vel_bias;
    std::cout << "csv_dir: " << csv_dir << std::endl;
    Eigen::MatrixXd model_info = read_csv(csv_dir + "/model_info.csv");
    num_layers_encoder_ = model_info(0,0);
    Eigen::MatrixXd weight_acc_encoder_layer_1 = read_csv(csv_dir + "/weight_acc_encoder_layer_1.csv");
    Eigen::MatrixXd weight_steer_encoder_layer_1 = read_csv(csv_dir + "/weight_steer_encoder_layer_1.csv");
    Eigen::MatrixXd weight_acc_encoder_layer_2 = read_csv(csv_dir + "/weight_acc_encoder_layer_2.csv");
    Eigen::MatrixXd weight_steer_encoder_layer_2 = read_csv(csv_dir + "/weight_steer_encoder_layer_2.csv");

    Eigen::MatrixXd weight_acc_layer_1 = read_csv(csv_dir + "/weight_acc_layer_1.csv");
    Eigen::MatrixXd weight_steer_layer_1 = read_csv(csv_dir + "/weight_steer_layer_1.csv");
    Eigen::MatrixXd weight_acc_layer_2 = read_csv(csv_dir + "/weight_acc_layer_2.csv");
    Eigen::MatrixXd weight_steer_layer_2 = read_csv(csv_dir + "/weight_steer_layer_2.csv");
    //todo

    Eigen::MatrixXd weight_lstm_ih = read_csv(csv_dir + "/weight_lstm_ih.csv");
    Eigen::MatrixXd weight_lstm_hh = read_csv(csv_dir + "/weight_lstm_hh.csv");
    Eigen::MatrixXd weight_complimentary_layer = read_csv(csv_dir + "/weight_complimentary_layer.csv");
    Eigen::MatrixXd weight_linear_relu = read_csv(csv_dir + "/weight_linear_relu.csv");
    Eigen::MatrixXd weight_final_layer = read_csv(csv_dir + "/weight_final_layer.csv");

    Eigen::VectorXd bias_acc_encoder_layer_1 = read_csv(csv_dir + "/bias_acc_encoder_layer_1.csv").col(0);
    Eigen::VectorXd bias_steer_encoder_layer_1 = read_csv(csv_dir + "/bias_steer_encoder_layer_1.csv").col(0);
    Eigen::VectorXd bias_acc_encoder_layer_2 = read_csv(csv_dir + "/bias_acc_encoder_layer_2.csv").col(0);
    Eigen::VectorXd bias_steer_encoder_layer_2 = read_csv(csv_dir + "/bias_steer_encoder_layer_2.csv").col(0);

    Eigen::VectorXd bias_acc_layer_1 = read_csv(csv_dir + "/bias_acc_layer_1.csv").col(0);
    Eigen::VectorXd bias_steer_layer_1 = read_csv(csv_dir + "/bias_steer_layer_1.csv").col(0);
    Eigen::VectorXd bias_acc_layer_2 = read_csv(csv_dir + "/bias_acc_layer_2.csv").col(0);
    Eigen::VectorXd bias_steer_layer_2 = read_csv(csv_dir + "/bias_steer_layer_2.csv").col(0);
    Eigen::VectorXd bias_lstm_ih = read_csv(csv_dir + "/bias_lstm_ih.csv").col(0);
    Eigen::VectorXd bias_lstm_hh = read_csv(csv_dir + "/bias_lstm_hh.csv").col(0);
    Eigen::VectorXd bias_complimentary_layer = read_csv(csv_dir + "/bias_complimentary_layer.csv").col(0);
    Eigen::VectorXd bias_linear_relu = read_csv(csv_dir + "/bias_linear_relu.csv").col(0);
    Eigen::VectorXd bias_final_layer = read_csv(csv_dir + "/bias_final_layer.csv").col(0);
    std::vector<Eigen::MatrixXd> weight_lstm_encoder_ih(num_layers_encoder_);
    std::vector<Eigen::MatrixXd> weight_lstm_encoder_hh(num_layers_encoder_);
    std::vector<Eigen::VectorXd> bias_lstm_encoder_ih(num_layers_encoder_);
    std::vector<Eigen::VectorXd> bias_lstm_encoder_hh(num_layers_encoder_);
    for (int i = 0; i < num_layers_encoder_; i++){
      weight_lstm_encoder_ih[i] = read_csv(csv_dir + "/weight_lstm_encoder_ih_" + std::to_string(i) + ".csv");
      weight_lstm_encoder_hh[i] = read_csv(csv_dir + "/weight_lstm_encoder_hh_" + std::to_string(i) + ".csv");
      bias_lstm_encoder_ih[i] = read_csv(csv_dir + "/bias_lstm_encoder_ih_" + std::to_string(i) + ".csv").col(0);
      bias_lstm_encoder_hh[i] = read_csv(csv_dir + "/bias_lstm_encoder_hh_" + std::to_string(i) + ".csv").col(0);
    }
    Eigen::MatrixXd vel_params = read_csv(csv_dir + "/vel_scale.csv");

    vel_scaling = vel_params(0,0);
    vel_bias = vel_params(1,0);
    std::vector<std::string> state_component_predicted = read_string_csv(csv_dir + "/state_component_predicted.csv");
    set_NN_params(
      weight_acc_encoder_layer_1, weight_steer_encoder_layer_1, weight_acc_encoder_layer_2, weight_steer_encoder_layer_2,
      weight_acc_layer_1, weight_steer_layer_1, weight_acc_layer_2, weight_steer_layer_2,
      weight_lstm_encoder_ih, weight_lstm_encoder_hh,
      weight_lstm_ih, weight_lstm_hh, weight_complimentary_layer, weight_linear_relu,
      weight_final_layer,
      bias_acc_encoder_layer_1, bias_steer_encoder_layer_1, bias_acc_encoder_layer_2, bias_steer_encoder_layer_2,
      bias_acc_layer_1, bias_steer_layer_1, bias_acc_layer_2, bias_steer_layer_2,
      bias_lstm_encoder_ih, bias_lstm_encoder_hh,
      bias_lstm_ih, bias_lstm_hh, bias_complimentary_layer,
      bias_linear_relu, bias_final_layer, vel_scaling, vel_bias, state_component_predicted);
  }
  void VehicleAdaptor::clear_NN_params()
  {
    adaptor_ilqr_.clear_NN_params();
  }
  void VehicleAdaptor::set_offline_data_set_for_compensation(
    Eigen::MatrixXd XXT, Eigen::MatrixXd YXT
  )
  {
    adaptor_ilqr_.set_offline_data_set_for_compensation(XXT, YXT);
  }
  void VehicleAdaptor::set_offline_data_set_for_compensation_from_csv(std::string csv_dir)
  {
    Eigen::MatrixXd XXT = read_csv(csv_dir + "/XXT.csv");
    Eigen::MatrixXd YXT = read_csv(csv_dir + "/YXT.csv");
    set_offline_data_set_for_compensation(XXT, YXT);
  }
  void VehicleAdaptor::unset_offline_data_set_for_compensation()
  {
    adaptor_ilqr_.unset_offline_data_set_for_compensation();
  }
  void VehicleAdaptor::set_projection_matrix_for_compensation(
    Eigen::MatrixXd P
  )
  {
    adaptor_ilqr_.set_projection_matrix_for_compensation(P);
  }
  void VehicleAdaptor::unset_projection_matrix_for_compensation()
  {
    adaptor_ilqr_.unset_projection_matrix_for_compensation();
  }
  void VehicleAdaptor::set_projection_matrix_for_compensation_from_csv(std::string csv_dir)
  {
    Eigen::MatrixXd P = read_csv(csv_dir + "/Projection.csv");
    set_projection_matrix_for_compensation(P);
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
    int controller_acc_input_history_len = std::max({controller_acc_input_history_len_, acc_delay_step_ + 1, past_len_for_acc_linear_extrapolation_ + 1});
    int controller_steer_input_history_len = std::max({controller_steer_input_history_len_, steer_delay_step_ + 1,past_len_for_steer_linear_extrapolation_ + 1});
    if (!initialized_) {
      start_time_ = time_stamp;
      if (states_ref_mode_ == "predict_by_polynomial_regression"){
        set_params();
      }
      previous_error_ = Eigen::VectorXd::Zero(NN_prediction_target_dim_);

      adaptor_ilqr_.initialize_compensation(acc_queue_size_, steer_queue_size_, predict_step_, h_dim_full_);

      adaptor_ilqr_.set_sg_filter_params(sg_deg_for_NN_diff_,horizon_len_,sg_window_size_for_NN_diff_);

      max_queue_size_ = std::max({acc_queue_size_ + 1, steer_queue_size_ + 1, controller_acc_input_history_len, controller_steer_input_history_len, predict_step_*update_lstm_len_+2});
      time_stamp_obs_ = std::vector<double>(max_queue_size_);
      for (int i = 0; i < max_queue_size_; i++)
      {
        time_stamp_obs_[i] = time_stamp - (max_queue_size_ - i - 1) * control_dt_;
      }


      acc_input_history_ = acc_controller_input * Eigen::VectorXd::Ones(acc_queue_size_);
      steer_input_history_ = steer_controller_input * Eigen::VectorXd::Ones(steer_queue_size_);

      acc_controller_input_history_ = acc_controller_input * Eigen::VectorXd::Ones(controller_acc_input_history_len);
      steer_controller_input_history_ = steer_controller_input * Eigen::VectorXd::Ones(controller_steer_input_history_len);
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

      std::vector<double> time_stamp_controller_acc_input_history(controller_acc_input_history_len);
      for (int i = 0; i < controller_acc_input_history_len; i++)
      {
        time_stamp_controller_acc_input_history[i] = time_stamp - (controller_acc_input_history_len - i - 1) * control_dt_;
      }
      std::vector<double> time_stamp_controller_steer_input_history(controller_steer_input_history_len);
      for (int i = 0; i < controller_steer_input_history_len; i++)
      {
        time_stamp_controller_steer_input_history[i] = time_stamp - (controller_steer_input_history_len - i - 1) * control_dt_;
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
    Eigen::VectorXd h_lstm = Eigen::VectorXd::Zero(h_dim_full_);
    Eigen::VectorXd c_lstm = Eigen::VectorXd::Zero(h_dim_full_);
    std::vector<Eigen::VectorXd> h_lstm_encoder(num_layers_encoder_, Eigen::VectorXd::Zero(h_dim_full_));
    std::vector<Eigen::VectorXd> c_lstm_encoder(num_layers_encoder_, Eigen::VectorXd::Zero(h_dim_full_));
    Eigen::VectorXd h_lstm_compensation = Eigen::VectorXd::Zero(h_dim_full_);
    Eigen::VectorXd c_lstm_compensation = Eigen::VectorXd::Zero(h_dim_full_);
    Eigen::Vector2d acc_steer_error = Eigen::Vector2d::Zero();
    Eigen::VectorXd previous_error_compensator = Eigen::VectorXd::Zero(previous_error_.size());
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
      adaptor_ilqr_.update_state_queue_for_compensation(states_tmp);
      if (i == update_lstm_len_ - compensation_lstm_len_ - 5){
        h_lstm_compensation = h_lstm_encoder[num_layers_encoder_-1];
        c_lstm_compensation = c_lstm_encoder[num_layers_encoder_-1];
      }
      if (i < update_lstm_len_ - compensation_lstm_len_ - 5){
        adaptor_ilqr_.update_state_queue_for_compensation(states_tmp);
      }
      else if (i < update_lstm_len_ - compensation_lstm_len_){//update previous_error_compensator
        adaptor_ilqr_.F_with_model_without_compensation(states_tmp, acc_input_history_concat, steer_input_history_concat, h_lstm_compensation, c_lstm_compensation, previous_error_compensator, i);
      }
      else {
        Eigen::VectorXd lstm_error;
        if (i < update_lstm_len_ - 1){
          lstm_error = (state_history_lstm_[predict_step_*(i+1)]
                                 - adaptor_ilqr_.F_with_model_without_compensation(states_tmp, acc_input_history_concat, steer_input_history_concat, h_lstm_compensation, c_lstm_compensation, previous_error_compensator, i)) / predict_dt_;
        }
        else{
          lstm_error = (states
                                 - adaptor_ilqr_.F_with_model_without_compensation(states_tmp, acc_input_history_concat, steer_input_history_concat, h_lstm_compensation, c_lstm_compensation, previous_error_compensator, i)) / predict_dt_;
        }
        adaptor_ilqr_.update_one_step_for_compensation(acc_input_history_concat, steer_input_history_concat, h_lstm_compensation, c_lstm_compensation, lstm_error);
      }
      if (i < update_lstm_len_ - 1){
       acc_steer_error = (state_history_lstm_[predict_step_*(i+1)]
                                 - adaptor_ilqr_.nominal_prediction(states_tmp, acc_input_history_concat, steer_input_history_concat)).tail(2) / predict_dt_;
      }
      else{
        acc_steer_error = (states
                                 - adaptor_ilqr_.nominal_prediction(states_tmp, acc_input_history_concat, steer_input_history_concat)).tail(2) / predict_dt_;
      }
      adaptor_ilqr_.update_lstm_states(states_tmp, acc_input_history_concat, steer_input_history_concat, h_lstm_encoder, c_lstm_encoder, acc_steer_error);
    }
    adaptor_ilqr_.save_state_queue_for_compensation();
    adaptor_ilqr_.update_regression_matrix_for_compensation();
    h_lstm = h_lstm_encoder[num_layers_encoder_-1];
    c_lstm = c_lstm_encoder[num_layers_encoder_-1];

///////// calculate states_ref /////////
    double acc_rate_cost = acc_rate_cost_;
    double steer_rate_cost = steer_rate_cost_;
    std::vector<Eigen::VectorXd> states_ref(horizon_len_ + 1);

    // calculate controller input history with schedule //

    Eigen::VectorXd acc_controller_input_history_with_schedule = Eigen::VectorXd::Zero(controller_acc_input_history_len+predict_step_*horizon_len_-1);
    Eigen::VectorXd steer_controller_input_history_with_schedule = Eigen::VectorXd::Zero(controller_steer_input_history_len+predict_step_*horizon_len_-1);
    acc_controller_input_history_with_schedule.head(controller_acc_input_history_len) = acc_controller_input_history_;
    steer_controller_input_history_with_schedule.head(controller_steer_input_history_len) = steer_controller_input_history_;

    Eigen::VectorXd acc_controller_inputs_prediction_by_NN(horizon_len_ * predict_step_ - 1);
    Eigen::VectorXd steer_controller_inputs_prediction_by_NN(horizon_len_ * predict_step_ - 1);
    Eigen::VectorXd acc_input_schedule_predictor_states(2);
    Eigen::VectorXd steer_input_schedule_predictor_states(2);
    acc_input_schedule_predictor_states << states[vel_index_], acc_controller_input;
    steer_input_schedule_predictor_states << states[vel_index_], steer_controller_input;
    if (use_acc_input_schedule_prediction_)
    {
      std::vector<double> controller_acc_input_prediction_by_NN = acc_input_schedule_prediction_.get_inputs_schedule_predicted(acc_input_schedule_predictor_states, time_stamp);
      for (int i = 0; i < horizon_len_ * predict_step_ - 1; i++)
      {
        if (i < acc_input_schedule_prediction_len_)
        {
          acc_controller_inputs_prediction_by_NN[i] = controller_acc_input_prediction_by_NN[i];
        }
        else{
          acc_controller_inputs_prediction_by_NN[i] = acc_controller_inputs_prediction_by_NN[acc_input_schedule_prediction_len_ - 1];
        }
      }
    }
    if (use_steer_input_schedule_prediction_)
    {
      std::vector<double> controller_steer_input_prediction_by_NN = steer_input_schedule_prediction_.get_inputs_schedule_predicted(steer_input_schedule_predictor_states, time_stamp);
      for (int i = 0; i < horizon_len_ * predict_step_ - 1; i++)
      {
        if (i < steer_input_schedule_prediction_len_)
        {
          steer_controller_inputs_prediction_by_NN[i] = controller_steer_input_prediction_by_NN[i];
        }
        else{
          steer_controller_inputs_prediction_by_NN[i] = steer_controller_inputs_prediction_by_NN[steer_input_schedule_prediction_len_ - 1];
        }
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

      Eigen::VectorXd acc_controller_input_prediction(horizon_len_*predict_step_-1);
      if (use_acc_input_schedule_prediction_)
      {
        acc_controller_input_prediction = acc_controller_inputs_prediction_by_NN; 
      }
      else if (use_acc_linear_extrapolation_)
      {
        acc_controller_input_prediction.head(acc_linear_extrapolation_len_) = (acc_controller_input_history_[acc_controller_input_history_.size() - 1]
                                                                             - acc_controller_input_history_[acc_controller_input_history_.size() - past_len_for_acc_linear_extrapolation_ - 1]) / past_len_for_acc_linear_extrapolation_
                                                                              * Eigen::VectorXd::LinSpaced(acc_linear_extrapolation_len_, 1, acc_linear_extrapolation_len_) + acc_controller_input_history_[acc_controller_input_history_.size() - 1] * Eigen::VectorXd::Ones(acc_linear_extrapolation_len_);
        acc_controller_input_prediction.tail(horizon_len_*predict_step_ - acc_linear_extrapolation_len_ -1) = acc_controller_input_prediction[acc_linear_extrapolation_len_ - 1]*Eigen::VectorXd::Ones(horizon_len_*predict_step_ - acc_linear_extrapolation_len_ -1);
      }
      else{
        acc_controller_input_prediction.head(acc_polynomial_prediction_len_) = polynomial_reg_for_predict_acc_input_.predict(acc_controller_input_history_.tail(controller_acc_input_history_len_)).head(acc_polynomial_prediction_len_);
        acc_controller_input_prediction.tail(horizon_len_*predict_step_ - acc_polynomial_prediction_len_ -1) = acc_controller_input_prediction[acc_polynomial_prediction_len_ - 1]*Eigen::VectorXd::Ones(horizon_len_*predict_step_ - acc_polynomial_prediction_len_ -1);
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
            acc_controller_input_history_with_schedule[controller_acc_input_history_len - 1 + i*predict_step_ + j] = acc_controller_input_history_with_schedule[controller_acc_input_history_len - 1 + i*predict_step_ + j -1] + control_dt_ * acc_controller_d_inputs_schedule_[i];
            steer_controller_input_history_with_schedule[controller_steer_input_history_len - 1 + i*predict_step_ + j] = steer_controller_input_history_with_schedule[controller_steer_input_history_len - 1 + i*predict_step_ + j -1] + control_dt_ * steer_controller_d_inputs_schedule_[i];
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
      Eigen::VectorXd acc_controller_input_prediction(horizon_len_*predict_step_-1);
      Eigen::VectorXd steer_controller_input_prediction(horizon_len_*predict_step_-1);
      if (use_acc_input_schedule_prediction_)
      {
        acc_controller_input_prediction = acc_controller_inputs_prediction_by_NN; 
      }
      else if (use_acc_linear_extrapolation_)
      {
        acc_controller_input_prediction.head(acc_linear_extrapolation_len_) = (acc_controller_input_history_[acc_controller_input_history_.size() - 1]
                                                                             - acc_controller_input_history_[acc_controller_input_history_.size() - past_len_for_acc_linear_extrapolation_ - 1]) / past_len_for_acc_linear_extrapolation_
                                                                              * Eigen::VectorXd::LinSpaced(acc_linear_extrapolation_len_, 1, acc_linear_extrapolation_len_) + acc_controller_input_history_[acc_controller_input_history_.size() - 1] * Eigen::VectorXd::Ones(acc_linear_extrapolation_len_);
        acc_controller_input_prediction.tail(horizon_len_*predict_step_ - acc_linear_extrapolation_len_ -1) = acc_controller_input_prediction[acc_linear_extrapolation_len_ - 1]*Eigen::VectorXd::Ones(horizon_len_*predict_step_ - acc_linear_extrapolation_len_ -1);
      }
      else{
        acc_controller_input_prediction.head(acc_polynomial_prediction_len_) = polynomial_reg_for_predict_acc_input_.predict(acc_controller_input_history_.tail(controller_acc_input_history_len_)).head(acc_polynomial_prediction_len_);
        acc_controller_input_prediction.tail(horizon_len_*predict_step_ - acc_polynomial_prediction_len_ -1) = acc_controller_input_prediction[acc_polynomial_prediction_len_ - 1]*Eigen::VectorXd::Ones(horizon_len_*predict_step_ - acc_polynomial_prediction_len_ -1);
      }
      if (use_steer_input_schedule_prediction_)
      {
        steer_controller_input_prediction = steer_controller_inputs_prediction_by_NN; 
      }
      else if (use_steer_linear_extrapolation_)
      {
        steer_controller_input_prediction.head(steer_linear_extrapolation_len_) = (steer_controller_input_history_[steer_controller_input_history_.size() - 1]
                                                                             - steer_controller_input_history_[steer_controller_input_history_.size() - past_len_for_steer_linear_extrapolation_ - 1]) / past_len_for_steer_linear_extrapolation_
                                                                              * Eigen::VectorXd::LinSpaced(steer_linear_extrapolation_len_, 1, steer_linear_extrapolation_len_) + steer_controller_input_history_[steer_controller_input_history_.size() - 1] * Eigen::VectorXd::Ones(steer_linear_extrapolation_len_);
        steer_controller_input_prediction.tail(horizon_len_*predict_step_ - steer_linear_extrapolation_len_ -1) = steer_controller_input_prediction[steer_linear_extrapolation_len_ - 1]*Eigen::VectorXd::Ones(horizon_len_*predict_step_ - steer_linear_extrapolation_len_ -1);
      }
      else{
        steer_controller_input_prediction.head(steer_polynomial_prediction_len_) = polynomial_reg_for_predict_steer_input_.predict(steer_controller_input_history_.tail(controller_steer_input_history_len_)).head(steer_polynomial_prediction_len_);
        steer_controller_input_prediction.tail(horizon_len_*predict_step_ - steer_polynomial_prediction_len_ -1) = steer_controller_input_prediction[steer_polynomial_prediction_len_ - 1]*Eigen::VectorXd::Ones(horizon_len_*predict_step_ - steer_polynomial_prediction_len_ -1);
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
          inputs_tmp << acc_controller_input_history_with_schedule[controller_acc_input_history_len + i*predict_step_ + j - acc_delay_step_controller_ - 1], steer_controller_input_history_with_schedule[controller_steer_input_history_len + i*predict_step_ + j - steer_delay_step_controller_ - 1];
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
    /*
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
      states_tmp_compensation = adaptor_ilqr_.F_with_model_without_compensation(states_tmp_compensation, acc_input_history_concat, steer_input_history_concat, h_lstm_compensation, c_lstm_compensation, previous_error_tmp, i) + compensation;
    }
    Eigen::VectorXd compensation_error_vector = (states - states_tmp_compensation)/compensation_lstm_len_;
    adaptor_ilqr_.update_regression_matrix_for_compensation(compensation_error_vector);
    */
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
    .def("set_offline_data_set_for_compensation", &VehicleAdaptor::set_offline_data_set_for_compensation)
    .def("set_offline_data_set_for_compensation_from_csv", &VehicleAdaptor::set_offline_data_set_for_compensation_from_csv)
    .def("unset_offline_data_set_for_compensation", &VehicleAdaptor::unset_offline_data_set_for_compensation)
    .def("set_projection_matrix_for_compensation", &VehicleAdaptor::set_projection_matrix_for_compensation)
    .def("unset_projection_matrix_for_compensation", &VehicleAdaptor::unset_projection_matrix_for_compensation)
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
