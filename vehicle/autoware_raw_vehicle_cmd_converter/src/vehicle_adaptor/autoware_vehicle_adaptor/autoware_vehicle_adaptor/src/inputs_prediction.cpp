#include <Eigen/Core>
#include <Eigen/Dense>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cmath>
#include "inputs_prediction.h"

namespace py = pybind11;

Eigen::VectorXd tanh(const Eigen::VectorXd & v)
{
  return v.array().tanh();
}
Eigen::VectorXd sigmoid(const Eigen::VectorXd & v)
{
  return 0.5 * (0.5 * v).array().tanh() + 0.5;
}
Eigen::VectorXd relu(const Eigen::VectorXd & x)
{
  Eigen::VectorXd x_relu = x;
  for (int i = 0; i < x.size(); i++) {
    if (x[i] < 0) {
      x_relu[i] = 0;
    }
  }
  return x_relu;
}
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
  InputsSchedulePrediction::InputsSchedulePrediction() {}
  InputsSchedulePrediction::~InputsSchedulePrediction() {}
  void InputsSchedulePrediction::set_params(int input_step, int output_step, double control_dt, std::string csv_dir){
    Eigen::MatrixXd weight_pre_encoder_0 = read_csv(csv_dir + "/weight_pre_encoder_0.csv");
    Eigen::MatrixXd weight_pre_encoder_1 = read_csv(csv_dir + "/weight_pre_encoder_1.csv");
    Eigen::MatrixXd weight_lstm_encoder_ih = read_csv(csv_dir + "/weight_lstm_encoder_ih.csv");
    Eigen::MatrixXd weight_lstm_encoder_hh = read_csv(csv_dir + "/weight_lstm_encoder_hh.csv");
    Eigen::MatrixXd weight_lstm_decoder_ih = read_csv(csv_dir + "/weight_lstm_decoder_ih.csv");
    Eigen::MatrixXd weight_lstm_decoder_hh = read_csv(csv_dir + "/weight_lstm_decoder_hh.csv");
    Eigen::MatrixXd weight_post_decoder_0 = read_csv(csv_dir + "/weight_post_decoder_0.csv");
    Eigen::MatrixXd weight_post_decoder_1 = read_csv(csv_dir + "/weight_post_decoder_1.csv");
    Eigen::MatrixXd weight_final_layer = read_csv(csv_dir + "/weight_final_layer.csv");
    Eigen::VectorXd bias_pre_encoder_0 = read_csv(csv_dir + "/bias_pre_encoder_0.csv").col(0);
    Eigen::VectorXd bias_pre_encoder_1 = read_csv(csv_dir + "/bias_pre_encoder_1.csv").col(0);
    Eigen::VectorXd bias_lstm_encoder_ih = read_csv(csv_dir + "/bias_lstm_encoder_ih.csv").col(0);
    Eigen::VectorXd bias_lstm_encoder_hh = read_csv(csv_dir + "/bias_lstm_encoder_hh.csv").col(0);
    Eigen::VectorXd bias_lstm_decoder_ih = read_csv(csv_dir + "/bias_lstm_decoder_ih.csv").col(0);
    Eigen::VectorXd bias_lstm_decoder_hh = read_csv(csv_dir + "/bias_lstm_decoder_hh.csv").col(0);
    Eigen::VectorXd bias_post_decoder_0 = read_csv(csv_dir + "/bias_post_decoder_0.csv").col(0);
    Eigen::VectorXd bias_post_decoder_1 = read_csv(csv_dir + "/bias_post_decoder_1.csv").col(0);
    Eigen::VectorXd bias_final_layer = read_csv(csv_dir + "/bias_final_layer.csv").col(0);
    Eigen::VectorXd adaptive_scale = read_csv(csv_dir + "/adaptive_scale.csv").col(0);
    Eigen::MatrixXd vel_params = read_csv(csv_dir + "/vel_params.csv");
    double vel_scaling = vel_params(0, 0);
    double vel_bias = vel_params(1, 0);
    Eigen::MatrixXd limits = read_csv(csv_dir + "/limits.csv");
    double jerk_lim= limits(0, 0);
    double steer_rate_lim = limits(1, 0);
    set_NN_params(
      weight_pre_encoder_0, weight_pre_encoder_1,
      weight_lstm_encoder_ih, weight_lstm_encoder_hh,
      weight_lstm_decoder_ih, weight_lstm_decoder_hh,
      weight_post_decoder_0, weight_post_decoder_1,
      weight_final_layer, bias_pre_encoder_0,
      bias_pre_encoder_1, bias_lstm_encoder_ih,
      bias_lstm_encoder_hh, bias_lstm_decoder_ih,
      bias_lstm_decoder_hh, bias_post_decoder_0,
      bias_post_decoder_1, bias_final_layer,
      adaptive_scale,
      input_step, output_step, jerk_lim, steer_rate_lim,
      vel_scaling, vel_bias, control_dt);
      initialized_ = false;
  }
  void InputsSchedulePrediction::set_NN_params(
    const Eigen::MatrixXd & weight_pre_encoder_0, const Eigen::MatrixXd & weight_pre_encoder_1,
    const Eigen::MatrixXd & weight_lstm_encoder_ih, const Eigen::MatrixXd & weight_lstm_encoder_hh,
    const Eigen::MatrixXd & weight_lstm_decoder_ih, const Eigen::MatrixXd & weight_lstm_decoder_hh,
    const Eigen::MatrixXd & weight_post_decoder_0, const Eigen::MatrixXd & weight_post_decoder_1,
    const Eigen::MatrixXd & weight_final_layer, const Eigen::VectorXd & bias_pre_encoder_0,
    const Eigen::VectorXd & bias_pre_encoder_1, const Eigen::VectorXd & bias_lstm_encoder_ih,
    const Eigen::VectorXd & bias_lstm_encoder_hh, const Eigen::VectorXd & bias_lstm_decoder_ih,
    const Eigen::VectorXd & bias_lstm_decoder_hh, const Eigen::VectorXd & bias_post_decoder_0,
    const Eigen::VectorXd & bias_post_decoder_1, const Eigen::VectorXd & bias_final_layer,
    const Eigen::VectorXd & adaptive_scale,
    int input_step, int output_step, double jerk_lim, double steer_rate_lim,
    double vel_scaling, double vel_bias, double control_dt)
  {
    weight_pre_encoder_0_ = weight_pre_encoder_0;
    weight_pre_encoder_1_ = weight_pre_encoder_1;
    weight_lstm_encoder_ih_ = weight_lstm_encoder_ih;
    weight_lstm_encoder_hh_ = weight_lstm_encoder_hh;
    weight_lstm_decoder_ih_ = weight_lstm_decoder_ih;
    weight_lstm_decoder_hh_ = weight_lstm_decoder_hh;
    weight_post_decoder_0_ = weight_post_decoder_0;
    weight_post_decoder_1_ = weight_post_decoder_1;
    weight_final_layer_ = weight_final_layer;
    bias_pre_encoder_0_ = bias_pre_encoder_0;
    bias_pre_encoder_1_ = bias_pre_encoder_1;
    bias_lstm_encoder_ih_ = bias_lstm_encoder_ih;
    bias_lstm_encoder_hh_ = bias_lstm_encoder_hh;
    bias_lstm_decoder_ih_ = bias_lstm_decoder_ih;
    bias_lstm_decoder_hh_ = bias_lstm_decoder_hh;
    bias_post_decoder_0_ = bias_post_decoder_0;
    bias_post_decoder_1_ = bias_post_decoder_1;
    bias_final_layer_ = bias_final_layer;
    adaptive_scale_ = adaptive_scale;
    input_step_ = input_step;
    output_step_ = output_step;
    vel_scaling_ = vel_scaling;
    vel_bias_ = vel_bias;
    jerk_lim_ = jerk_lim;
    steer_rate_lim_ = steer_rate_lim;
    limit_scaling_ = Eigen::VectorXd(2);
    limit_scaling_ << jerk_lim, steer_rate_lim;

    h_dim_ = weight_lstm_encoder_hh.cols();
    control_dt_ = control_dt;
  }
  void InputsSchedulePrediction::update_encoder_cells(
    const Eigen::VectorXd & x, Eigen::VectorXd & h_lstm_encoder, Eigen::VectorXd & c_lstm_encoder
  ){
    Eigen::VectorXd x_scaled = x;
    x_scaled[vel_index_] = (x[vel_index_] - vel_bias_) * vel_scaling_;
    Eigen::VectorXd h_pre_encoder = relu(weight_pre_encoder_0_ * x_scaled + bias_pre_encoder_0_);
    h_pre_encoder = relu(weight_pre_encoder_1_ * h_pre_encoder + bias_pre_encoder_1_);
    Eigen::VectorXd i_f_g_o = weight_lstm_encoder_ih_ * h_pre_encoder + bias_lstm_encoder_ih_ + weight_lstm_encoder_hh_ * h_lstm_encoder + bias_lstm_encoder_hh_;
    Eigen::VectorXd i_lstm_encoder = sigmoid(i_f_g_o.segment(0, h_dim_));
    Eigen::VectorXd f_lstm_encoder = sigmoid(i_f_g_o.segment(h_dim_, h_dim_));
    Eigen::VectorXd g_lstm_encoder = tanh(i_f_g_o.segment(2 * h_dim_, h_dim_));
    Eigen::VectorXd o_lstm_encoder = sigmoid(i_f_g_o.segment(3 * h_dim_, h_dim_));
    c_lstm_encoder = f_lstm_encoder.array() * c_lstm_encoder.array() + i_lstm_encoder.array() * g_lstm_encoder.array();
    h_lstm_encoder = o_lstm_encoder.array() * tanh(c_lstm_encoder).array();
  }
  void InputsSchedulePrediction::get_encoder_cells(
    const std::vector<Eigen::VectorXd> & x_seq, Eigen::VectorXd & h_lstm_encoder, Eigen::VectorXd & c_lstm_encoder
  ){
    h_lstm_encoder = Eigen::VectorXd::Zero(h_dim_);
    c_lstm_encoder = Eigen::VectorXd::Zero(h_dim_);
    for (int i = 0; i < int(x_seq.size()); i++) {
      update_encoder_cells(x_seq[i], h_lstm_encoder, c_lstm_encoder);
    }
  }
  Eigen::VectorXd InputsSchedulePrediction::update_decoder_cells(
    const Eigen::VectorXd & decoder_input, Eigen::VectorXd & h_lstm_decoder, Eigen::VectorXd & c_lstm_decoder
  ){
    Eigen::VectorXd i_f_g_o = weight_lstm_decoder_ih_ * decoder_input + bias_lstm_decoder_ih_ + weight_lstm_decoder_hh_ * h_lstm_decoder + bias_lstm_decoder_hh_;
    Eigen::VectorXd i_lstm_decoder = sigmoid(i_f_g_o.segment(0, h_dim_));
    Eigen::VectorXd f_lstm_decoder = sigmoid(i_f_g_o.segment(h_dim_, h_dim_));
    Eigen::VectorXd g_lstm_decoder = tanh(i_f_g_o.segment(2 * h_dim_, h_dim_));
    Eigen::VectorXd o_lstm_decoder = sigmoid(i_f_g_o.segment(3 * h_dim_, h_dim_));
    c_lstm_decoder = f_lstm_decoder.array() * c_lstm_decoder.array() + i_lstm_decoder.array() * g_lstm_decoder.array();
    h_lstm_decoder = o_lstm_decoder.array() * tanh(c_lstm_decoder).array();
    Eigen::VectorXd h_decoder = relu(weight_post_decoder_0_ * h_lstm_decoder + bias_post_decoder_0_);
    Eigen::VectorXd h_decoder_scaled = h_decoder.array() * adaptive_scale_.array();
    h_decoder_scaled = relu(weight_post_decoder_1_ * h_decoder_scaled + bias_post_decoder_1_);
    Eigen::VectorXd output = weight_final_layer_ * h_decoder_scaled + bias_final_layer_;
    //output[0] = jerk_lim_ * output[0];
    //output[1] = steer_rate_lim_ * output[1];
    return output;
  }
  std::vector<Eigen::VectorXd> InputsSchedulePrediction::predict(
    const std::vector<Eigen::VectorXd> & x_seq
  ){
    Eigen::VectorXd h_lstm_encoder, c_lstm_encoder;
    get_encoder_cells(x_seq, h_lstm_encoder, c_lstm_encoder);
    //Eigen::VectorXd decoder_output = (x_seq[x_seq.size() - 1].tail(2) - x_seq[x_seq.size() - 2].tail(2)) / control_dt_;
    Eigen::VectorXd output(4);
    output.head(2) = x_seq[x_seq.size() - 1].tail(2);
    output.tail(2) = (x_seq[x_seq.size() - 1].tail(2) - x_seq[x_seq.size() - 2].tail(2)) / control_dt_;
    Eigen::VectorXd h_lstm_decoder = h_lstm_encoder;
    Eigen::VectorXd c_lstm_decoder = c_lstm_encoder;
    std::vector<Eigen::VectorXd> output_seq;
    for (int i = 0; i < output_step_; i++) {
      Eigen::VectorXd decoder_output = tanh(update_decoder_cells(output, h_lstm_decoder, c_lstm_decoder));
      decoder_output[0] = jerk_lim_ * decoder_output[0];
      decoder_output[1] = steer_rate_lim_ * decoder_output[1];
      output.head(2) += decoder_output * control_dt_;
      output.tail(2) = decoder_output;
      //output += decoder_output * control_dt_;
      output_seq.push_back(output.head(2));
    }
    return output_seq;
  }
  std::vector<Eigen::VectorXd> InputsSchedulePrediction::get_inputs_schedule_predicted(
    const Eigen::VectorXd & x, double timestamp
  )
  {
    if (!initialized_) {
        x_seq_obs_.resize(input_step_);
        timestamps_obs_.resize(input_step_);
        for (int i = 0; i < input_step_; i++) {
          x_seq_obs_[i] = x;
          timestamps_obs_[i] = timestamp - control_dt_ * (input_step_ - i -1);
        }
        initialized_ = true;
    }
    else{
        x_seq_obs_.push_back(x);
        timestamps_obs_.push_back(timestamp);
        if (timestamp - timestamps_obs_[1] > control_dt_ * (input_step_ - 1)) {
          x_seq_obs_.erase(x_seq_obs_.begin());
          timestamps_obs_.erase(timestamps_obs_.begin());
        }
    }
    std::vector<double> timestamp_for_prediction(input_step_);
    for (int i = 0; i < input_step_; i++) {
      timestamp_for_prediction[i] = timestamp - control_dt_ * (input_step_ - i - 1);
    }
    std::vector<Eigen::VectorXd> x_seq = interpolate_vector(x_seq_obs_, timestamps_obs_, timestamp_for_prediction);
    return predict(x_seq);
  }
  void InputsSchedulePrediction::send_initialized_flag(){
    initialized_ = false;
  }

PYBIND11_MODULE(inputs_prediction, m)
{
  py::class_<InputsSchedulePrediction>(m, "InputsSchedulePrediction")
    .def(py::init())
    .def("set_params", &InputsSchedulePrediction::set_params)
    .def("get_inputs_schedule_predicted", &InputsSchedulePrediction::get_inputs_schedule_predicted)
    .def("send_initialized_flag", &InputsSchedulePrediction::send_initialized_flag);
}


