#ifndef INPUTS_PREDICTION_H
#define INPUTS_PREDICTION_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>


Eigen::VectorXd tanh(const Eigen::VectorXd & v);
Eigen::VectorXd d_tanh(const Eigen::VectorXd & v);
Eigen::VectorXd sigmoid(const Eigen::VectorXd & v);
Eigen::VectorXd relu(const Eigen::VectorXd & x);
Eigen::VectorXd d_relu(const Eigen::VectorXd & x);
Eigen::MatrixXd d_relu_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x);
Eigen::MatrixXd d_tanh_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x);
Eigen::VectorXd d_tanh_product_vec(const Eigen::VectorXd & v, const Eigen::VectorXd & x);
Eigen::MatrixXd d_sigmoid_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x);
Eigen::VectorXd d_sigmoid_product_vec(const Eigen::VectorXd & v, const Eigen::VectorXd & x);
Eigen::VectorXd rotate_data(Eigen::VectorXd states, double yaw);
Eigen::VectorXd vector_power(Eigen::VectorXd vec, int power);
double double_power(double val, int power);
double calc_table_value(double domain_value, std::vector<double> domain_table, std::vector<double> target_table);
std::string get_param_dir_path();
Eigen::MatrixXd read_csv(std::string file_path);
std::vector<std::string> read_string_csv(std::string file_path);
Eigen::VectorXd interpolate_eigen(Eigen::VectorXd y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new);
std::vector<Eigen::VectorXd> interpolate_vector(std::vector<Eigen::VectorXd> y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new);

class InputsSchedulePrediction
{
private:
  Eigen::MatrixXd weight_lstm_encoder_ih_0_;
  Eigen::MatrixXd weight_lstm_encoder_hh_0_;
  Eigen::MatrixXd weight_lstm_decoder_ih_0_;
  Eigen::MatrixXd weight_lstm_decoder_hh_0_;
  Eigen::MatrixXd weight_lstm_encoder_ih_1_;
  Eigen::MatrixXd weight_lstm_encoder_hh_1_;
  Eigen::MatrixXd weight_lstm_decoder_ih_1_;
  Eigen::MatrixXd weight_lstm_decoder_hh_1_;
  Eigen::MatrixXd weight_post_decoder_0_;
  Eigen::MatrixXd weight_post_decoder_1_;
  Eigen::MatrixXd weight_final_layer_;
  Eigen::VectorXd bias_lstm_encoder_ih_0_;
  Eigen::VectorXd bias_lstm_encoder_hh_0_;
  Eigen::VectorXd bias_lstm_decoder_ih_0_;
  Eigen::VectorXd bias_lstm_decoder_hh_0_;
  Eigen::VectorXd bias_lstm_encoder_ih_1_;
  Eigen::VectorXd bias_lstm_encoder_hh_1_;
  Eigen::VectorXd bias_lstm_decoder_ih_1_;
  Eigen::VectorXd bias_lstm_decoder_hh_1_;
  Eigen::VectorXd bias_post_decoder_0_;
  Eigen::VectorXd bias_post_decoder_1_;
  Eigen::VectorXd bias_final_layer_;
  Eigen::VectorXd adaptive_scale_;
  double vel_scaling_, vel_bias_;

  int vel_index_ = 0;
  int predict_step_ = 3;
  int input_step_,output_step_;
  int h_dim_;
  double control_dt_ = 0.033;
  bool initialized_ = false;
  std::vector<Eigen::VectorXd> x_seq_obs_;
  int input_rate_lim_;
  std::vector<double> timestamps_obs_;
  Eigen::VectorXd limit_scaling_;
  int augmented_input_size_;
public:
  InputsSchedulePrediction();
  virtual ~InputsSchedulePrediction();
  void set_params(
    int input_step, int output_step, double control_dt, std::string csv_dir, const int & adaptive_scale_index=-1);
  void set_NN_params(
    const Eigen::MatrixXd & weight_lstm_encoder_ih_0, const Eigen::MatrixXd & weight_lstm_encoder_hh_0,
    const Eigen::MatrixXd & weight_lstm_decoder_ih_0, const Eigen::MatrixXd & weight_lstm_decoder_hh_0,
    const Eigen::MatrixXd & weight_lstm_encoder_ih_1, const Eigen::MatrixXd & weight_lstm_encoder_hh_1,
    const Eigen::MatrixXd & weight_lstm_decoder_ih_1, const Eigen::MatrixXd & weight_lstm_decoder_hh_1,
    const Eigen::MatrixXd & weight_post_decoder_0,
    const Eigen::MatrixXd & weight_post_decoder_1,
    const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_lstm_encoder_ih_0, const Eigen::VectorXd & bias_lstm_encoder_hh_0,
    const Eigen::VectorXd & bias_lstm_decoder_ih_0, const Eigen::VectorXd & bias_lstm_decoder_hh_0,
    const Eigen::VectorXd & bias_lstm_encoder_ih_1, const Eigen::VectorXd & bias_lstm_encoder_hh_1,
    const Eigen::VectorXd & bias_lstm_decoder_ih_1, const Eigen::VectorXd & bias_lstm_decoder_hh_1,
    const Eigen::VectorXd & bias_post_decoder_0,
    const Eigen::VectorXd & bias_post_decoder_1,
    const Eigen::VectorXd & bias_final_layer,
    const Eigen::VectorXd & adaptive_scale,
    int input_step, int output_step, double input_rate_lim,
    double vel_scaling, double vel_bias, double control_dt);
  void update_encoder_cells(
    const Eigen::VectorXd & x, Eigen::VectorXd & h_lstm_encoder_0, Eigen::VectorXd & h_lstm_encoder_1, Eigen::VectorXd & c_lstm_encoder_0, Eigen::VectorXd & c_lstm_encoder_1
  );
  void get_encoder_cells(
    const std::vector<Eigen::VectorXd> & x_seq, Eigen::VectorXd & h_lstm_encoder_0, Eigen::VectorXd & c_lstm_encoder_0, Eigen::VectorXd & h_lstm_encoder_1, Eigen::VectorXd & c_lstm_encoder_1
  );
  Eigen::VectorXd update_decoder_cells(
    const Eigen::VectorXd & decoder_input, Eigen::VectorXd & h_lstm_decoder_0, Eigen::VectorXd & c_lstm_decoder_0, Eigen::VectorXd & h_lstm_decoder_1, Eigen::VectorXd & c_lstm_decoder_1
  );
  std::vector<double> predict(const std::vector<Eigen::VectorXd> & x_seq);
  std::vector<double> get_inputs_schedule_predicted(
    const Eigen::VectorXd & x, double timestamp
  );
  void send_initialized_flag();
};
#endif // INPUTS_PREDICTION_H