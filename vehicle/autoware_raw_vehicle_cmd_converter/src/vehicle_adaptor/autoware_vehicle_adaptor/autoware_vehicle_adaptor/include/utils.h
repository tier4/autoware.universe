#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "inputs_prediction.h"
#include "nominal_dynamics.h"
#include "transform_vehicle_adaptor_model.h"
#include <pybind11/stl.h>
#include "linear_regression_compensator.h"

Eigen::VectorXd states_vehicle_to_world(Eigen::VectorXd states_vehicle, double yaw);
Eigen::VectorXd states_world_to_vehicle(Eigen::VectorXd states_world, double yaw);
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
class PolynomialRegression
{
private:
  int degree_ = 2;
  int num_samples_ = 10;
  std::vector<double> lambda_ = {0.0, 0.0};
  Eigen::MatrixXd weight_matrix_, coef_matrix_, prediction_matrix_;
  bool predict_and_ignore_intercept_ = false;
  double minimum_decay_ = 1.0;
  // Template function to handle both Eigen::MatrixXd and Eigen::VectorXd
  template <typename T>
  std::vector<T> fit_transform_impl(const std::vector<T>& raw_data) {
      std::vector<T> result;
      for (int i = 0; i < num_samples_; i++) {
          T vec_tmp = T::Zero(raw_data[0].rows(), raw_data[0].cols());
          for (int j = 0; j < num_samples_; j++) {
              vec_tmp += weight_matrix_(j, i) * raw_data[j];
          }
          result.push_back(vec_tmp);
      }
      return result;
  }
public:
  PolynomialRegression();
  virtual ~PolynomialRegression();
  void set_params(int degree, int num_samples, std::vector<double> lambda);
  void set_minimum_decay(double minimum_decay);
  void set_ignore_intercept();
  void calc_coef_matrix();
  void calc_prediction_matrix(int horizon_len);
  Eigen::VectorXd fit_transform(Eigen::VectorXd vec);
  std::vector<Eigen::MatrixXd> fit_transform(const std::vector<Eigen::MatrixXd>& raw_data);
  std::vector<Eigen::VectorXd> fit_transform(const std::vector<Eigen::VectorXd>& raw_data);
  Eigen::VectorXd predict(Eigen::VectorXd vec);
};


class PolynomialFilter
{
private:
  int degree_ = 2;
  int num_samples_ = 10;
  double lambda_ = 0.0;
  double minimum_decay_ = 1.0;
  std::vector<double> timestamps_;
  std::vector<double> samples_;

public:
  PolynomialFilter();
  virtual ~PolynomialFilter();
  void set_params(int degree, int num_samples, double lambda, double minimum_decay);
  double fit_transform(double timestamp, double sample);
};

class SgFilter
{
private:
  int degree_;
  int window_size_;
  std::vector<Eigen::VectorXd> sg_vector_left_edge_, sg_vector_right_edge_;
  Eigen::VectorXd sg_vector_center_;

  // Template function to handle both Eigen::MatrixXd and Eigen::VectorXd
  template <typename T>
  std::vector<T> sg_filter_impl(const std::vector<T>& raw_data) {
      int input_len = raw_data.size();
      std::vector<T> result;
      for (int i = 0; i < input_len; i++) {
          T tmp_filtered_vector = T::Zero(raw_data[0].rows(), raw_data[0].cols());
          if (i < window_size_) {
              for (int j = 0; j < i + window_size_ + 1; j++) {
                  tmp_filtered_vector += sg_vector_left_edge_[i][j] * raw_data[j];
              }
          }
          else if (i > input_len - window_size_ - 1) {
              for (int j = 0; j < input_len - i + window_size_; j++) {
                  tmp_filtered_vector += sg_vector_right_edge_[input_len - i - 1][j] * raw_data[i - window_size_ + j];
              }
          }
          else {
              for (int j = 0; j < 2 * window_size_ + 1; j++) {
                  tmp_filtered_vector += sg_vector_center_[j] * raw_data[i - window_size_ + j];
              }
          }
          result.push_back(tmp_filtered_vector);
      }
      return result;
  }
public:
  SgFilter();
  virtual ~SgFilter();
  void set_params(int degree, int window_size);
  void calc_sg_filter_weight();
  std::vector<Eigen::MatrixXd> sg_filter(const std::vector<Eigen::MatrixXd>& raw_data);
  std::vector<Eigen::VectorXd> sg_filter(const std::vector<Eigen::VectorXd>& raw_data);
};
class FilterDiffNN
{
private:
  int state_size_;
  int h_dim_;
  int acc_queue_size_;
  int steer_queue_size_;
  int predict_step_;
  double control_dt_;
  SgFilter sg_filter_;
public:
  FilterDiffNN();
  virtual ~FilterDiffNN();
  void set_sg_filter_params(int degree, int window_size, int state_size, int h_dim, int acc_queue_size, int steer_queue_size, int predict_step, double control_dt);
  void fit_transform_for_NN_diff(
    std::vector<Eigen::MatrixXd> A, std::vector<Eigen::MatrixXd> B, std::vector<Eigen::MatrixXd> C,std::vector<Eigen::MatrixXd> & dF_d_states,
    std::vector<Eigen::MatrixXd> & dF_d_inputs);

};
class ButterworthFilter
{
private:
  int order_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<Eigen::VectorXd> x_;
  std::vector<Eigen::VectorXd> y_;
  bool initialized_ = false;
public:
  ButterworthFilter();
  virtual ~ButterworthFilter();
  void set_params();
  Eigen::VectorXd apply(Eigen::VectorXd input_value);
};
/*
class LinearRegressionCompensator
{
private:
  double lambda_ = 0.0;
  double lambda_bias_ = 0.0;
  double decay_ = 0.9;
  bool fit_yaw_ = false;
  int x_index_ = 0;
  int y_index_ = 1;
  int vel_index_ = 2;
  int yaw_index_ = 3;
  int acc_index_ = 4;
  int steer_index_ = 5;
  int state_size_ = 6;
  Eigen::MatrixXd XXT_, YXT_, XXT_inv_;
  Eigen::MatrixXd regression_matrix_;
  std::vector<Eigen::VectorXd> input_queue_ = {};
  bool use_compensator_ = true;
  double max_yaw_compensation_, max_acc_compensation_, max_steer_compensation_;
public:
  LinearRegressionCompensator();
  virtual ~LinearRegressionCompensator();
  void initialize();
  void update_input_queue(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat);
  void update_regression_matrix(Eigen::VectorXd error_vector);
  Eigen::VectorXd predict(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat);
  Eigen::MatrixXd Predict(Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat);
  Eigen::VectorXd get_bias();
};
*/
class TrainedDynamics
{
private:
  NominalDynamics nominal_dynamics_;
  TransformModelToEigen transform_model_to_eigen_;
  double wheel_base_ = 2.79;
  double acc_time_delay_ = 0.1;
  double steer_time_delay_ = 0.27;
  double acc_time_constant_ = 0.1;
  double steer_time_constant_ = 0.24;
  int acc_queue_size_ = 15;
  int steer_queue_size_ = 15;

  int x_index_ = 0;
  int y_index_ = 1;
  int vel_index_ = 2;
  int yaw_index_ = 3;
  int acc_index_ = 4;
  int steer_index_ = 5;
  int state_size_ = 6;

  int acc_input_start_index_ = 6;
  int acc_input_end_index_ = 6 + acc_queue_size_ - 1;
  int steer_input_start_index_ = 6 + acc_queue_size_;
  int steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
  double control_dt_ = 0.033;
  int predict_step_ = 3;
  double predict_dt_ = control_dt_ * predict_step_;

  double error_decay_rate_ = 0.5;

  int acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
  int steer_delay_step_ =
    std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);

  FilterDiffNN filter_diff_NN_;
  int h_dim_;
  int h_dim_full_;
  int state_dim_ = 6;
  double minimum_steer_diff_ = 0.03;
  double minimum_acc_diff_ = 0.1;

  std::vector<std::string> all_state_name_ = {"x", "y", "vel", "yaw", "acc", "steer"};
  std::vector<std::string> state_component_predicted_ = {"acc", "steer"};
  std::vector<int> state_component_predicted_index_;
  LinearRegressionCompensator linear_regression_compensation_;
public:
  TrainedDynamics();
  virtual ~TrainedDynamics();
  void set_vehicle_params(
    double wheel_base, double acc_time_delay, double steer_time_delay, double acc_time_constant,
    double steer_time_constant, int acc_queue_size, int steer_queue_size, double control_dt,
    int predict_step);
  void set_NN_params(
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
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted);
  void clear_NN_params();
  void set_sg_filter_params(int degree, int window_size);
  Eigen::VectorXd nominal_prediction(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat);
  void update_lstm_states(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat,
    std::vector<Eigen::VectorXd> & h_lstm, std::vector<Eigen::VectorXd> & c_lstm, const Eigen::Vector2d & acc_steer_error);
  void initialize_compensation(int acc_queue_size, int steer_queue_size, int predict_step, int h_dim);
  void update_state_queue_for_compensation(Eigen::VectorXd states);
  void update_one_step_for_compensation(Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm, Eigen::VectorXd error_vector);
  void update_regression_matrix_for_compensation();
  void save_state_queue_for_compensation();
  void load_state_queue_for_compensation();
  void initialize_for_candidates_compensation(int num_candidates);
  void set_offline_data_set_for_compensation(Eigen::MatrixXd XXT, Eigen::MatrixXd YYT);  
  void unset_offline_data_set_for_compensation();
  void set_projection_matrix_for_compensation(Eigen::MatrixXd projection_matrix);
  void unset_projection_matrix_for_compensation();
  Eigen::VectorXd prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm
  );
  Eigen::MatrixXd Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat, Eigen::MatrixXd H_lstm, Eigen::MatrixXd C_lstm
  );
  /*
  void update_input_queue_for_compensation(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat);
  void update_regression_matrix_for_compensation(Eigen::VectorXd error_vector);
  Eigen::VectorXd prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat
  );
  Eigen::MatrixXd Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat
  );
  Eigen::VectorXd get_compensation_bias();
  */
  Eigen::VectorXd F_with_model_for_calc_controller_prediction_error(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon);
  Eigen::VectorXd F_with_model_without_compensation(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon);
  Eigen::VectorXd F_with_model(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat,
    const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon);
  Eigen::VectorXd F_with_model(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon);
  Eigen::VectorXd F_with_model_diff(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat,
    const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon,
    Eigen::MatrixXd & A, Eigen::MatrixXd & B, Eigen::MatrixXd & C);
  Eigen::VectorXd F_with_model_diff(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon,
    Eigen::MatrixXd & A, Eigen::MatrixXd & B, Eigen::MatrixXd & C);
  Eigen::MatrixXd F_with_model_for_candidates(
    const Eigen::MatrixXd & States, Eigen::MatrixXd & Acc_input_history,
    Eigen::MatrixXd & Steer_input_history, 
    Eigen::MatrixXd & Acc_input_history_concat, Eigen::MatrixXd & Steer_input_history_concat,
    const Eigen::MatrixXd & D_inputs,
    Eigen::MatrixXd & H_lstm, Eigen::MatrixXd & C_lstm, Eigen::MatrixXd & Previous_error, const int horizon);
  Eigen::MatrixXd F_with_model_for_candidates(
    const Eigen::MatrixXd & States, Eigen::MatrixXd & Acc_input_history,
    Eigen::MatrixXd & Steer_input_history, const Eigen::MatrixXd & D_inputs,
    Eigen::MatrixXd & H_lstm, Eigen::MatrixXd & C_lstm, Eigen::MatrixXd & Previous_error, const int horizon);
  void calc_forward_trajectory_with_diff(Eigen::VectorXd states, Eigen::VectorXd acc_input_history,
                                         Eigen::VectorXd steer_input_history, std::vector<Eigen::VectorXd> d_inputs_schedule,
                                         const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
                                         const Eigen::VectorXd & previous_error, std::vector<Eigen::VectorXd> & states_prediction,
                                         std::vector<Eigen::MatrixXd> & dF_d_states, std::vector<Eigen::MatrixXd> & dF_d_inputs,
                                         std::vector<Eigen::Vector2d> & inputs_schedule);
};
class AdaptorILQR
{
private:
  double wheel_base_ = 2.79;
  double acc_time_delay_ = 0.1;
  double steer_time_delay_ = 0.27;
  double acc_time_constant_ = 0.1;
  double steer_time_constant_ = 0.24;
  int acc_queue_size_ = 15;
  int steer_queue_size_ = 15;

  int x_index_ = 0;
  int y_index_ = 1;
  int vel_index_ = 2;
  int yaw_index_ = 3;
  int acc_index_ = 4;
  int steer_index_ = 5;
  int state_size_ = 6;

  int acc_input_start_index_ = 6;
  int acc_input_end_index_ = 6 + acc_queue_size_ - 1;
  int steer_input_start_index_ = 6 + acc_queue_size_;
  int steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
  double control_dt_ = 0.033;
  int predict_step_ = 3;
  double predict_dt_ = control_dt_ * predict_step_;
  
  double error_decay_rate_ = 0.5;

  int acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
  int steer_delay_step_ =
    std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);

  int horizon_len_ = 10;
  int h_dim_;
  int state_dim_ = 6;
  TrainedDynamics trained_dynamics_;
  
  double x_cost_ = 0.0;
  double y_cost_ = 0.0;
  double vel_cost_ = 0.0;
  double yaw_cost_ = 1.0;

  double steer_cost_ = 10.0;
  double acc_cost_ = 1.0;
  double steer_rate_cost_ = 0.1;
  double acc_rate_cost_ = 1.0;
  
  double x_terminal_cost_ = 0.0;
  double y_terminal_cost_ = 0.0;
  double vel_terminal_cost_ = 0.0;
  double yaw_terminal_cost_ = 1.0;
  
  double steer_terminal_cost_ = 10.0;
  double acc_terminal_cost_ = 10.0;

  double steer_rate_rate_cost_ = 0.1;
  double acc_rate_rate_cost_ = 1.0;

  int intermediate_cost_index_ = 5;
  double x_intermediate_cost_ = 0.0;
  double y_intermediate_cost_ = 0.0;
  double vel_intermediate_cost_ = 0.0;
  double yaw_intermediate_cost_ = 1.0;
  double steer_intermediate_cost_ = 10.0;
  double acc_intermediate_cost_ = 10.0;
  std::vector<std::string> all_state_name_ = {"x", "y", "vel", "yaw", "acc", "steer"};
  std::vector<std::string> state_component_ilqr_ = {"vel", "acc", "steer"};
  std::vector<int> state_component_ilqr_index_;
  int num_state_component_ilqr_;

  std::vector<double> acc_input_weight_target_table_, longitudinal_coef_target_table_, longitudinal_coef_by_vel_error_domain_table_, longitudinal_coef_by_acc_error_domain_table_;
  std::vector<double> steer_input_weight_target_table_, lateral_coef_target_table_, lateral_coef_by_lateral_error_domain_table_, lateral_coef_by_yaw_error_domain_table_, lateral_coef_by_steer_error_domain_table_;
public:
  AdaptorILQR();
  virtual ~AdaptorILQR();
  void set_params();
  void set_states_cost(
    double x_cost, double y_cost, double vel_cost, double yaw_cost, double acc_cost, double steer_cost
  );
  void set_inputs_cost(
     double acc_rate_cost, double steer_rate_cost
  );
  void set_rate_cost(
    double acc_rate_rate_cost, double steer_rate_rate_cost
  );
  void set_intermediate_cost(
    double x_intermediate_cost, double y_intermediate_cost, double vel_intermediate_cost, double yaw_intermediate_cost, double acc_intermediate_cost, double steer_intermediate_cost, int intermediate_cost_index
  );
  void set_terminal_cost(
    double x_terminal_cost, double y_terminal_cost, double vel_terminal_cost, double yaw_terminal_cost, double acc_terminal_cost, double steer_terminal_cost
  );
  void set_vehicle_params(
    double wheel_base, double acc_time_delay, double steer_time_delay, double acc_time_constant,
    double steer_time_constant, int acc_queue_size, int steer_queue_size, double control_dt,
    int predict_step);
  void set_NN_params(
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
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted);
  void clear_NN_params();
  void set_sg_filter_params(int degree, int horizon_len,int window_size);
  Eigen::VectorXd nominal_prediction(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat);
  void update_lstm_states(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat,
    std::vector<Eigen::VectorXd> & h_lstm, std::vector<Eigen::VectorXd> & c_lstm, const Eigen::Vector2d & previous_error);
  Eigen::VectorXd F_with_model_without_compensation(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, int horizon);
  void initialize_compensation(int acc_queue_size, int steer_queue_size, int predict_step, int h_dim);
  void update_state_queue_for_compensation(Eigen::VectorXd states);
  void update_one_step_for_compensation(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm);
  void update_regression_matrix_for_compensation();
  void save_state_queue_for_compensation();
  void load_state_queue_for_compensation();
  void set_offline_data_set_for_compensation(Eigen::MatrixXd XXT, Eigen::MatrixXd YYT);
  void unset_offline_data_set_for_compensation();
  void set_projection_matrix_for_compensation(Eigen::MatrixXd projection_matrix);
  void unset_projection_matrix_for_compensation();
  Eigen::VectorXd prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm
  );
  /*
  Eigen::MatrixXd Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat, Eigen::MatrixXd H_lstm, Eigen::MatrixXd C_lstm
  );
  void update_input_queue_for_compensation(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat);
  void update_regression_matrix_for_compensation(Eigen::VectorXd error_vector);
  Eigen::VectorXd prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat
  );
  Eigen::MatrixXd Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat
  );
  Eigen::VectorXd get_compensation_bias();
  */
  void calc_forward_trajectory_with_cost(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history, Eigen::VectorXd steer_input_history,
    std::vector<Eigen::MatrixXd> D_inputs_schedule, const Eigen::VectorXd & h_lstm,
    const Eigen::VectorXd & c_lstm,
    const Eigen::VectorXd & previous_error, std::vector<Eigen::MatrixXd> & states_prediction,
    const std::vector<Eigen::VectorXd> & states_ref,
    const std::vector<Eigen::VectorXd> & d_input_ref, 
    std::vector<Eigen::Vector2d> inputs_ref, double longitudinal_coef, double lateral_coef, double acc_input_weight, double steer_input_weight,
    Eigen::VectorXd & Cost);
  void calc_inputs_ref_info(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history,
    const Eigen::VectorXd & steer_input_history, const Eigen::VectorXd & h_lstm,
    const Eigen::VectorXd & c_lstm, 
    const Eigen::VectorXd & previous_error, const std::vector<Eigen::VectorXd> & states_ref,
    const Eigen::VectorXd & acc_controller_input_schedule, const Eigen::VectorXd & steer_controller_input_schedule,
    std::vector<Eigen::Vector2d> & inputs_ref, double & longitudinal_coef, double & lateral_coef,  double & acc_input_weight, double & steer_input_weight);
  Eigen::MatrixXd extract_dF_d_state(Eigen::MatrixXd dF_d_state_with_history);
  Eigen::MatrixXd extract_dF_d_input(Eigen::MatrixXd dF_d_input);
  Eigen::MatrixXd right_action_by_state_diff_with_history(Eigen::MatrixXd Mat, Eigen::MatrixXd dF_d_state_with_history);
  Eigen::MatrixXd left_action_by_state_diff_with_history(Eigen::MatrixXd dF_d_state_with_history, Eigen::MatrixXd Mat);
  Eigen::MatrixXd right_action_by_input_diff(Eigen::MatrixXd Mat, Eigen::MatrixXd dF_d_input);
  Eigen::MatrixXd left_action_by_input_diff(Eigen::MatrixXd dF_d_input, Eigen::MatrixXd Mat);
  void compute_ilqr_coefficients(
    const std::vector<Eigen::MatrixXd> & dF_d_states, const std::vector<Eigen::MatrixXd> & dF_d_inputs,
    const std::vector<Eigen::VectorXd> & states_prediction, const std::vector<Eigen::VectorXd> & d_inputs_schedule,
    const std::vector<Eigen::VectorXd> & states_ref,
    const std::vector<Eigen::VectorXd> & d_input_ref, const double prev_acc_rate, const double prev_steer_rate,
    std::vector<Eigen::Vector2d> inputs_ref, double longitudinal_coef, double lateral_coef, double acc_input_weight, double steer_input_weight, const std::vector<Eigen::Vector2d> & inputs_schedule,
    std::vector<Eigen::MatrixXd> & K, std::vector<Eigen::VectorXd> & k);
  std::vector<Eigen::MatrixXd> calc_line_search_candidates(std::vector<Eigen::MatrixXd> K, std::vector<Eigen::VectorXd> k, std::vector<Eigen::MatrixXd> dF_d_states, std::vector<Eigen::MatrixXd> dF_d_inputs,  std::vector<Eigen::VectorXd> d_inputs_schedule, Eigen::VectorXd ls_points);
  void compute_optimal_control(Eigen::VectorXd states, Eigen::VectorXd acc_input_history,
                              Eigen::VectorXd steer_input_history, std::vector<Eigen::VectorXd> & d_inputs_schedule,
                              const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
                              const std::vector<Eigen::VectorXd> & states_ref,
                              const std::vector<Eigen::VectorXd> & d_input_ref, Eigen::VectorXd & previous_error,
                              std::vector<Eigen::VectorXd> & states_prediction,
                              const Eigen::VectorXd & acc_controller_input_schedule, const Eigen::VectorXd & steer_controller_input_schedule,
                              double & acc_input, double & steer_input);

};
namespace Proxima{
class VehicleAdaptor
{
private:
  NominalDynamics nominal_dynamics_controller_;
  AdaptorILQR adaptor_ilqr_;
  PolynomialRegression polynomial_reg_for_predict_acc_input_, polynomial_reg_for_predict_steer_input_;
  SgFilter sg_filter_for_d_inputs_schedule_;
  ButterworthFilter butterworth_filter_;
  PolynomialFilter polynomial_filter_for_acc_inputs_, polynomial_filter_for_steer_inputs_;
  InputsSchedulePrediction acc_input_schedule_prediction_, steer_input_schedule_prediction_;
  int sg_window_size_for_d_inputs_schedule_, sg_deg_for_d_inputs_schedule_;
  bool use_sg_for_d_inputs_schedule_;
  double wheel_base_ = 2.79;
  double acc_time_delay_ = 0.1;
  double steer_time_delay_ = 0.27;
  double acc_time_constant_ = 0.1;
  double steer_time_constant_ = 0.24;
  int acc_queue_size_ = 15;
  int steer_queue_size_ = 15;
  double steer_dead_band_ = 0.0012;

  int x_index_ = 0;
  int y_index_ = 1;
  int vel_index_ = 2;
  int yaw_index_ = 3;
  int acc_index_ = 4;
  int steer_index_ = 5;

  int acc_input_start_index_ = 6;
  int acc_input_end_index_ = 6 + acc_queue_size_ - 1;
  int steer_input_start_index_ = 6 + acc_queue_size_;
  int steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
  double control_dt_ = 0.033;
  int predict_step_ = 3;
  double predict_dt_ = control_dt_ * predict_step_;

  double error_decay_rate_ = 0.5;

  int acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
  int steer_delay_step_ =
    std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);

  int horizon_len_ = 10;
  int sg_deg_for_NN_diff_ = 2;
  int sg_window_size_for_NN_diff_ = 5;
  int h_dim_;
  int h_dim_full_;
  int state_dim_ = 6;
  double x_cost_ = 0.0;
  double y_cost_ = 0.0;
  double vel_cost_ = 0.0;
  double yaw_cost_ = 1.0;
  double steer_cost_ = 10.0;
  double acc_cost_ = 1.0;
  double steer_rate_cost_ = 0.1;
  double acc_rate_cost_ = 1.0;
  double x_terminal_cost_ = 0.0;
  double y_terminal_cost_ = 0.0;
  double vel_terminal_cost_ = 0.0;
  double yaw_terminal_cost_ = 1.0;
  double steer_terminal_cost_ = 10.0;
  double acc_terminal_cost_ = 10.0;
  double x_intermediate_cost_ = 0.0;
  double y_intermediate_cost_ = 0.0;
  double vel_intermediate_cost_ = 0.0;
  double yaw_intermediate_cost_ = 1.0;
  double acc_intermediate_cost_ = 10.0;
  double steer_intermediate_cost_ = 10.0;
  int intermediate_cost_index_ = 5;

  int NN_prediction_target_dim_;


  std::vector<double> steer_rate_input_table_ = {0.0001,0.001,0.01};
  std::vector<double> steer_rate_cost_coef_table_ = {100000.0,100.0,1.0};
  //std::vector<double> steer_rate_cost_coef_table = {1.0,1.0,1.0,1.0};
  std::vector<double> acc_rate_input_table_ = {0.001,0.01,0.1};
  std::vector<double> acc_rate_cost_coef_table_ = {100.0,10.0,1.0};


  bool initialized_ = false;
  Eigen::VectorXd acc_input_history_;
  Eigen::VectorXd steer_input_history_;
  Eigen::VectorXd acc_controller_input_history_, steer_controller_input_history_;
  std::vector<Eigen::VectorXd> d_inputs_schedule_;
  std::vector<Eigen::VectorXd> state_history_lstm_, acc_input_history_lstm_, steer_input_history_lstm_;

  int controller_acc_input_history_len_, controller_steer_input_history_len_;
  int deg_controller_acc_input_history_, deg_controller_steer_input_history_;
  std::vector<double> lam_controller_acc_input_history_, lam_controller_steer_input_history_;
  double minimum_decay_controller_acc_input_history_, minimum_decay_controller_steer_input_history_;
  int acc_polynomial_prediction_len_, steer_polynomial_prediction_len_;

  bool use_acc_linear_extrapolation_ = false;
  bool use_steer_linear_extrapolation_ = false;
  int acc_linear_extrapolation_len_ = 15;
  int steer_linear_extrapolation_len_ = 15;
  int past_len_for_acc_linear_extrapolation_ = 3;
  int past_len_for_steer_linear_extrapolation_ = 3;


  int update_lstm_len_ = 50;
  int compensation_lstm_len_ = 10;

  Eigen::VectorXd previous_error_;


  std::string states_ref_mode_ = "predict_by_polynomial_regression";

  Eigen::VectorXd acc_controller_d_inputs_schedule_, steer_controller_d_inputs_schedule_;
  Eigen::VectorXd x_controller_prediction_, y_controller_prediction_, vel_controller_prediction_, yaw_controller_prediction_, acc_controller_prediction_, steer_controller_prediction_;

  double reflect_controller_d_input_ratio_ = 0.5;
  std::vector<double> time_stamp_obs_;
  std::vector<double> acc_input_history_obs_, steer_input_history_obs_, acc_controller_input_history_obs_, steer_controller_input_history_obs_;
  std::vector<Eigen::VectorXd> state_history_lstm_obs_, acc_input_history_lstm_obs_, steer_input_history_lstm_obs_;
  int max_queue_size_;

  bool use_controller_inputs_as_target_;
  std::vector<Eigen::VectorXd> states_prediction_;
  std::vector<double> mix_ratio_vel_target_table_, mix_ratio_vel_domain_table_;
  std::vector<double> mix_ratio_time_target_table_, mix_ratio_time_domain_table_;

  std::string input_filter_mode_ = "none";
  bool steer_controller_prediction_aided_ = false;
  double start_time_;
  bool use_acc_input_schedule_prediction_ = false;
  bool use_steer_input_schedule_prediction_ = false;
  int acc_input_schedule_prediction_len_, steer_input_schedule_prediction_len_;

  int num_layers_encoder_;
  bool acc_input_schedule_prediction_initialized_ = false;
  bool steer_input_schedule_prediction_initialized_ = false;
public:
  bool use_controller_steer_input_schedule_ = false;
  bool use_vehicle_adaptor_;
  VehicleAdaptor();
  virtual ~VehicleAdaptor();
  void set_params();
  void set_NN_params(
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
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted);
  void set_NN_params_from_csv(std::string csv_dir);
  void clear_NN_params();
  void set_offline_data_set_for_compensation(Eigen::MatrixXd XXT, Eigen::MatrixXd YYT);
  void set_offline_data_set_for_compensation_from_csv(std::string csv_dir);
  void unset_offline_data_set_for_compensation();
  void set_projection_matrix_for_compensation(Eigen::MatrixXd projection_matrix);
  void unset_projection_matrix_for_compensation();
  void set_projection_matrix_for_compensation_from_csv(std::string csv_dir);
  void set_controller_d_inputs_schedule(const Eigen::VectorXd & acc_controller_d_inputs_schedule, const Eigen::VectorXd & steer_controller_d_inputs_schedule);
  void set_controller_d_steer_schedule(const Eigen::VectorXd & steer_controller_d_inputs_schedule);
  void set_controller_steer_input_schedule(double timestamp, const std::vector<double> & steer_controller_input_schedule);
  void set_controller_prediction(const Eigen::VectorXd & x_controller_prediction, const Eigen::VectorXd & y_controller_prediction, const Eigen::VectorXd & vel_controller_prediction, const Eigen::VectorXd & yaw_controller_prediction, const Eigen::VectorXd & acc_controller_prediction, const Eigen::VectorXd & steer_controller_prediction);
  void set_controller_steer_prediction(const Eigen::VectorXd & steer_controller_prediction);
  Eigen::VectorXd get_adjusted_inputs(
    double time_stamp, const Eigen::VectorXd & states, const double acc_controller_input, const double steer_controller_input);
  Eigen::MatrixXd get_states_prediction();
  Eigen::MatrixXd get_d_inputs_schedule();
  void send_initialized_flag();
};
}
#endif // UTILS_H