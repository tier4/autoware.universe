#ifndef LINEAR_REGRESSION_COMPENSATOR_H
#define LINEAR_REGRESSION_COMPENSATOR_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "inputs_prediction.h"

///////////////// LinearRegressionCompensator /////////////////////// 
class LinearRegressionCompensator
{
private:
  double lambda_, lambda_bias_, default_forgetting_factor_, min_forgetting_factor_;
  bool fit_yaw_ = false;
  int x_index_ = 0;
  int y_index_ = 1;
  int vel_index_ = 2;
  int yaw_index_ = 3;
  int acc_index_ = 4;
  int steer_index_ = 5;
  int state_size_ = 6;
  double vel_scale_;
  bool is_offline_data_set_ = false;
  bool is_projection_matrix_set_ = false;
  int projected_feature_size_;
  Eigen::MatrixXd projection_matrix_;
  double sigma_p_narrow_, sigma_p_wide_, sigma_e_narrow_, sigma_e_wide_, sigma_p_offline_, sigma_e_offline_;
  Eigen::MatrixXd XXT_, YXT_, XXT_inv_;
  Eigen::MatrixXd XXT_online_, YXT_online_, XXT_new_, YXT_new_, XXT_offline_, YXT_offline_;
  Eigen::MatrixXd XXT_offline_inv_, regression_matrix_offline_;
  Eigen::MatrixXd regression_matrix_;
  std::vector<Eigen::VectorXd> states_queue_, states_queue_saved_;
  Eigen::MatrixXd State_Inputs_;
  int num_samples_ = 0;
  bool use_compensator_;
  double max_yaw_compensation_, max_acc_compensation_, max_steer_compensation_;
  double default_weight_offline_ratio_;
  int x_dim_, x_history_len_, acc_queue_size_, steer_queue_size_, h_dim_, feature_size_, target_size_;
  int predict_step_;
  double adaptive_forgetting_factor_, adaptive_weight_offline_ratio_;
  double current_sigma_p_2_mean_, current_sigma_e_2_mean_;
  double current_sigma_p_2_offline_mean_, current_sigma_e_2_offline_mean_;
  double coef_narrow_, coef_wide_;
public:
  LinearRegressionCompensator();
  virtual ~LinearRegressionCompensator();
  void initialize(int acc_queue_size, int steer_queue_size, int predict_step, int h_dim);
  void set_offline_data_set(Eigen::MatrixXd XXT, Eigen::MatrixXd YXT);
  void set_offline_data_from_csv(std::string csv_dir);
  void unset_offline_data_set();
  void set_projection_matrix(Eigen::MatrixXd projection_matrix);
  void unset_projection_matrix();
  void update_state_queue(Eigen::VectorXd states);
  void save_state_queue();
  void load_state_queue();
  Eigen::VectorXd get_regression_state_input();
  Eigen::VectorXd get_regression_input(Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm);
  void initialize_for_candidates(int num_candidates);
  void update_one_step(Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm, Eigen::VectorXd error_vector);
  void update_regression_matrix();
  Eigen::VectorXd predict(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm);
  Eigen::MatrixXd Predict(Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat, Eigen::MatrixXd H_lstm, Eigen::MatrixXd C_lstm);
};

#endif // LINEAR_REGRESSION_COMPENSATOR_H