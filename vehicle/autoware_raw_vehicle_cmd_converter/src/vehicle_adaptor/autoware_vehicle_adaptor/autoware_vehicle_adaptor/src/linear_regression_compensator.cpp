#include <yaml-cpp/yaml.h>
#include "linear_regression_compensator.h"



///////////////// LinearRegressionCompensator ///////////////////////

  LinearRegressionCompensator::LinearRegressionCompensator() {
  }
  
  LinearRegressionCompensator::~LinearRegressionCompensator() {}
  void LinearRegressionCompensator::initialize(int acc_queue_size, int steer_queue_size, int predict_step, int h_dim){
    YAML::Node optimization_param_node = YAML::LoadFile(get_param_dir_path() + "/optimization_param.yaml");
    lambda_ = optimization_param_node["optimization_parameter"]["compensation"]["lambda"].as<double>();
    lambda_bias_ = optimization_param_node["optimization_parameter"]["compensation"]["lambda_bias"].as<double>();
    default_forgetting_factor_ = optimization_param_node["optimization_parameter"]["compensation"]["default_forgetting_factor"].as<double>();
    min_forgetting_factor_ = optimization_param_node["optimization_parameter"]["compensation"]["min_forgetting_factor"].as<double>();
    fit_yaw_ = optimization_param_node["optimization_parameter"]["compensation"]["fit_yaw"].as<bool>();
    use_compensator_ = optimization_param_node["optimization_parameter"]["compensation"]["use_compensator"].as<bool>();
    max_yaw_compensation_ = optimization_param_node["optimization_parameter"]["compensation"]["max_yaw_compensation"].as<double>();
    max_acc_compensation_ = optimization_param_node["optimization_parameter"]["compensation"]["max_acc_compensation"].as<double>();
    max_steer_compensation_ = optimization_param_node["optimization_parameter"]["compensation"]["max_steer_compensation"].as<double>();
    default_weight_offline_ratio_ = optimization_param_node["optimization_parameter"]["compensation"]["default_weight_offline_ratio"].as<double>();
    x_dim_ = 3;
    x_history_len_ = optimization_param_node["optimization_parameter"]["compensation"]["x_history_len"].as<int>();
    vel_scale_ = optimization_param_node["optimization_parameter"]["compensation"]["vel_scale"].as<double>();
    coef_wide_ = optimization_param_node["optimization_parameter"]["compensation"]["coef_wide"].as<double>();
    coef_narrow_ = optimization_param_node["optimization_parameter"]["compensation"]["coef_narrow"].as<double>();

    predict_step_ = predict_step;
    acc_queue_size_ = acc_queue_size + predict_step_;
    steer_queue_size_ = steer_queue_size + predict_step_;
    h_dim_ = h_dim;
    feature_size_ = 1 + x_dim_ * x_history_len_ + acc_queue_size_ + steer_queue_size_ + 2 * h_dim_;
    if (!is_projection_matrix_set_){
      projected_feature_size_ = feature_size_;
    }
    target_size_ = fit_yaw_ ? 3 : 2; // if fit_yaw_ is true, target_size_ = 3, otherwise target_size_ = 2
    XXT_ = Eigen::MatrixXd::Zero(projected_feature_size_, projected_feature_size_);
    YXT_ = Eigen::MatrixXd::Zero(target_size_, projected_feature_size_);
    XXT_inv_ = Eigen::MatrixXd::Zero(projected_feature_size_, projected_feature_size_);
    XXT_online_ = Eigen::MatrixXd::Zero(projected_feature_size_, projected_feature_size_);
    YXT_online_ = Eigen::MatrixXd::Zero(target_size_, projected_feature_size_);
    XXT_new_ = Eigen::MatrixXd::Zero(projected_feature_size_, projected_feature_size_);
    YXT_new_ = Eigen::MatrixXd::Zero(target_size_, projected_feature_size_);
    if (!is_offline_data_set_){
      XXT_offline_ = Eigen::MatrixXd::Zero(projected_feature_size_, projected_feature_size_);
      YXT_offline_ = Eigen::MatrixXd::Zero(target_size_, projected_feature_size_);
      regression_matrix_offline_ = Eigen::MatrixXd::Zero(target_size_, projected_feature_size_);
      XXT_offline_inv_ = Eigen::MatrixXd::Zero(projected_feature_size_, projected_feature_size_);
    }
    regression_matrix_ = Eigen::MatrixXd::Zero(target_size_, projected_feature_size_);
    states_queue_ = {};
    sigma_p_narrow_ = 1.0;
    sigma_p_wide_ = 1.0;
    sigma_e_narrow_ = 1.0;
    sigma_e_wide_ = 1.0;
    sigma_p_offline_ = 1.0;
    sigma_e_offline_ = 1.0;
    current_sigma_p_2_mean_ = 0.0;
    current_sigma_e_2_mean_ = 0.0;
    current_sigma_p_2_offline_mean_ = 0.0;
    current_sigma_e_2_offline_mean_ = 0.0;
    adaptive_forgetting_factor_ = default_forgetting_factor_;
    adaptive_weight_offline_ratio_ = default_weight_offline_ratio_;
  }
  void LinearRegressionCompensator::set_offline_data_set(Eigen::MatrixXd XXT, Eigen::MatrixXd YXT){
    is_offline_data_set_ = true;
    if (int(XXT.rows()) == 2){
      fit_yaw_ = false;
    }
    else{
      fit_yaw_ = true;
    }
    XXT_offline_ = XXT;
    YXT_offline_ = YXT;
    Eigen::MatrixXd regularized_matrix = lambda_ * Eigen::MatrixXd::Identity(XXT_offline_.rows(), XXT_offline_.cols());
    regularized_matrix(0, 0) = lambda_bias_;
    Eigen::MatrixXd XXT_offline_reg = XXT_offline_ + regularized_matrix;
    XXT_offline_inv_ = XXT_offline_reg.llt().solve(Eigen::MatrixXd::Identity(XXT_offline_.rows(), XXT_offline_.cols()));
    //(XXT_offline_ + regularized_matrix).inverse();
  }
  void LinearRegressionCompensator::set_offline_data_from_csv(std::string csv_dir){
    Eigen::MatrixXd XXT = read_csv(csv_dir + "/XXT.csv");
    Eigen::MatrixXd YXT = read_csv(csv_dir + "/YXT.csv");
    set_offline_data_set(XXT, YXT);
  }
  void LinearRegressionCompensator::unset_offline_data_set(){
    is_offline_data_set_ = false;
  }
  void LinearRegressionCompensator::set_projection_matrix(Eigen::MatrixXd projection_matrix){
    is_projection_matrix_set_ = true;
    projection_matrix_ = projection_matrix;
    projected_feature_size_ = projection_matrix_.rows() + 1;
  }
  void LinearRegressionCompensator::unset_projection_matrix(){
    is_projection_matrix_set_ = false;
  }
  void LinearRegressionCompensator::update_state_queue(Eigen::VectorXd states){
    states_queue_.push_back(states);
    if (int(states_queue_.size()) > x_history_len_){
      states_queue_.erase(states_queue_.begin());
    }
  }
  void LinearRegressionCompensator::save_state_queue(){
    if (!use_compensator_){
      return;
    }
    states_queue_saved_ = states_queue_;
  }
  void LinearRegressionCompensator::load_state_queue(){
    if (!use_compensator_){
      return;
    }
    states_queue_ = states_queue_saved_;
  }
  Eigen::VectorXd LinearRegressionCompensator::get_regression_state_input(){
    Eigen::VectorXd state_input = Eigen::VectorXd::Zero(1 +x_dim_ * x_history_len_);
    state_input[0] = 1.0;
    for (int i = 0; i < x_history_len_; i++){
      state_input[1 + x_dim_ * i] = vel_scale_ * states_queue_[i][vel_index_];
      state_input[2 + x_dim_ * i] = states_queue_[i][acc_index_];
      state_input[3 + x_dim_ * i] = states_queue_[i][steer_index_];
    }
    return state_input;
  }
  Eigen::VectorXd LinearRegressionCompensator::get_regression_input(Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm){
    Eigen::VectorXd input = Eigen::VectorXd::Zero(feature_size_);
    input.head(1 + x_dim_ * x_history_len_) = get_regression_state_input();
    input.segment(1 + x_dim_ * x_history_len_, acc_queue_size_) = acc_input_history_concat;
    input.segment(1 + x_dim_ * x_history_len_ + acc_queue_size_, steer_queue_size_) = steer_input_history_concat;
    input.segment(1 + x_dim_ * x_history_len_ + acc_queue_size_ + steer_queue_size_, h_dim_) = h_lstm;
    input.segment(1 + x_dim_ * x_history_len_ + acc_queue_size_ + steer_queue_size_ + h_dim_, h_dim_) = c_lstm;
    if (is_projection_matrix_set_){
      Eigen::VectorXd projected_input = Eigen::VectorXd::Zero(projected_feature_size_);
      projected_input[0] = 1.0;
      projected_input.segment(1, projected_feature_size_ - 1) = projection_matrix_ * input.segment(1, feature_size_ - 1);
      return projected_input;
    }
    else{
      return input;
    }
  }
  void LinearRegressionCompensator::initialize_for_candidates(int num_candidates){
    if (!use_compensator_){
      return;
    }
    load_state_queue(); 
    Eigen::VectorXd input = get_regression_state_input();
    State_Inputs_ = input.replicate(1, num_candidates);
  }
  void LinearRegressionCompensator::update_one_step(Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm, Eigen::VectorXd error_vector){
    if (!use_compensator_){
      return;
    }
    Eigen::VectorXd input = get_regression_input(acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
    XXT_new_ += input * input.transpose();
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
    YXT_new_ += error_vector_extracted * input.transpose();
    num_samples_ += 1;
    current_sigma_e_2_mean_ += (error_vector_extracted - regression_matrix_ * input).squaredNorm();
    current_sigma_p_2_mean_ += input.dot(XXT_inv_ * input);
    if (is_offline_data_set_){
      current_sigma_e_2_offline_mean_ += (error_vector_extracted - regression_matrix_offline_ * input).squaredNorm();
      current_sigma_p_2_offline_mean_ += input.dot(XXT_offline_inv_ * input);
    }
  }
  void LinearRegressionCompensator::update_regression_matrix(){
    if (!use_compensator_){
      return;
    }
    current_sigma_e_2_mean_ /= num_samples_;
    current_sigma_p_2_mean_ /= num_samples_;
    sigma_p_narrow_ = std::sqrt(coef_narrow_ * sigma_p_narrow_ * sigma_p_narrow_ + (1.0 - coef_narrow_) * current_sigma_p_2_mean_);
    sigma_p_wide_ = std::sqrt(coef_wide_ * sigma_p_wide_ * sigma_p_wide_ + (1.0 - coef_wide_) * current_sigma_p_2_mean_);
    sigma_e_narrow_ = std::sqrt(coef_narrow_ * sigma_e_narrow_ * sigma_e_narrow_ + (1.0 - coef_narrow_) * current_sigma_e_2_mean_);
    sigma_e_wide_ = std::sqrt(coef_wide_ * sigma_e_wide_ * sigma_e_wide_ + (1.0 - coef_wide_) * current_sigma_e_2_mean_);
    adaptive_forgetting_factor_ = std::max(min_forgetting_factor_, 1 - (1 - default_forgetting_factor_) * sigma_e_wide_ * sigma_p_narrow_ / (sigma_e_narrow_ * sigma_p_wide_));
    // if sigma_p_narrow is small adaptive_forgetting_factor_ is close to 1.0 (do not forget)
    // if sigma_e_narrow is small adaptive_forgetting_factor_ is small (forget)

    if (is_offline_data_set_){
        current_sigma_e_2_offline_mean_ /= num_samples_;
        current_sigma_p_2_offline_mean_ /= num_samples_;
        sigma_p_offline_ = std::sqrt(coef_narrow_ * sigma_p_offline_ * sigma_p_offline_ + (1.0 - coef_narrow_) * current_sigma_p_2_offline_mean_);
        sigma_e_offline_ = std::sqrt(coef_narrow_ * sigma_e_offline_ * sigma_e_offline_ + (1.0 - coef_narrow_) * current_sigma_e_2_offline_mean_);
        adaptive_weight_offline_ratio_ = sigma_e_narrow_ * sigma_p_offline_ / (sigma_e_offline_ * sigma_p_narrow_ + sigma_e_narrow_ * sigma_p_offline_);
    }
    XXT_online_ = adaptive_forgetting_factor_ * XXT_online_ + (1.0 - adaptive_forgetting_factor_)/ num_samples_ * XXT_new_;
    YXT_online_ = adaptive_forgetting_factor_ * YXT_online_ + (1.0 - adaptive_forgetting_factor_)/ num_samples_ * YXT_new_;
    XXT_ = adaptive_weight_offline_ratio_ * XXT_offline_ + (1.0 - adaptive_weight_offline_ratio_) * XXT_online_;
    YXT_ = adaptive_weight_offline_ratio_ * YXT_offline_ + (1.0 - adaptive_weight_offline_ratio_) * YXT_online_;
    Eigen::MatrixXd regularized_matrix = lambda_ * Eigen::MatrixXd::Identity(XXT_.rows(), XXT_.cols());
    regularized_matrix(0, 0) = lambda_bias_;
    Eigen::MatrixXd XXT_reg = XXT_ + regularized_matrix;
    XXT_inv_ = XXT_reg.llt().solve(Eigen::MatrixXd::Identity(XXT_.rows(), XXT_.cols())); //(XXT_ + regularized_matrix).inverse();
    regression_matrix_ = YXT_ * XXT_inv_;
    XXT_new_ = Eigen::MatrixXd::Zero(projected_feature_size_, projected_feature_size_);
    YXT_new_ = Eigen::MatrixXd::Zero(target_size_, projected_feature_size_);

    num_samples_ = 0;
    current_sigma_e_2_mean_ = 0.0;
    current_sigma_p_2_mean_ = 0.0;
    current_sigma_e_2_offline_mean_ = 0.0;
    current_sigma_p_2_offline_mean_ = 0.0;
  }


  Eigen::VectorXd LinearRegressionCompensator::predict(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm){
    if (!use_compensator_){
      return Eigen::VectorXd::Zero(state_size_);
    }
    update_state_queue(states);
    Eigen::VectorXd input = get_regression_input(acc_input_history_concat, steer_input_history_concat, h_lstm, c_lstm);
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
  Eigen::MatrixXd LinearRegressionCompensator::Predict(Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat, Eigen::MatrixXd H_lstm, Eigen::MatrixXd C_lstm){
    if (!use_compensator_){
      return Eigen::MatrixXd::Zero(state_size_, States.cols());
    }
    Eigen::MatrixXd Input = Eigen::MatrixXd::Zero(feature_size_, States.cols());
    Input.block(0, 0, 1, States.cols()) = Eigen::MatrixXd::Ones(1, States.cols());
    Input.block(1, 0, x_dim_ * (x_history_len_ - 1), States.cols()) = State_Inputs_.block(1 + x_dim_, 0, x_dim_ * (x_history_len_ - 1), States.cols());
    Input.block(1 + x_dim_ * (x_history_len_ - 1), 0, 1, States.cols()) = vel_scale_ * States.block(vel_index_, 0, 1, States.cols());
    Input.block(1 + x_dim_ * (x_history_len_ - 1) + 1, 0, 1, States.cols()) = States.block(acc_index_, 0, 1, States.cols());
    Input.block(1 + x_dim_ * (x_history_len_ - 1) + 2, 0, 1, States.cols()) = States.block(steer_index_, 0, 1, States.cols());
    Input.block(1 + x_dim_ * x_history_len_, 0, acc_queue_size_, States.cols()) = Acc_input_history_concat;
    Input.block(1 + x_dim_ * x_history_len_ + acc_queue_size_, 0, steer_queue_size_, States.cols()) = Steer_input_history_concat;
    Input.block(1 + x_dim_ * x_history_len_ + acc_queue_size_ + steer_queue_size_, 0, h_dim_, States.cols()) = H_lstm;
    Input.block(1 + x_dim_ * x_history_len_ + acc_queue_size_ + steer_queue_size_ + h_dim_, 0, h_dim_, States.cols()) = C_lstm;
    Eigen::MatrixXd Prediction = Eigen::MatrixXd::Zero(state_size_, States.cols());
    Eigen::MatrixXd Prediction_extracted;
    if (is_projection_matrix_set_){
      Eigen::MatrixXd projected_input = Eigen::MatrixXd::Zero(projected_feature_size_, States.cols());
      projected_input.row(0) = Eigen::MatrixXd::Ones(1, States.cols());
      projected_input.block(1, 0, projected_feature_size_ - 1, States.cols()) = projection_matrix_ * Input.block(1, 0, feature_size_ - 1, States.cols());
      Prediction_extracted = regression_matrix_ * projected_input;
    }
    else{
      Prediction_extracted = regression_matrix_ * Input;    
    }
    
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