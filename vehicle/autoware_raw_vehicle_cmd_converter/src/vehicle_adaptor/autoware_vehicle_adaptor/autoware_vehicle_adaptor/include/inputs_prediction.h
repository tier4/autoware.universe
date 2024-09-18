
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

Eigen::VectorXd tanh(const Eigen::VectorXd & v);
Eigen::VectorXd sigmoid(const Eigen::VectorXd & v);
Eigen::VectorXd relu(const Eigen::VectorXd & x);
Eigen::MatrixXd read_csv(std::string file_path);
Eigen::VectorXd interpolate_eigen(Eigen::VectorXd y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new);
std::vector<Eigen::VectorXd> interpolate_vector(std::vector<Eigen::VectorXd> y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new);
class InputsSchedulePrediction
{
private:
  Eigen::MatrixXd weight_pre_encoder_0_;
  Eigen::MatrixXd weight_pre_encoder_1_;
  Eigen::MatrixXd weight_lstm_encoder_ih_;
  Eigen::MatrixXd weight_lstm_encoder_hh_;
  Eigen::MatrixXd weight_lstm_decoder_ih_;
  Eigen::MatrixXd weight_lstm_decoder_hh_;
  Eigen::MatrixXd weight_post_decoder_0_;
  Eigen::MatrixXd weight_post_decoder_1_;
  Eigen::MatrixXd weight_final_layer_;
  Eigen::VectorXd bias_pre_encoder_0_;
  Eigen::VectorXd bias_pre_encoder_1_;
  Eigen::VectorXd bias_lstm_encoder_ih_;
  Eigen::VectorXd bias_lstm_encoder_hh_;
  Eigen::VectorXd bias_lstm_decoder_ih_;
  Eigen::VectorXd bias_lstm_decoder_hh_;
  Eigen::VectorXd bias_post_decoder_0_;
  Eigen::VectorXd bias_post_decoder_1_;
  Eigen::VectorXd bias_final_layer_;
  Eigen::VectorXd adaptive_scale_;
  double vel_scaling_, vel_bias_;

  int vel_index_ = 0;
  int acc_index_ = 1;
  int steer_index_ = 2;
  int acc_input_index_ = 3;
  int steer_input_index_ = 4;
  int predict_step_ = 3;
  int input_step_,output_step_;
  int h_dim_;
  double control_dt_ = 0.033;
  bool initialized_ = false;
  std::vector<Eigen::VectorXd> x_seq_obs_;
  int jerk_lim_, steer_rate_lim_;
  std::vector<double> timestamps_obs_;
  Eigen::VectorXd limit_scaling_;
public:
  InputsSchedulePrediction();
  virtual ~InputsSchedulePrediction();
  void set_params(
    int input_step, int output_step, double control_dt, std::string csv_dir);
  void set_NN_params(
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
    double vel_scaling, double vel_bias, double control_dt);
  void update_encoder_cells(
    const Eigen::VectorXd & x, Eigen::VectorXd & h_lstm_encoder, Eigen::VectorXd & c_lstm_encoder
  );
  void get_encoder_cells(
    const std::vector<Eigen::VectorXd> & x_seq, Eigen::VectorXd & h_lstm_encoder, Eigen::VectorXd & c_lstm_encoder
  );
  Eigen::VectorXd update_decoder_cells(
    const Eigen::VectorXd & decoder_input, Eigen::VectorXd & h_lstm_decoder, Eigen::VectorXd & c_lstm_decoder
  );
  std::vector<Eigen::VectorXd> predict(const std::vector<Eigen::VectorXd> & x_seq);
  std::vector<Eigen::VectorXd> get_inputs_schedule_predicted(
    const Eigen::VectorXd & x, double timestamp
  );
  void send_initialized_flag();
};