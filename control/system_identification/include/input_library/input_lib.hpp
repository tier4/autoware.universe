// Copyright 2022 The Autoware Foundation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SYSTEM_IDENTIFICATION_INCLUDE_INPUT_LIBRARY_INPUT_LIB_HPP_
#define SYSTEM_IDENTIFICATION_INCLUDE_INPUT_LIBRARY_INPUT_LIB_HPP_

#include <memory>
#include <utility>
#include <chrono>
#include <random>
#include "prbs_tap_bits.hpp"
#include "utils_act/act_utils.hpp"
#include "utils_act/act_utils_eigen.hpp"
#include "signal_processing/lowpass_filter.hpp"

namespace sysid
{
/**
 * @brief common time tracking struct for the input-classes.
 * */
struct sTimeTracker
{
  // Constructors
  sTimeTracker(double const &activation_vx, double const &max_vx_guard) : activation_vx_msec_{activation_vx},
                                                                          max_vx_guard_msec_{max_vx_guard}
  {}

  // Data members
  std::chrono::time_point<std::chrono::system_clock>
    construction_time_point{std::chrono::system_clock::now()};

  double activation_vx_msec_{2}; // activation velocity in m/s
  double max_vx_guard_msec_{5};
  std::chrono::time_point<std::chrono::system_clock> last_tick_time{std::chrono::system_clock::now()};

  bool is_initialized_{false};

  // Methods
  double operator()()
  {
    last_tick_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(last_tick_time - construction_time_point);

    return static_cast<double>(duration.count());
  }

  void reInitialize()
  {
    construction_time_point = std::chrono::system_clock::now();
  }

  [[nodiscard]] bool isInitialized() const
  { return is_initialized_; }

  bool canReleaseInput(double const &vx);

};

/**
 * @brief Input Concept type erasure implementation.
 * */
class InputConcept
{
 public:

  virtual ~InputConcept() = default;
  [[nodiscard]] virtual double generateInput(double const &vx) = 0;
  [[nodiscard]] virtual std::unique_ptr<InputConcept> clone() const = 0;

};

template<typename InputType>
class OwningInputModel : public InputConcept
{
 public:
  explicit OwningInputModel(InputType input_object) : input_object_{std::move(input_object)}
  {}

  [[nodiscard]] double generateInput(double const &vx) override
  {
    // Implement
    return input_object_.generateInput(vx);
  };

  [[nodiscard]] std::unique_ptr<InputConcept> clone() const override
  {
    return std::make_unique<OwningInputModel>(*this);
  }

 private:
  InputType input_object_;
  // sGenerator input_generator_;
};

class InputWrapper
{
 public:
  template<class InputType>
  explicit InputWrapper(InputType input_type):
    pimpl_{std::make_unique<OwningInputModel<InputType>>(std::move(input_type))}
  {

  }

  InputWrapper(InputWrapper const &other) : pimpl_(other.pimpl_->clone())
  {}

  InputWrapper &operator=(InputWrapper const &other)
  {
    // Copy-and-Swap Idiom
    InputWrapper copied(other);
    pimpl_.swap(copied.pimpl_);
    return *this;
  }

  ~InputWrapper() = default;
  InputWrapper(InputWrapper &&) = default;
  InputWrapper &operator=(InputWrapper &&) = default;

  /** Common function */
  [[nodiscard]] double generateInput(double const &vx) const
  {
    return pimpl_->generateInput(vx);
  }

 private:
  std::unique_ptr<InputConcept> pimpl_{};

};

// ****************************************** INPUT IMPLEMENTATIONS ***************************************
/**
 * @brief Input types.
 * */

// ****************************************** Sum of Sinusoidals ******************************************
struct sSumOfSinParameters
{

  double start_time{0.5};  // the class start to generate signal after speed conditions are met.
  std::array<double, 2> frequency_band{2, 8}; // frequency content of sum of sinusoids.
  size_t num_of_sins{3};
  double max_amplitude{0.1};

  // additive noise.
  double noise_mean{};
  double noise_stddev{0.01};
};

class InpSumOfSinusoids
{
 public:
  InpSumOfSinusoids(double const &activation_vx,
                    double const &max_vx_guard,
                    sSumOfSinParameters const &params = sSumOfSinParameters{});

  [[nodiscard]] double generateInput(double const &vx);

 private:
  sTimeTracker time_tracker_;
  double Ts_; // starting time in seconds as long as conditions hold.

  // Sinusoid characterization
  std::array<double, 2> frequency_band_hz_; // w im [wlow, whigh]
  size_t num_sins_;  // number of sinusoids in the test signal
  std::vector<double> sin_frequencies_hz_;  // frequencies of sin signals stored in a vector
  double max_amplitude_;  // maximum amplitude of the sin signals

  // Noise definitions.
  double noise_mean_;  // noise mean and std.
  double noise_stddev_;

  std::normal_distribution<double> normal_distribution_;
  std::mt19937 random_engine_;

};

// ****************************************** Pseudo Random Binary Signal *********************************
/**
 * @brief PRBS design reference:
 * 5.3.2. Pseudo-Random Binary Sequences (PRBS)
 * Landau, I.D. and Zito, G., 2006. Digital control systems: design, identification
 * and implementation (Vol. 130).  London: Springer.
 * */

struct sPRBSparams
{

  sPRBSparams(double start_time_ts,
              double const &estimated_rise_time,
              double const &sampling_time,
              size_t const &prbs_type_N) : start_time{start_time_ts},
                                           trise{estimated_rise_time},
                                           dt_s{sampling_time},
                                           prbs_type{prbs_type_N},
                                           p_prbs_ratio{trise / dt_s / static_cast<double>(prbs_type)},
                                           dt_prbs{dt_s * p_prbs_ratio}
  {


    // Random initial seed.
    std::uniform_int_distribution<uint8_t> temp_dist(1, std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto random_engine = std::mt19937(random_device());
    seed = temp_dist(random_engine);

    // linear feedback shift register tap bits.
    tap_bits_v = tapbits_vectors(prbs_type);
  }

  double start_time{0.5};  // the class start to generate signal after speed conditions are met.
  double trise{0.5};  // estimated rise time for a system of interest.
  double dt_s{0.02};  // sampling time
  size_t prbs_type{8};  // 3 <= N <=32
  double max_amplitude{0.5};

  // Computed in the construction time.
  double p_prbs_ratio; // p = fs / f_prbs : ratio that defines PRBS frequency.
  double dt_prbs; // PRBS frequency

  // initial seed
  uint8_t seed;
  std::vector<size_t> tap_bits_v{};
};

/**
 * @brief PRBS class
 * period of each bit: dt_prbs = p*N*Ts > rise_time of the system, where p is a ratio, N is the number of LFSR bit
 * cells and Ts is the sampling time of the control system.
 *
 * L = p * (2^{N-1}*Ts) minimum experiment length to visit all the frequencies in the PRBS signal.
 *
 * */
template<size_t N = 8>
class InpPRBS
{
 public:
  InpPRBS(double const &activation_vx,
          double const &max_vx_guard, const sPRBSparams &params);

  [[nodiscard]] double generateInput(double const &vx);

 private:
  sTimeTracker time_tracker_;
  double Ts_; // starting time in seconds as long as conditions hold.
  const size_t prbs_type_{N}; // PRBS type
  double max_amplitute_;
  double p_prbs_ratio_; // p = fs / f_prbs : ratio that defines PRBS frequency.

  // Useful data members (about the experiments)
  double dt_sampling_;  // system sampling time [s].
  double dt_prbs_; // prbs sampling time [s]
  double le_;  // minimum time in seconds for experiments to visit all frequencies

  // LFSR members.
  const std::bitset<N> initial_seed_;
  std::bitset<N> current_seed_;
  std::vector<size_t> tap_bits_v_{};

  // when the signal starts to repeat itself
  int64_t counter_{0};
  double current_input_{};
  std::chrono::time_point<std::chrono::system_clock> last_tick_time_{std::chrono::system_clock::now()};

  // linear feedback shift register method
  bool lfsr();

};
template<size_t N>
InpPRBS<N>::InpPRBS(const double &activation_vx, const double &max_vx_guard,
                    const sPRBSparams &params):time_tracker_{activation_vx, max_vx_guard}, Ts_{params.start_time},
                                               max_amplitute_{params.max_amplitude},
                                               p_prbs_ratio_{params.p_prbs_ratio}, dt_sampling_{params.dt_s},
                                               dt_prbs_{params.dt_prbs},
                                               le_{p_prbs_ratio_ * std::pow(2., N - 1) * dt_sampling_},
                                               initial_seed_{params.seed}, current_seed_{params.seed},
                                               tap_bits_v_{params.tap_bits_v}
{

}
template<size_t N>
double InpPRBS<N>::generateInput(const double &vx)
{

  // Check if the vehicle in the desired speed range
  if (!time_tracker_.canReleaseInput(vx))
  {
    return 0.;
  }

  // Check a specific time point is passed : additional safety guard
  // current time in milliseconds
  if (auto const &t_ms = time_tracker_();t_ms < Ts_ * 1000)
  {
    ns_utils::print("Time after experiment regime reached : ", t_ms);
    return 0;
  }

  // Compute the generator output.
  std::chrono::time_point<std::chrono::system_clock> time_now{std::chrono::system_clock::now()};
  auto time_since_last_call = std::chrono::duration_cast<std::chrono::microseconds>(time_now - last_tick_time_);

  if (auto dt_temp = static_cast<double>(time_since_last_call.count()) / 1e6;dt_temp > dt_prbs_)
  {
    auto const b0 = lfsr();
    current_input_ = static_cast<double>(b0) * max_amplitute_ - max_amplitute_ / 2.; // signal - mean(signal)

    // update the time at which the control is updated.
    last_tick_time_ = std::chrono::system_clock::now();

    ns_utils::print("prbs time passed ", dt_prbs_, "--> == ? ", dt_temp);
  }

  //  if (current_seed_ != initial_seed_)
  //  {
  //
  //    counter_ += 1;
  //
  //  } else
  //  {
  //    ns_utils::print("Current counter for repetition ; ", counter_ + 1);
  //  }

  return current_input_;
}

template<size_t N>
bool InpPRBS<N>::lfsr()
{
  auto n0 = tap_bits_v_[0] - 1; // tap bits' first index
  bool b0 = current_seed_[n0]; // get the bit of the current seed

  // xor by the tap bits
  for (size_t k = 1; k < tap_bits_v_.size(); ++k)
  {
    auto idx = tap_bits_v_[k] - 1;
    b0 ^= current_seed_[idx];
  }

  b0 = b0 & 1;

  // shift the current_seed
  current_seed_ = current_seed_ << 1;  // shift left
  current_seed_[0] = b0;

  return b0;
}

// ****************************************** Step Inputs : Up - Down *********************************
struct sStepParameters
{
  double start_time{0.5};  // the class start to generate signal after speed conditions are met.
  double step_period{0.1};
  double max_amplitude{0.1};
  int8_t step_direction_flag{1}; // 1: Up, -1: Down, 0:Up and Down
};

class InpStepUpDown
{
 public:
  InpStepUpDown(double const &activation_vx,
                double const &max_vx_guard,
                sStepParameters const &params = sStepParameters{});

  [[nodiscard]] double generateInput(double const &vx);

 private:
  sTimeTracker time_tracker_;
  double Ts_{0.5}; // starting time in seconds as long as conditions hold.
  double Tp_{1.}; // step period, step becomes zero after this time in second.
  double max_amplitude_{0.1};

  int8_t step_direction_flag_{1}; // 1: Up, -1: Down, 0:Up and Down
  std::vector<double> input_set_{1, 0}; // number of predefined values {1, 0} or {-1, 0} or {-1, 0, 0}
  double current_input_value_{0.};

  std::chrono::time_point<std::chrono::system_clock> last_tick_time_{std::chrono::system_clock::now()};
};

// ****************************************** Filtered White Noise *********************************

struct sFilteredWhiteNoiseParameters
{
  double start_time{0.5};  // the class start to generate signal after speed conditions are met.
  double cutoff_frequency_hz{5.};
  double sampling_frequency_hz{100.};
  double max_amplitude{0.3};

  // Filter definitions.
  int filter_order{3};

  double noise_mean{0.0};
  double noise_stddev{0.2};
};

class InpFilteredWhiteNoise
{
 public:

  InpFilteredWhiteNoise(double const &activation_vx,
                        double const &max_vx_guard,
                        sFilteredWhiteNoiseParameters const &params = sFilteredWhiteNoiseParameters{});

  [[nodiscard]] double generateInput(double const &vx);

 private:
  sTimeTracker time_tracker_;
  double Ts_{0.5}; // starting time in seconds as long as conditions hold.

  double fc_hz_{};  // cut-off frequency in Hertz
  double fs_hz_{}; // sampling frequency
  double max_amplitude_{}; // maximum amplitude of the signal.

  // low-pass filter definitions.
  int filter_order_{1};

  LowPassFilter low_pass_filter_;

  // Noise definitions.
  double noise_mean_;  // noise mean and std.
  double noise_stddev_;

  std::normal_distribution<double> normal_distribution_;
  std::mt19937 random_engine_;

};

}

#endif  // SYSTEM_IDENTIFICATION_INCLUDE_INPUT_LIBRARY_INPUT_LIB_HPP_
