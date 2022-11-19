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

#include "input_library/input_lib.hpp"

namespace sysid
{

bool sTimeTracker::canReleaseInput(const double &vx)
{
  // Check if the current speed is within the desired range
  if (vx >= activation_vx_msec_ && !is_initialized_ && vx <= max_vx_guard_msec_)
  {
    is_initialized_ = true;
    reInitialize(); // set the time starting point
    return is_initialized_;
  }

  // If not initialized or the speed is dangerous do not send signal
  if (!is_initialized_ || vx > max_vx_guard_msec_)
  {
    return false;
  }

  // otherwise we can use
  return is_initialized_;
}

/**
 * @brief Sum of Sinusoid's definitions ----------------------------------------------------
 * */
InpSumOfSinusoids::InpSumOfSinusoids(const double &activation_vx,
                                     const double &max_vx_guard,
                                     sSumOfSinParameters const &params) :
  time_tracker_{activation_vx, max_vx_guard},
  Ts_{params.start_time},
  frequency_band_hz_{params.frequency_band},
  num_sins_{params.num_of_sins},
  max_amplitude_{params.max_amplitude},
  noise_mean_{params.noise_mean},
  noise_stddev_{params.noise_stddev}
{
  // Frequency definitions
  auto const &freq_starts = frequency_band_hz_[0];
  auto const &freq_ends = frequency_band_hz_[1];
  sin_frequencies_hz_ = ns_utils::linspace(freq_starts, freq_ends, num_sins_);

  // Noise definitions.
  normal_distribution_ = std::normal_distribution<double>(noise_mean_, noise_stddev_);

  std::random_device random_device;
  random_engine_ = std::mt19937(random_device());
}

double InpSumOfSinusoids::generateInput(double const &vx)
{

  // Check if the vehicle in the desired speed range
  if (!time_tracker_.canReleaseInput(vx))
  {
    return 0.;
  }

  // Check a specific time point is passed : additional safety guard
  auto const &t_ms = time_tracker_(); // current time in milliseconds
  if (t_ms < Ts_ * 1000)
  {
    // ns_utils::print("Time after experiment regime reached : ", t_ms);
    return 0;
  }

  // Compute the generator output.
  double input_val_at_t{};

  for (auto const &wt_hz : sin_frequencies_hz_)
  {
    auto &&current_sin_time = (t_ms / 1000 - Ts_);
    auto &&arg = wt_hz * 2. * M_PI * current_sin_time;
    input_val_at_t += std::sin(arg);
  }

  ns_utils::print("In Sum of Sinusoids ... and current time ", t_ms);

  // Generate noise
  auto const &additive_noise = normal_distribution_(random_engine_);
  auto const &clean_signal = max_amplitude_ * input_val_at_t / static_cast<double>(num_sins_);
  auto const &noisy_signal = clean_signal + additive_noise;

  return noisy_signal;
}

/**
 * @brief  ----------------------Filtered Gaussian White Noise -------------------------------------
 * */

InpFilteredWhiteNoise::InpFilteredWhiteNoise(double const &activation_vx,
                                             double const &max_vx_guard,
                                             sFilteredWhiteNoiseParameters const &params)
  : time_tracker_{activation_vx, max_vx_guard},
    Ts_{params.start_time},
    fc_hz_{params.cutoff_frequency_hz},
    fs_hz_{params.sampling_frequency_hz},
    max_amplitude_{params.max_amplitude},
    filter_order_{params.filter_order},
    noise_mean_{params.noise_mean},
    noise_stddev_{params.noise_stddev}
{

  // Design filter here calling the Butterworth and set b_, a_.
  auto b = std::vector<double>{0.002898194633721, 0.008694583901164,
                               0.008694583901164, 0.002898194633721};

  auto a = std::vector<double>{1.0, -2.374094743709352,
                               1.929355669091215, -0.532075368312092};

  low_pass_filter_ = LowPassFilter(b, a);

  // Noise definitions.
  normal_distribution_ = std::normal_distribution<double>(noise_mean_, noise_stddev_);
  std::random_device random_device;
  random_engine_ = std::mt19937(random_device());
}

double InpFilteredWhiteNoise::generateInput(double const &vx)
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

  // Generate noise
  auto filtered_noise = normal_distribution_(random_engine_);
  filtered_noise = low_pass_filter_.filter_pointwise(filtered_noise);
  filtered_noise = std::clamp(filtered_noise, -max_amplitude_, max_amplitude_);

  return filtered_noise;
}

/**
 * @brief  ---------------------- Step Inputs -----------------------------------------------------
 * */

InpStepUpDown::InpStepUpDown(double const &activation_vx,
                             double const &max_vx_guard,
                             sStepParameters const &params)
  : time_tracker_{activation_vx, max_vx_guard},
    Ts_{params.start_time},
    Tp_{params.step_period},
    max_amplitude_{params.max_amplitude},
    step_direction_flag_{params.step_direction_flag}
{

  if (step_direction_flag_ == 1)
  {
    input_set_ = std::vector<double>{1., 0.};
  } else if (step_direction_flag_ == -1)
  {
    input_set_ = std::vector<double>{-1., 0.};
  } else
  {
    input_set_ = std::vector<double>{-1., 0., 1., 0.};
  }

}

double InpStepUpDown::generateInput(double const &vx)
{
  // Check if the vehicle in the desired speed range
  if (!time_tracker_.canReleaseInput(vx))
  {
    return 0.;
  }

  // Check a specific time point is passed : additional safety guard
  // current time in milliseconds
  if (auto const &t_ms = time_tracker_(); t_ms < Ts_ * 1000)
  {
    ns_utils::print("Time after experiment regime reached : ", t_ms);
    return 0;
  }

  // Compute the generator output.
  std::chrono::time_point<std::chrono::system_clock> time_now{std::chrono::system_clock::now()};

  if (auto time_since_last_call = std::chrono::duration_cast<std::chrono::microseconds>(time_now - last_tick_time_);
    static_cast<double>(time_since_last_call.count()) / 1e6 >= Tp_)
  {
    std::rotate(input_set_.begin(), input_set_.begin() + 1, input_set_.end());
    current_input_value_ = input_set_.front() * max_amplitude_;

    // update the last time tick.
    last_tick_time_ = std::chrono::system_clock::now();

    // auto const &&tt = static_cast<double>(time_since_last_call.count()) / 1e6;
    // ns_utils::print("Time since last : ", tt);
  }

  return current_input_value_;
}

} // namespace sysid


