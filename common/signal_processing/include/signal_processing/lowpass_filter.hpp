// Copyright 2021 Tier IV, Inc.
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

#ifndef SIGNAL_PROCESSING__LOWPASS_FILTER_HPP_
#define SIGNAL_PROCESSING__LOWPASS_FILTER_HPP_

#include "signal_processing/lowpass_filter_1d.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "eigen3/Eigen/Core"
/**
 * @class First-order low-pass filter
 * @brief filtering values
 */
template<typename T>
class LowpassFilterInterface
{
 protected:
  boost::optional<T> x_;  //!< @brief current filtered value
  double gain_;           //!< @brief gain value of first-order low-pass filter

 public:
  explicit LowpassFilterInterface(const double gain) : gain_(gain)
  {}

  void reset()
  { x_ = {}; }
  void reset(const T &x)
  { x_ = x; }

  boost::optional<T> getValue() const
  { return x_; }

  virtual T filter(const T &u) = 0;
};

class LowpassFilterTwist : public LowpassFilterInterface<geometry_msgs::msg::Twist>
{
 public:
  explicit LowpassFilterTwist(const double gain)
    : LowpassFilterInterface<geometry_msgs::msg::Twist>(gain)
  {
  }

  geometry_msgs::msg::Twist filter(const geometry_msgs::msg::Twist &u) override;
};

/**
 * @brief Linear forward and forward-backward filter implementation based on the following references:
 * 1 - Manolakis, D.G. and Ingle, V.K., 2011. Applied digital signal processing: theory and practice. Section 9.2
 * Transposed Direct form II.  Cambridge university press.
 * */


class LowPassFilter
{
 public:

  LowPassFilter() = default;
  LowPassFilter(std::vector<double> b, std::vector<double> a);

  // filter interface.
  double filter_pointwise(const double &x);
  double filtfilt_pointwise(const double &x);

  std::vector<double> filter_vector(const std::vector<double> &xunfiltered,
                                    bool const &compute_initial_values = false);
  std::vector<double> filtfilt_vector(const std::vector<double> &xunfiltered);

 private:

  std::vector<double> b_{};
  std::vector<double> a_{};
  size_t filter_order_{};

  // filter queues x is the unfiltered input, y is filtered output
  std::vector<double> y_buffer_{};

  // Filter helper methods.
  std::vector<double> filterStepResponseInitialization();
  static Eigen::MatrixXd getCompanionForm(std::vector<double> const &a);

};

#endif  // SIGNAL_PROCESSING__LOWPASS_FILTER_HPP_
