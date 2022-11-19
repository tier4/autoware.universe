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

#include "signal_processing/lowpass_filter.hpp"

geometry_msgs::msg::Twist LowpassFilterTwist::filter(const geometry_msgs::msg::Twist &u)
{
  if (x_)
  {
    x_->linear.x = gain_ * x_->linear.x + (1.0 - gain_) * u.linear.x;
    x_->linear.y = gain_ * x_->linear.y + (1.0 - gain_) * u.linear.y;
    x_->linear.z = gain_ * x_->linear.z + (1.0 - gain_) * u.linear.z;

    x_->angular.x = gain_ * x_->angular.x + (1.0 - gain_) * u.angular.x;
    x_->angular.y = gain_ * x_->angular.y + (1.0 - gain_) * u.angular.y;
    x_->angular.z = gain_ * x_->angular.z + (1.0 - gain_) * u.angular.z;

    return x_.get();
  }

  x_ = u;
  return x_.get();
}

/**
 * Filter Implementation Using Transposed Direct Form II.
 * */
LowPassFilter::LowPassFilter(std::vector<double> b, std::vector<double> a)
  : b_{std::move(b)}, a_{std::move(a)}
{
  filter_order_ = a_.size();

  // Prepare the step response of the filter and buffers.
  y_buffer_ = std::vector(a_.size() - 1, 0.);

}

double LowPassFilter::filter_pointwise(const double &x)
{

  /**
   *  Transposed Direct Form II implementation.
   * */

  auto const &ynew = y_buffer_[0] + b_[0] * x;

  // Update the state vector y_buffer_;
  for (size_t k = 0; k < y_buffer_.size() - 1; ++k)
  {
    y_buffer_[k] = y_buffer_[k + 1] - a_[k + 1] * ynew + b_[k + 1] * x;
  }

  y_buffer_.back() = b_.back() * x - a_.back() * ynew;

  return ynew;
}

std::vector<double> LowPassFilter::filter_vector(const std::vector<double> &xunfiltered,
                                                 bool const &compute_initial_values)
{
  std::vector<double> filtered_vec;
  filtered_vec.reserve(xunfiltered.size());

  if (compute_initial_values)
  {
    y_buffer_ = filterStepResponseInitialization();
    auto xunfilt0 = xunfiltered[0];

    for (auto &y : y_buffer_)
    {
      y *= xunfilt0;
    }
  }

  for (auto const &uk : xunfiltered)
  {
    auto const &temp_xfilt = filter_pointwise(uk);
    filtered_vec.emplace_back(temp_xfilt);
  }

  return filtered_vec;
}

double LowPassFilter::filtfilt_pointwise(const double &)
{
  // to be implemented.
  return 0;
}
std::vector<double> LowPassFilter::filtfilt_vector(const std::vector<double> &xunfiltered)
{
  /**
   * Filter forward with the filter initial conditions.
   * */
  auto const &la = a_.size();
  auto const &lb = b_.size();

  auto const &npad = std::max(la, lb) * 3;

  /**
   * Create the padded signal.
   * */
  std::vector<double> left_pad;
  std::vector<double> right_pad;
  left_pad.reserve(npad);
  right_pad.reserve(npad);

  auto const &xstart = xunfiltered.front();
  auto const &xend = xunfiltered.back();

  for (size_t k = npad; k > 0; --k)
  {
    left_pad.emplace_back(2. * xstart - xunfiltered[k]);
  }

  for (size_t k = 1; k < npad + 1; ++k)
  {
    right_pad.emplace_back(2. * xend - xunfiltered.rbegin()[static_cast<long>(k)]);
  }

  // Join all vectors.
  std::vector<double> xunfiltered_padded;
  xunfiltered_padded.reserve(left_pad.size() + xunfiltered.size() + right_pad.size());

  // put first the left_pad.
  xunfiltered_padded.insert(xunfiltered_padded.end(), std::make_move_iterator(left_pad.begin()),
                            std::make_move_iterator(left_pad.end()));

  // copy the input
  xunfiltered_padded.insert(xunfiltered_padded.end(), xunfiltered.begin(), xunfiltered.end());


  // put the right_pad.
  xunfiltered_padded.insert(xunfiltered_padded.end(), std::make_move_iterator(right_pad.begin()),
                            std::make_move_iterator(right_pad.end()));

  auto &&forward_filtered = filter_vector(xunfiltered_padded, true);
  std::reverse(forward_filtered.begin(), forward_filtered.end());

  auto &&backward_filtered = filter_vector(forward_filtered, true);
  std::reverse(backward_filtered.begin(), backward_filtered.end());

  std::vector<double> filtfilted_x;
  filtfilted_x.reserve(xunfiltered.size());

  filtfilted_x.insert(filtfilted_x.begin(), std::make_move_iterator(backward_filtered.begin() + npad),
                      std::make_move_iterator(backward_filtered.end() - npad));

  return filtfilted_x;
}

Eigen::MatrixXd LowPassFilter::getCompanionForm(const std::vector<double> &a)
{
  auto const &nx = a.size();
  Eigen::MatrixXd A(nx, nx);

  A.setZero();
  A.diagonal(-1).setOnes();

  for (size_t k = 0; k < nx; ++k)
  {
    A.row(0)(static_cast<Eigen::Index>(k)) = a[k];
  }

  return A;
}

std::vector<double> LowPassFilter::filterStepResponseInitialization()
{
  // Convert "a" to a companion matrix form.
  std::vector<double> atemp;
  atemp.reserve(filter_order_ - 1);

  std::transform(a_.cbegin() + 1, a_.cend(), std::back_inserter(atemp),
                 [](auto const &x)
                 { return -x; });

  auto A = getCompanionForm(atemp);
  Eigen::VectorXd B(filter_order_ - 1, 1);
  B.setZero();

  for (size_t k = 1; k < filter_order_; ++k)
  {
    B(static_cast<Eigen::Index>(k - 1), 0) = b_[k] - a_[k] * b_[0];
  }

  auto const &IminusAT = Eigen::MatrixXd::Identity(static_cast<long>(filter_order_ - 1),
                                                   static_cast<long>(filter_order_ - 1)) - A.transpose();

  std::vector<double> z_initial_state(filter_order_ - 1, 0.);
  z_initial_state.at(0) = B.sum() / IminusAT.col(0).sum();

  double asum{1.};
  double csum{0.};

  for (size_t k = 1; k < filter_order_ - 1; ++k)
  {
    asum += a_[k];
    csum += b_[k] - a_[k] * b_[0];

    const auto &temp = asum * z_initial_state[0] - csum;
    z_initial_state.at(k) = temp;
  }

  return z_initial_state;
}
