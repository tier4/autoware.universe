//
// Created by ali.boyali@tier4.jp on 8/11/22.
//
#include "utils/writetopath.hpp"
#include <iostream>
#include "utils/act_utils.hpp"
#include "utils/act_utils_eigen.hpp"
#include "sysid_inputs.hpp"
#include "linear_filter.hpp"

#include <numeric>
#include <functional>

using namespace std::chrono_literals;

int main()
{
  auto save_path = getOutputPath();
  double fs_hz{100.};  // sampling frequency
  double fc_hz{5.};  // cut-off frequency
  size_t nf{3}; // filter order

  // Computed filter for these values.
  //   std::vector<double> b{0.002898194633721, 0.008694583901164,
  //                         0.008694583901164, 0.002898194633721};
  //   std::vector<double> a{1.0, -2.374094743709352,
  //                         1.929355669091215, -0.532075368312092};

  std::vector<double> b{5.9796e-05, 0.00029898, 0.00059796,
                        0.00059796, 0.00029898, 5.9796e-05};

  std::vector<double> a{1, -3.9845, 6.4349,
                        -5.2536, 2.1651, -0.35993};

  // inner product
  double r1 = std::inner_product(a.begin(), a.end(), b.begin(), 0.);
  std::cout << "Inner product of a and b: " << r1 << '\n';

  // rotate
  std::vector<int> v{2, 4, 1, 0, 5, 10, 7, 3, 7, 1};

  // simple rotation to the left
  std::rotate(v.begin(), v.begin() + 1, v.end());
  ns_utils::print_container(v);

  ns_utils::print("hello ");

  // Off-diagonal matrix
  //  Eigen::MatrixXd A(4, 4);
  //  A.setZero();
  //  A.diagonal(-1).setOnes();
  //  ns_eigen_utils::printEigenMat(A);

  // Create a filter:
  auto filter = LowPassFilter(b, a);

  // Generate a sinus signal:
  double tfinal = 5.;

  std::vector<double> time_vec = ns_utils::linspace(0., tfinal, 100);
  ns_utils::print_container(time_vec);

  std::vector<double> sin_signal;
  sin_signal.reserve(time_vec.size());

  double w1 = 1. * M_PI * 2.;  // 1 hz signal
  double w2 = 50. * M_PI * 2.;  // 20 hz signal to be filtered out.


  /**
   * Create the sum of sinusoidal signal.
   * */
  for (auto const &t : time_vec)
  {
    auto const &sumof_sins = std::sin(w1 * t + 0.2) + std::sin(w2 * t + 0.1);
    sin_signal.emplace_back(sumof_sins);
  }

  /**
   * Filter a frequency from the sum of sinusoids.
   * filter initial condition for scipy array([ 0.99710181, -1.38568752,  0.53497356])
   * */

  /**
   * @brief filter.filter_vector(signal_to_be_filtered, compute_initial_condition_bool);
   * */

  auto signal_filtered = filter.filter_vector(sin_signal, true);
  auto signal_filtfilted = filter.filtfilt_vector(sin_signal);

  writeToFile(save_path, time_vec, "filt_time");
  writeToFile(save_path, sin_signal, "filt_sin_signal");
  writeToFile(save_path, signal_filtered, "filt_sin_signal_filtered");
  writeToFile(save_path, signal_filtfilted, "signal_filtfilted");

  return EXIT_SUCCESS;
}
