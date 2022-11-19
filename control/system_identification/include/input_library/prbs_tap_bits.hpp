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


#ifndef SYSTEM_IDENTIFICATION_INCLUDE_UTILS_PRBS_TAP_BITS_HPP_
#define SYSTEM_IDENTIFICATION_INCLUDE_UTILS_PRBS_TAP_BITS_HPP_

#include <vector>
#include <array>
#include <bitset>

/**
 * @brief Pseudo-Random Binary Generator by Linear Feedback Shift-Registers.
 * Landau, I.D. and Zito, G., 2006. Digital control systems: design, identification and implementation (Vol. 130).
 * London: Springer.
 * */

std::vector<size_t> tapbits_vectors(size_t const &prbs_type = 8);

template<size_t N>
using bitset_t = std::bitset<N>;

#endif //SYSTEM_IDENTIFICATION_INCLUDE_UTILS_PRBS_TAP_BITS_HPP_
