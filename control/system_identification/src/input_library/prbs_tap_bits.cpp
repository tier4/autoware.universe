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

#include "input_library/prbs_tap_bits.hpp"

std::vector<size_t> tapbits_vectors(size_t const &prbs_type)
{
  auto const &n = prbs_type;
  switch (n)
  {
    case 3:return {1, 3}; // PRBS_3
    case 4:return {1, 4};
    case 5:return {2, 5};
    case 6:return {1, 6};
    case 7:return {1, 7}; // PRBS_7
    case 8:return {1, 2, 7, 8}; // PRBS_8
    case 9:return {4, 9};
    case 10:return {3, 10};
    case 11:return {9, 11};
    case 12:return {6, 8, 11, 12};
    case 13:return {9, 10, 12, 13};
    case 14:return {4, 8, 13, 14};
    case 15:return {14, 15};
    case 16:return {4, 13, 15, 16};
    case 17:return {14, 17};
    case 18:return {11, 18};
    case 19:return {1, 2, 6, 19};
    case 20:return {17, 20};
    case 21:return {19, 21};
    case 22:return {21, 22};
    case 23:return {18, 23};
    case 24:return {17, 22, 23, 24};
    case 25:return {22, 25};
    case 26:return {1, 2, 6, 26};
    case 27:return {1, 2, 5, 27};
    case 28:return {25, 28};
    case 29:return {27, 29};
    case 30:return {1, 4, 6, 30};
    case 31:return {28, 31};
    case 32:return {1, 2, 22, 32};

    default:return {1, 3};

  }

}

