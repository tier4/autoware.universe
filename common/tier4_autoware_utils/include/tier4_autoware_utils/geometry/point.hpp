// Copyright 2020 Tier IV, Inc.
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

#ifndef TIER4_AUTOWARE_UTILS__GEOMETRY__POINT_HPP_
#define TIER4_AUTOWARE_UTILS__GEOMETRY__POINT_HPP_

#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/geometries/register/point.hpp>

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>

#include <geometry_msgs/msg/point.hpp>

namespace tier4_autoware_utils
{
struct Point2dS;
struct Point3dS;

struct Point2dS : public Eigen::Vector2d
{
  Point2dS() = default;
  explicit Point2dS(geometry_msgs::msg::Point p) : Eigen::Vector2d(p.x, p.y) { computeAngle(); }
  Point2dS(const double x, const double y) : Eigen::Vector2d(x, y) { computeAngle(); }

  [[nodiscard]] Point3dS to_3d(const double z = 0.0) const;
  double angle;

  // get angle between this point and another.
  // Will be used to organize points CCW
  double get_angle(Point2dS & P)
  {
    // check to make sure the angle won't be "0"
    if (P.x() == this->x()) {
      return 0;
    }

    return (std::atan2((P.y() - this->y()), (P.x() - this->x())));
  }
  void set_angle(double d) { angle = d; }
  void computeAngle() { angle = std::atan2(this->y(), this->x()); }

  // for sorting based on angles
  bool operator<(const Point2dS & p) const { return (angle < p.angle); }

  Point2dS operator*(const double & scalar) const
  {
    Point2dS res;
    res.x() = this->x() * scalar;
    res.y() = this->y() * scalar;
    return res;
  }

  Point2dS operator+(const Point2dS & P) const
  {
    Point2dS res;
    res.x() = this->x() + P.x();
    res.y() = this->y() + P.y();
    res.computeAngle();
    return res;
  }

  Point2dS operator-(const Point2dS & P) const
  {
    Point2dS res;
    res.x() = this->x() - P.x();
    res.y() = this->y() - P.y();
    res.computeAngle();
    return res;
  }
};

struct Point3dS : public Eigen::Vector3d
{
  Point3dS() = default;
  Point3dS(const double x, const double y, const double z) : Eigen::Vector3d(x, y, z) {}

  [[nodiscard]] Point2dS to_2d() const;
};

inline Point3dS Point2dS::to_3d(const double z) const
{
  return Point3dS{x(), y(), z};
}

inline Point2dS Point3dS::to_2d() const
{
  return Point2dS{x(), y()};
}

}  // namespace tier4_autoware_utils

#endif  // TIER4_AUTOWARE_UTILS__GEOMETRY__POINT_HPP_
