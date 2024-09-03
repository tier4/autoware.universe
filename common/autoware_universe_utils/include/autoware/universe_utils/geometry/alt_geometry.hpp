// Copyright 2020-2024 Tier IV, Inc.
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

#ifndef AUTOWARE__UNIVERSE_UTILS__GEOMETRY__ALT_GEOMETRY_HPP_
#define AUTOWARE__UNIVERSE_UTILS__GEOMETRY__ALT_GEOMETRY_HPP_

#include "autoware/universe_utils/geometry/boost_geometry.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace autoware::universe_utils
{
// Alternatives for Boost.Geometry ----------------------------------------------------------------
// TODO(soblin): use inlinen namespace for API versioning
namespace alt
{
class Vector2d
{
public:
  Vector2d() : x_(0.0), y_(0.0) {}

  Vector2d(const double x, const double y) : x_(x), y_(y) {}

  double cross(const Vector2d & other) const { return x_ * other.y() - y_ * other.x(); }

  double dot(const Vector2d & other) const { return x_ * other.x() + y_ * other.y(); }

  double norm2() const { return x_ * x_ + y_ * y_; }

  double norm() const { return std::sqrt(norm2()); }

  Vector2d vector_triple(const Vector2d & v1, const Vector2d & v2) const
  {
    const auto tmp = this->cross(v1);
    return {-v2.y() * tmp, v2.x() * tmp};
  }

  const double & x() const { return x_; }

  double & x() { return x_; }

  const double & y() const { return y_; }

  double & y() { return y_; }

private:
  double x_;
  double y_;
};

// We use Vector2d to represent points, but we do not name the class Point2d directly
// as it has some vector operation functions.
using Point2d = Vector2d;
using Points2d = std::vector<Point2d>;

class ConvexPolygon2d
{
public:
  // TODO(soblin): this factory should check if the inputs points really form a valid "Convex"
  // polygon, because current implementation allows the user to create actually NonConvexPolygon and
  // calculate wrong result without notice
  static std::optional<ConvexPolygon2d> createConvexPolygon2d(const Points2d & vertices) noexcept;

  static std::optional<ConvexPolygon2d> createConvexPolygon2d(Points2d && vertices) noexcept;

  static void correct(Points2d & vertices) noexcept;

private:
  explicit ConvexPolygon2d(const Points2d & vertices) { vertices_ = vertices; }

  explicit ConvexPolygon2d(Points2d && vertices) { vertices_ = std::move(vertices); }

public:
  const Points2d & vertices() const noexcept { return vertices_; }

  Points2d & vertices() noexcept { return vertices_; }

private:
  Points2d vertices_;
};

inline Vector2d operator+(const Vector2d & v1, const Vector2d & v2)
{
  return {v1.x() + v2.x(), v1.y() + v2.y()};
}

inline Vector2d operator-(const Vector2d & v1, const Vector2d & v2)
{
  return {v1.x() - v2.x(), v1.y() - v2.y()};
}

inline Vector2d operator-(const Vector2d & v)
{
  return {-v.x(), -v.y()};
}

inline Vector2d operator*(const double & s, const Vector2d & v)
{
  return {s * v.x(), s * v.y()};
}

Point2d from_boost(const autoware::universe_utils::Point2d & point) noexcept;

std::optional<ConvexPolygon2d> from_boost(
  const autoware::universe_utils::Polygon2d & polygon) noexcept;

autoware::universe_utils::Point2d to_boost(const Point2d & point) noexcept;

autoware::universe_utils::Polygon2d to_boost(const ConvexPolygon2d & polygon) noexcept;
}  // namespace alt

double area(const alt::ConvexPolygon2d & poly) noexcept;

std::optional<alt::ConvexPolygon2d> convex_hull(const alt::Points2d & points) noexcept;

bool covered_by(const alt::Point2d & point, const alt::ConvexPolygon2d & poly);

bool disjoint(const alt::ConvexPolygon2d & poly1, const alt::ConvexPolygon2d & poly2);

double distance(
  const alt::Point2d & point, const alt::Point2d & seg_start, const alt::Point2d & seg_end);

double distance(const alt::Point2d & point, const alt::ConvexPolygon2d & poly);

bool equals(const alt::Point2d & point1, const alt::Point2d & point2);

bool equals(const alt::ConvexPolygon2d & poly1, const alt::ConvexPolygon2d & poly2);

bool intersects(
  const alt::Point2d & seg1_start, const alt::Point2d & seg1_end, const alt::Point2d & seg2_start,
  const alt::Point2d & seg2_end);

bool intersects(const alt::ConvexPolygon2d & poly1, const alt::ConvexPolygon2d & poly2);

bool is_above(
  const alt::Point2d & point, const alt::Point2d & seg_start, const alt::Point2d & seg_end);

bool is_clockwise(const alt::ConvexPolygon2d & poly);

bool touches(
  const alt::Point2d & point, const alt::Point2d & seg_start, const alt::Point2d & seg_end);

bool touches(const alt::Point2d & point, const alt::ConvexPolygon2d & poly);

bool within(const alt::Point2d & point, const alt::ConvexPolygon2d & poly);

bool within(
  const alt::ConvexPolygon2d & poly_contained, const alt::ConvexPolygon2d & poly_containing);
}  // namespace autoware::universe_utils

#endif  // AUTOWARE__UNIVERSE_UTILS__GEOMETRY__ALT_GEOMETRY_HPP_
