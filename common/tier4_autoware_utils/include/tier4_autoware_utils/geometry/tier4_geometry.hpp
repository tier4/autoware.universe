// Copyright 2022 TIER IV, Inc.
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

#ifndef TIER4_AUTOWARE_UTILS__GEOMETRY__TIER4_GEOMETRY_HPP_
#define TIER4_AUTOWARE_UTILS__GEOMETRY__TIER4_GEOMETRY_HPP_

#include "tier4_autoware_utils/geometry/point.hpp"

#include <geometry_msgs/msg/pose.hpp>

#include <vector>

namespace tier4_autoware_utils
{
namespace tier4_geometry
{
using tier4_autoware_utils::Point2d;

struct Matrix2d : public Eigen::Matrix2d
{
  Matrix2d(const Point2d & P1, const Point2d & P2) : Eigen::Matrix2d()
  {
    // Initialize the matrix based on the Point2d objects
    // For example, set the matrix elements to the x and y values of the points
    // Replace the following lines with your specific logic
    (*this)(0, 0) = P1.x();
    (*this)(0, 1) = P1.y();
    (*this)(1, 0) = P2.x();
    (*this)(1, 1) = P2.y();
  }
};

struct Line2d
{
public:
  Point2d p1, p2;
  Line2d(Point2d & p1_, Point2d & p2_) : p1(p1_), p2(p2_) {}
  Line2d() {}
  Line2d(const Line2d & other) = default;
  Line2d & operator=(const Line2d & other) = default;
};

class Polygon2d
{
public:
  std::vector<Point2d> apex;
  std::vector<Line2d> line_segments;
  double area;
  int id;

  Polygon2d();
  Polygon2d(std::vector<Point2d> const & apex_, int id_) : apex(apex_), id(id_)
  {
    assert(apex.size() >= 3);
    computeArea();
    computeLineSegments();
  };
  Polygon2d(const Polygon2d & other) = default;
  Polygon2d & operator=(const Polygon2d & other) = default;

  double getArea();

  int getNvertices();
  int getNSegments();

  /**
   * Fill an empty convex hull with its apexes.
   * @param apex_: Vector of points (C. Hull vertices ordered CCW)
   **/
  void setApexes(std::vector<Point2d> const & apex_);

  /**
   * Uses the Ray casting algorithm:
   *https://en.wikipedia.org/wiki/Point_in_polygon to determine if a Point is
   *inside this Polygon
   **/
  bool isPointInside(const Point2d & P);

private:
  /**
      * Area of convex polygon computed following this approach
  https://byjus.com/maths/convex-polygon/ We compute and add the area of the
  inner triangles of the c. hull to get its total area. \return convex hull area
  (double)
  **/
  void computeArea()
  {  // The inner triangles of the Polygon are
     // added to get the areaof the polygon
    // A formula for this is:
    // area = 0.5 * det{([x1,x2],[y1,y2]) + ([x2,x3],[y2,y3]) + ... +
    // ([xn,x1],[yn,y1])}
    area = 0;
    Matrix2d apexMatrix(apex[apex.size() - 1], apex[0]);
    // The loop below ignores the Matrix ([xn,x1],[yn,y1]), so we add it manually
    // to the sum
    area = apexMatrix.determinant();
    for (int i = 0; i < apex.size() - 1; ++i) {
      Matrix2d temp(apex[i], apex[i + 1]);
      area += temp.determinant();
    }

    area = 0.5 * area;
    if (area < 0.0) area *= -1.;
  };
  /**
   * Fills an internal class member: line_segments with the lines or "edges"
   * connecting each polygon vertex
   */
  void computeLineSegments();
};

bool pointInPolygon(std::vector<Point2d> const & vertices, const Point2d P);

/**
 * Check if two Line segments intersect
 *@param vertices: Vertices of the Polygon
 *@param L1: First Line Segment to test.
 *@param L2: Second Line Segment to test.
 *@param intersect_point: The intersection point data (if it exists) will be
 *copied here.
 *@param epsilon: The tolerance used when comparing floating point substraction
 *results to 0
 *@return true or false
 */
bool segmentsIntersect(Line2d * L1, Line2d * L2, Point2d * intersect_point, const double & epsilon);

/**
 * Sorts a vector of Points CCW by setting one Point as "center", and checking
 * the angle difference between said point and the rest. The points are then
 * arranged based on the pair-wise angle different between each point and the
 * "Center".
 * @param point_vector: Vector of points to sort CCW
 */
void sortPointsCCW(std::vector<Point2d> * point_vector);

/**
 * If two polygons intersect, returns a non-ordered list of vertices
 * corresponding to the polygon formed by the intersection of the two original
 * polygons
 * @param P1: Polygon to check for intersection.
 * @param P2: Polygon to check for intersection.
 * @returns
 */
std::vector<Point2d> getIntersectionPolygonVertices(Polygon2d * P1, Polygon2d * P2);

/**
 * If two polygons intersect, stores the corresponding polygon created by the
 * intersection
 * @param P1: Polygon to check for intersection.
 * @param P2: Polygon to check for intersection.
 * @param Intersection: Intersecting polygon (if it exists) data is stored here
 * @returns true or false, f the polygons intersect or not
 */
bool getIntersectingPolygon(Polygon2d * P1, Polygon2d * P2, Polygon2d * Intersection);

}  // namespace tier4_geometry
}  // namespace tier4_autoware_utils

#endif  // TIER4_AUTOWARE_UTILS__GEOMETRY__TIER4_GEOMETRY_HPP_
