// Copyright 2024 Tier IV, Inc.
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
// Original implementation: https://github.com/danielsanchezaran/CPP_Convex_Hulls

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

bool getIntersectingPolygon(const Polygon2d & p1, const Polygon2d & p2, Polygon2d & intersection);
std::vector<Point2d> getIntersectionPolygonVertices(const Polygon2d & p1, const Polygon2d & p2);
bool segmentsIntersect(
  const Line2d & l1, const Line2d & l2, Point2d & intersect_point, const double & epsilon = 1E-5);
bool isPointInPolygon(const std::vector<Point2d> & vertices, const Point2d & p);
struct Matrix2d : public Eigen::Matrix2d
{
  Matrix2d(const Point2d & p1, const Point2d & p2) : Eigen::Matrix2d()
  {
    // Initialize the matrix based on the Point2d objects
    // For example, set the matrix elements to the x and y values of the points
    // Replace the following lines with your specific logic
    (*this)(0, 0) = p1.x();
    (*this)(0, 1) = p1.y();
    (*this)(1, 0) = p2.x();
    (*this)(1, 1) = p2.y();
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

  double getArea() const { return area; };

  int getNVertices() const { return apex.size(); };
  int getNSegments() const { return line_segments.size(); };

  /**
   * Fill an empty convex hull with its apexes.
   * @param apex_: Vector of points (C. Hull vertices ordered CCW)
   **/
  void setApexes(std::vector<Point2d> const & apex_)
  {
    apex = apex_;
    assert(apex.size() >= 3);
    computeArea();
    computeLineSegments();
  };

  /**
   * Uses the Ray casting algorithm:
   *https://en.wikipedia.org/wiki/Point_in_polygon to determine if a Point is
   *inside this Polygon
   **/
  bool isPointInside(const Point2d & p) const { return isPointInPolygon(apex, p); };

private:
  /**
      * Area of convex polygon computed following this approach
  https://byjus.com/maths/convex-polygon/ We compute and add the area of the
  inner triangles of the c. hull to get its total area. \return convex hull area
  (double)
  **/
  void computeArea()
  {  // The inner triangles of the Polygon are
     // added to get the area of the polygon
    // A formula for this is:
    // area = 0.5 * det{([x1,x2],[y1,y2]) + ([x2,x3],[y2,y3]) + ... +
    // ([xn,x1],[yn,y1])}
    area = 0;
    const Matrix2d apexMatrix(apex[apex.size() - 1], apex[0]);
    // The loop below ignores the Matrix ([xn,x1],[yn,y1]), so we add it manually
    // to the sum
    area = apexMatrix.determinant();
    for (int i = 0; i < apex.size() - 1; ++i) {
      const Matrix2d temp(apex[i], apex[i + 1]);
      area += temp.determinant();
    }

    area = 0.5 * area;
    if (area < 0.0) area *= -1.;
  };
  /**
   * Fills an internal class member: line_segments with the lines or "edges"
   * connecting each polygon vertex
   */
  void computeLineSegments()
  {
    line_segments.reserve(apex.size());

    for (int i = 0; i < apex.size() - 1; ++i) {
      Line2d segment(apex[i], apex[i + 1]);
      line_segments.push_back(segment);
    }
    Line2d segment(apex[apex.size() - 1], apex[0]);
    line_segments.push_back(segment);
  };
};

/**
    This Function uses the ray-casting algorithm to decide whether the point is
   inside the given polygon. See
   https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm. For
   this implementation, the ray goes in the -x direction, from P(x,y) to
   P(-inf,y)
    @param vertices: Vertices of the Polygon. vector of Points.
    @param p: The Point that is being tested.
    @return true or false
*/
bool isPointInPolygon(const std::vector<Point2d> & vertices, const Point2d & p)
{
  const int n_vertices = vertices.size();
  int i, j;
  bool inside = false;
  // looping for all the edges
  for (i = 0; i < n_vertices; ++i) {
    int j = (i + 1) % n_vertices;

    // The vertices of the edge we are checking.
    const double xp0 = vertices[i].x();
    const double yp0 = vertices[i].y();
    const double xp1 = vertices[j].x();
    const double yp1 = vertices[j].y();

    // Check whether the edge intersects a line from (-inf,p.y()) to (p.x(),p.y()).

    // First check if the line crosses the horizontal line at p.y() in either
    // direction.
    if ((yp0 <= p.y()) && (yp1 > p.y()) || (yp1 <= p.y()) && (yp0 > p.y())) {
      // If so, get the point where it crosses that line. Note that we can't get
      // a division by zero here - if yp1 == yp0 then the above condition is
      // false.
      const double cross = (xp1 - xp0) * (p.y() - yp0) / (yp1 - yp0) + xp0;

      // Finally check if it crosses to the left of our test point.
      if (cross < p.x()) inside = !inside;
    }
  }
  return inside;
};

/**
 * Check if two Line segments intersect
 *@param vertices: Vertices of the Polygon
 *@param l1: First Line Segment to test.
 *@param l2: Second Line Segment to test.
 *@param intersect_point: The intersection point data (if it exists) will be
 *copied here.
 *@param epsilon: The tolerance used when comparing floating point subtraction
 *results to 0
 *@return true or false
 */
bool segmentsIntersect(
  const Line2d & l1, const Line2d & l2, Point2d & intersect_point, const double epsilon = 1E-5)
{
  const double ax = l1.p2.x() - l1.p1.x();  // direction of line a
  const double ay = l1.p2.y() - l1.p1.y();  // ax and ay as above

  const double bx = l2.p1.x() - l2.p2.x();  // direction of line b, reversed
  const double by = l2.p1.y() - l2.p2.y();

  const double dx = l2.p1.x() - l1.p1.x();  // right-hand side
  const double dy = l2.p1.y() - l1.p1.y();

  const double det = ax * by - ay * bx;

  // floating point error forces us to use a non zero, small epsilon
  // lines are parallel, they could be collinear, but in that
  // case,  we dont care since the points will be detected when
  // we check if other lines of the polygon intersect
  if (std::abs(det) < epsilon) return false;

  const double t = (dx * by - dy * bx) / det;
  const double u = (ax * dy - ay * dx) / det;
  // if both t and u between 0 and 1, the segments intersect
  const bool intersect = !(t < 0 || t > 1 || u < 0 || u > 1);

  if (intersect) {
    // If both lines intersect, we have the point by the equation P = p1 +
    // (p2-p1)*t or p = p3 + (p4-p3) * u
    const auto segment = l1.p2 - l1.p1;
    intersect_point = l1.p1 + (segment)*t;
  }

  return intersect;
};

/**
 * Sorts a vector of Points CCW by setting one Point as "center", and checking
 * the angle difference between said point and the rest. The points are then
 * arranged based on the pair-wise angle different between each point and the
 * "Center".
 * @param point_vector: Vector of points to sort CCW
 */
void sortPointsCCW(std::vector<Point2d> & point_vector)
{
  Point2d center = point_vector.at(0);  //  We make a pivot to check angles against

  // sort all points by polar angle
  for (Point2d & p : point_vector) {
    double angle = center.get_angle(p);
    p.set_angle(angle);
  }

  // sort the points using overloaded < operator from the Point struct
  // this program sorts them counterclockwise;
  std::sort(point_vector.begin(), point_vector.end());
};

/**
 * If two polygons intersect, returns a non-ordered list of vertices
 * corresponding to the polygon formed by the intersection of the two original
 * polygons
 * @param p1: Polygon to check for intersection.
 * @param p2: Polygon to check for intersection.
 * @returns
 */
std::vector<Point2d> getIntersectionPolygonVertices(const Polygon2d & p1, const Polygon2d & p2)
{
  std::vector<Point2d> intersection_vertices;
  int n_vert_p1 = p1.getNVertices();
  int n_vert_p2 = p2.getNVertices();
  int n_segment_p1 = p1.getNSegments();
  int n_segment_p2 = p2.getNSegments();

  intersection_vertices.reserve(n_vert_p1 + n_vert_p2);
  // Check which apexes of Polygon1 (if any) are inside polygon2
  for (int i = 0; i < n_vert_p1; ++i) {
    if (p2.isPointInside(p1.apex[i])) {
      intersection_vertices.push_back(p1.apex[i]);
    }
  }
  // Check which apexes of Polygon2 (if any) are inside polygon1
  for (int i = 0; i < n_vert_p2; ++i) {
    if (p1.isPointInside(p2.apex[i])) {
      intersection_vertices.push_back(p2.apex[i]);
    }
  }
  // Check if the line segments connecting each apex of each polygon, happen
  // to intersect
  for (int i = 0; i < n_segment_p1; ++i) {
    for (int j = 0; j < n_segment_p2; ++j) {
      Point2d intersection;
      const double eps = 0.00001;

      const bool segments_intersect =
        segmentsIntersect(p1.line_segments[i], p2.line_segments[j], intersection, eps);
      if (segments_intersect) {
        // If the segments intersect, they create a Vertex for the intersection
        // polygon
        intersection_vertices.push_back(intersection);
      }
    }
  }
  return intersection_vertices;
};

/**
 * If two polygons intersect, stores the corresponding polygon created by the
 * intersection
 * @param p1: Polygon to check for intersection.
 * @param p2: Polygon to check for intersection.
 * @param intersection: Intersecting polygon (if it exists) data is stored here
 * @returns true or false, if the polygons intersect or not
 */
bool getIntersectingPolygon(const Polygon2d & p1, const Polygon2d & p2, Polygon2d & intersection)
{
  std::vector<Point2d> intersection_vertices = getIntersectionPolygonVertices(p1, p2);

  if (intersection_vertices.size() < 3)
    return false;  // intersect polygon requires 3 vertices to exist

  // Now we have the points that make the intersection of two polygons, we
  // need to organize them CCW
  sortPointsCCW(intersection_vertices);
  // Add Points to the C. Hull
  intersection.setApexes(intersection_vertices);
  return true;
};

}  // namespace tier4_geometry
}  // namespace tier4_autoware_utils

#endif  // TIER4_AUTOWARE_UTILS__GEOMETRY__TIER4_GEOMETRY_HPP_
