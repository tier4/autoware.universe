// Copyright 2015-2019 Autoware Foundation
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

#ifndef NDT__BASE_HPP_
#define NDT__BASE_HPP_

#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <vector>

template <class PointSource, class PointTarget>
class NormalDistributionsTransformBase
{
public:
  NormalDistributionsTransformBase();
  virtual ~NormalDistributionsTransformBase() = default;

  virtual void align(pcl::PointCloud<PointSource> & output, const Eigen::Matrix4f & guess) = 0;
  virtual void setInputTarget(const pcl::shared_ptr<pcl::PointCloud<PointTarget>> & map_ptr) = 0;
  virtual void setInputSource(const pcl::shared_ptr<pcl::PointCloud<PointSource>> & scan_ptr) = 0;

  virtual void setMaximumIterations(int max_iter) = 0;
  virtual void setResolution(float res) = 0;
  virtual void setStepSize(double step_size) = 0;
  virtual void setTransformationEpsilon(double trans_eps) = 0;

  virtual int getMaximumIterations() = 0;
  virtual int getFinalNumIteration() const = 0;
  virtual float getResolution() const = 0;
  virtual double getStepSize() const = 0;
  virtual double getTransformationEpsilon() = 0;
  virtual double getTransformationProbability() const = 0;
  virtual double getNearestVoxelTransformationLikelihood() const = 0;
  virtual double getFitnessScore() = 0;
  virtual pcl::shared_ptr<const pcl::PointCloud<PointTarget>> getInputTarget() const = 0;
  virtual pcl::shared_ptr<const pcl::PointCloud<PointSource>> getInputSource() const = 0;
  virtual Eigen::Matrix4f getFinalTransformation() const = 0;
  virtual std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
  getFinalTransformationArray() const = 0;

  virtual Eigen::Matrix<double, 6, 6> getHessian() const = 0;

  virtual pcl::shared_ptr<pcl::search::KdTree<PointTarget>> getSearchMethodTarget() const = 0;

  virtual double calculateTransformationProbability(
    const pcl::PointCloud<PointSource> & trans_cloud) const = 0;
  virtual double calculateNearestVoxelTransformationLikelihood(
    const pcl::PointCloud<PointSource> & trans_cloud) const = 0;
};

#include "ndt/impl/base.hpp"

#endif  // NDT__BASE_HPP_
