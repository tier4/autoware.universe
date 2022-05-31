// Copyright 2022 Tier IV, Inc. All rights reserved.
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

#include "circular_graph.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <iostream>

struct Node
{
  size_t id;
  std::vector<size_t> child_ids;
  void add_child(size_t id) { child_ids.push_back(id); }
};

const auto is_stoppable_trajectory_default = [](const std::vector<Node> &) { return true; };

class SimpleGraph : public auto_parking_planner::CircularGraphBase<Node>
{
public:
  explicit SimpleGraph(
    const std::vector<Node> & nodes, std::function<bool(const std::vector<Node> &)> f_is_stoppable)
  : auto_parking_planner::CircularGraphBase<Node>(f_is_stoppable), nodes_(nodes)
  {
  }

  size_t getID(const Node & node) const override { return node.id; }

  size_t getElementNum() const override { return nodes_.size(); }

  Node get_node(size_t id) const { return nodes_.at(id); }

  std::vector<Node> getFollowings(const Node & node) const override
  {
    std::vector<Node> child_nodes;
    for (size_t id : node.child_ids) {
      child_nodes.push_back(nodes_[id]);
    }
    return child_nodes;
  };

  std::vector<Node> getReachables(const Node & node) const override
  {
    std::unordered_set<size_t> visited_set;
    std::vector<Node> reachables;
    std::stack<Node> s;
    s.push(node);
    while (!s.empty()) {
      const auto node_here = s.top();
      reachables.push_back(node_here);
      visited_set.insert(node_here.id);
      s.pop();
      for (const auto & node_child : getFollowings(node_here)) {
        if (visited_set.find(node_child.id) != visited_set.end()) continue;
        s.push(node_child);
      }
    }
    return reachables;
  }

  std::vector<Node> nodes_;
};

SimpleGraph build_graph_with_loop(
  const std::function<bool(const std::vector<Node> &)> & f_is_stoppable)
{
  std::vector<Node> nodes;
  for (size_t i = 0; i < 14; i++) {
    nodes.push_back(Node{i, {}});
  }
  nodes[0].add_child(1);
  nodes[1].add_child(2);
  nodes[2].add_child(3);
  nodes[3].add_child(4);
  nodes[3].add_child(8);
  nodes[4].add_child(5);
  nodes[5].add_child(6);
  nodes[6].add_child(7);
  nodes[7].add_child(9);
  nodes[8].add_child(7);
  nodes[9].add_child(10);
  nodes[10].add_child(11);
  nodes[11].add_child(2);
  nodes[11].add_child(12);
  nodes[12].add_child(13);
  return SimpleGraph(nodes, f_is_stoppable);
}

void expect_consistent_nodes_and_ids(const std::vector<Node> & nodes, const std::vector<size_t> ids)
{
  EXPECT_TRUE(nodes.size() == ids.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    EXPECT_TRUE(nodes[i].id == ids[i]);
  }
}

SimpleGraph build_graph_without_loop()
{
  std::vector<Node> nodes;
  const size_t n_node = 6;
  for (size_t i = 0; i < n_node; i++) {
    nodes.push_back(Node{i, {}});
  }

  for (size_t i = 0; i < n_node - 1; i++) {
    nodes[i].add_child(i + 1);
  }
  return SimpleGraph(nodes, [](const std::vector<Node> &) { return true; });
}

TEST(SimpleGraphTestSuite, OverrideMethods)
{
  const auto graph = build_graph_with_loop(is_stoppable_trajectory_default);

  // test SimpleGraph
  for (size_t i = 0; i < 14; ++i) {
    const auto node = graph.get_node(i);
    const auto nodes_following = graph.getFollowings(node);
    if (i == 3) {
      EXPECT_TRUE(nodes_following.size() == 2);
    } else if (i == 11) {
      EXPECT_TRUE(nodes_following.size() == 2);
    } else if (i == 13) {
      EXPECT_TRUE(nodes_following.size() == 0);
    } else {
      EXPECT_TRUE(nodes_following.size() == 1);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    const auto node = graph.get_node(i);
    const auto nodes_reachable = graph.getReachables(node);
    EXPECT_TRUE(nodes_reachable.size() == 14 - i);
  }

  for (size_t i = 3; i < 12; ++i) {
    const auto node = graph.get_node(i);
    const auto nodes_reachable = graph.getReachables(node);
    EXPECT_TRUE(nodes_reachable.size() == 12);
  }

  for (size_t i = 12; i < 14; ++i) {
    const auto node = graph.get_node(i);
    const auto nodes_reachable = graph.getReachables(node);
    EXPECT_TRUE(nodes_reachable.size() == 14 - i);
  }
}

TEST(CircularGraph, LoopCase)
{
  const auto graph = build_graph_with_loop(is_stoppable_trajectory_default);

  {
    const auto partial_path_seq = graph.planCircularPathSequence(graph.get_node(0));
    EXPECT_TRUE(partial_path_seq.size() == 2);
    std::vector<size_t> idseq_expected1{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11};
    std::vector<size_t> idseq_expected2{11, 2, 3, 8};
    expect_consistent_nodes_and_ids(partial_path_seq.at(0), idseq_expected1);
    expect_consistent_nodes_and_ids(partial_path_seq.at(1), idseq_expected2);
  }

  {
    const auto partial_path_seq = graph.planCircularPathSequence(graph.get_node(8));
    EXPECT_TRUE(partial_path_seq.size() == 1);
    std::vector<size_t> idseq_expected1{8, 7, 9, 10, 11, 2, 3, 4, 5, 6};
    expect_consistent_nodes_and_ids(partial_path_seq.at(0), idseq_expected1);
  }
}

TEST(CircularGraph, LoopCaseWithStoppableCondition)
{
  const auto is_stoppable = [](const std::vector<Node> & node_seq) {
    const auto & node = node_seq.back();
    if (node.id == 11) return false;
    if (node.id == 8) return false;
    return true;
  };
  const auto graph = build_graph_with_loop(is_stoppable);
  const auto partial_path_seq = graph.planCircularPathSequence(graph.get_node(0));
  EXPECT_TRUE(partial_path_seq.size() == 2);
  {
    std::vector<size_t> idseq_expected1{0, 1, 2, 3, 4, 5, 6, 7, 9, 10};
    std::vector<size_t> idseq_expected2{10, 11, 2, 3};
    expect_consistent_nodes_and_ids(partial_path_seq.at(0), idseq_expected1);
    expect_consistent_nodes_and_ids(partial_path_seq.at(1), idseq_expected2);
  }
}

TEST(CircularGraph, WithoutLoopCase)
{
  const auto graph = build_graph_without_loop();
  const auto partial_path_seq = graph.planCircularPathSequence(graph.get_node(0));
  EXPECT_TRUE(partial_path_seq.size() == 1);

  std::vector<size_t> idseq_expected{0, 1, 2, 3, 4, 5};
  expect_consistent_nodes_and_ids(partial_path_seq.at(0), idseq_expected);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
