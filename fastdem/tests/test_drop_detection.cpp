// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Ikhyeon Cho <tre0430@korea.ac.kr>

/*
 * test_drop_detection.cpp
 *
 * Unit tests for drop detection algorithm.
 */

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "fastdem/postprocess/drop_detection.hpp"

using namespace fastdem;

// ─── Fixture ────────────────────────────────────────────────────────────────

class DropDetectionTest : public ::testing::Test {
 protected:
  ElevationMap map;
  config::DropDetection config;

  void SetUp() override {
    // 10m x 10m, 0.5m resolution → 20x20 grid
    map.setGeometry(10.0f, 10.0f, 0.5f);

    // Default config
    config.enabled = true;
    config.reference_z_offset = 0.0f;
    config.drop_height_threshold = 0.05f;
    config.unknown_is_drop = true;
    config.min_detection_range = 0.3f;
    config.max_detection_range = 3.0f;
    config.filter_small_unknown_holes = false;
    config.max_safe_unknown_hole_size = 0.10f;
    config.inflation_radius = 0.0f;
    config.virtual_obstacle_height = 0.30f;
    config.compensate_robot_tilt = false;
  }

  nanogrid::Index centerIndex() const {
    auto idxOpt = map.index(nanogrid::Position(0.0, 0.0));
    return *idxOpt;
  }

  /// Get pose with robot at origin, height z
  Eigen::Isometry3d getRobotPose(float z) const {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(0.0, 0.0, z);
    return pose;
  }
};

// ─── Test: Basic Detection ──────────────────────────────────────────────────

TEST_F(DropDetectionTest, DetectsDropBelowThreshold) {
  // Set center cell to low elevation (a drop)
  auto center = centerIndex();
  map.at(layer::elevation, center) = 0.5f;  // Well below robot (at 1.0)

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop));
  float drop_value = map.at(layer::drop, center);
  EXPECT_FLOAT_EQ(drop_value, 1.0f) << "Center should be marked as drop";
}

TEST_F(DropDetectionTest, IgnoresSafeTerrain) {
  // Set center cell to high elevation (safe)
  auto center = centerIndex();
  map.at(layer::elevation, center) = 0.98f;  // Just below robot (at 1.0)

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop));
  float drop_value = map.at(layer::drop, center);
  EXPECT_FLOAT_EQ(drop_value, 0.0f) << "Center should be safe";
}

TEST_F(DropDetectionTest, RespectDetectionRange) {
  // Place a cell outside max detection range
  auto center = centerIndex();

  // Find a cell far away (> 3.0 m)
  auto far_idx_opt = map.index(nanogrid::Position(3.5, 0.0));
  if (!far_idx_opt) return;  // Skip if out of map
  auto far_idx = *far_idx_opt;

  map.at(layer::elevation, far_idx) = 0.5f;  // Drop elevation

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop));
  float drop_value = map.at(layer::drop, far_idx);
  EXPECT_FLOAT_EQ(drop_value, 0.0f) << "Far cell should not be detected";
}

TEST_F(DropDetectionTest, HandlesUnknownAsDropHazard) {
  // Unknown (NaN) elevation at center
  auto center = centerIndex();
  // Leave elevation as NaN (unmeasured)

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop));
  float drop_value = map.at(layer::drop, center);
  EXPECT_FLOAT_EQ(drop_value, 1.0f) << "Unknown should be treated as drop";
}

TEST_F(DropDetectionTest, HandlesUnknownAsSafeWhenDisabled) {
  // Set unknown_is_drop = false
  config.unknown_is_drop = false;

  auto center = centerIndex();
  // Leave elevation as NaN (unmeasured)

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop));
  float drop_value = map.at(layer::drop, center);
  EXPECT_FLOAT_EQ(drop_value, 0.0f) << "Unknown should be safe when disabled";
}

// ─── Test: Obstacle Z Layer ────────────────────────────────────────────────

TEST_F(DropDetectionTest, ObstacleZIsSetForDrops) {
  auto center = centerIndex();
  map.at(layer::elevation, center) = 0.5f;

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop_obstacle_z));
  float obstacle_z = map.at(layer::drop_obstacle_z, center);
  EXPECT_NEAR(obstacle_z, 1.0f + 0.30f, 1e-5f)
      << "Obstacle Z should be reference + virtual_obstacle_height";
}

TEST_F(DropDetectionTest, ObstacleZIsNaNForSafe) {
  auto center = centerIndex();
  map.at(layer::elevation, center) = 0.98f;

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop_obstacle_z));
  float obstacle_z = map.at(layer::drop_obstacle_z, center);
  EXPECT_TRUE(std::isnan(obstacle_z)) << "Safe cells should have NaN obstacle_z";
}

// ─── Test: Reference Z Offset ──────────────────────────────────────────────

TEST_F(DropDetectionTest, ReferencesZOffsetApplied) {
  config.reference_z_offset = 0.5f;  // Robot is 0.5m above base

  auto center = centerIndex();
  map.at(layer::elevation, center) = 0.5f;

  Eigen::Isometry3d pose = getRobotPose(1.0f);  // Robot base at 1.0m
  applyDropDetection(map, pose, config);

  // Reference should be 1.0 + 0.5 = 1.5, so drop depth = 1.5 - 0.5 = 1.0 > threshold
  ASSERT_TRUE(map.exists(layer::drop));
  float drop_value = map.at(layer::drop, center);
  EXPECT_FLOAT_EQ(drop_value, 1.0f) << "Reference offset should affect detection";
}

// ─── Test: Threshold ───────────────────────────────────────────────────────

TEST_F(DropDetectionTest, RespectThreshold) {
  config.drop_height_threshold = 0.3f;  // High threshold

  auto center = centerIndex();
  map.at(layer::elevation, center) = 0.9f;  // Drop depth = 1.0 - 0.9 = 0.1 < 0.3

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop));
  float drop_value = map.at(layer::drop, center);
  EXPECT_FLOAT_EQ(drop_value, 0.0f) << "Small drop should not trigger";
}

// ─── Test: Inflation ───────────────────────────────────────────────────────

TEST_F(DropDetectionTest, InflatesDropRegion) {
  config.inflation_radius = 1.0f;  // 1m dilation

  auto center = centerIndex();
  map.at(layer::elevation, center) = 0.5f;

  // Set adjacent cell elevation so it would normally be safe
  auto right_idx_opt = map.index(nanogrid::Position(0.5, 0.0));
  if (right_idx_opt) {
    map.at(layer::elevation, *right_idx_opt) = 0.99f;
  }

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop));
  // Center should definitely be drop
  EXPECT_FLOAT_EQ(map.at(layer::drop, center), 1.0f);

  // Adjacent cell should also be marked as drop due to inflation
  if (right_idx_opt) {
    float right_drop = map.at(layer::drop, *right_idx_opt);
    EXPECT_FLOAT_EQ(right_drop, 1.0f)
        << "Adjacent cell should be inflated to drop";
  }
}

// ─── Test: Disabled Detection ──────────────────────────────────────────────

TEST_F(DropDetectionTest, IgnoredWhenDisabled) {
  config.enabled = false;

  auto center = centerIndex();
  map.at(layer::elevation, center) = 0.5f;

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  // Layers should not be created if disabled
  EXPECT_FALSE(map.exists(layer::drop)) << "Disabled detection should not create layers";
}

// ─── Test: Filter Small Unknown Holes ───────────────────────────────────────

TEST_F(DropDetectionTest, FilterSmallNaNHoles) {
  config.filter_small_unknown_holes = true;
  config.max_safe_unknown_hole_size = 0.5f;  // Max 0.5 m²

  auto center = centerIndex();
  // Leave center as NaN (unknown)

  // Set all neighbors to known safe elevation
  constexpr int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
  constexpr int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
  for (int i = 0; i < 8; ++i) {
    nanogrid::Index neighbor(center(0) + dy[i], center(1) + dx[i]);
    map.at(layer::elevation, neighbor) = 0.95f;
  }

  Eigen::Isometry3d pose = getRobotPose(1.0f);
  applyDropDetection(map, pose, config);

  ASSERT_TRUE(map.exists(layer::drop));
  float drop_value = map.at(layer::drop, center);
  EXPECT_FLOAT_EQ(drop_value, 0.0f)
      << "Small isolated NaN hole should be filtered out";
}

}  // namespace fastdem
