// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Ikhyeon Cho <tre0430@korea.ac.kr>

/*
 * drop_detection.cpp
 *
 * Detection of negative obstacles (drops, cliffs, edges).
 *
 *  Created on: Aug 2026
 *      Author: FastDEM Contributors
 *   Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#include "fastdem/postprocess/drop_detection.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <queue>

namespace fastdem {

namespace {

/**
 * @brief Compute reference height for drop detection.
 *
 * For now, this returns a constant height based on robot z position.
 * In future, can be extended to compute support plane using roll/pitch.
 */
float computeReferenceZ(const Eigen::Isometry3d& T_world_base,
                        const config::DropDetection& config) {
  float reference_z = static_cast<float>(T_world_base.translation().z());
  reference_z += config.reference_z_offset;
  return reference_z;
}

/**
 * @brief Apply morphological dilation (inflation) to binary drop mask.
 *
 * Simple 4-neighbor BFS-based dilation.
 */
void dilateDropMask(nanogrid::Matrix& drop_mat, float resolution,
                    float inflation_radius) {
  if (inflation_radius <= 0.0f) return;

  const int num_dilations =
      static_cast<int>(std::ceil(inflation_radius / resolution));
  if (num_dilations <= 0) return;

  nanogrid::Matrix result = drop_mat;
  Eigen::Index rows = result.rows();
  Eigen::Index cols = result.cols();

  for (int dil = 0; dil < num_dilations; ++dil) {
    nanogrid::Matrix temp = result;
    for (Eigen::Index r = 0; r < rows; ++r) {
      for (Eigen::Index c = 0; c < cols; ++c) {
        if (temp(r, c) > 0.5f) {
          // Dilate to 4-neighbors
          if (r > 0) result(r - 1, c) = 1.0f;
          if (r < rows - 1) result(r + 1, c) = 1.0f;
          if (c > 0) result(r, c - 1) = 1.0f;
          if (c < cols - 1) result(r, c + 1) = 1.0f;
        }
      }
    }
  }

  drop_mat = result;
}

/**
 * @brief Filter small connected NaN (unknown) components.
 *
 * Cells that are NaN and part of small connected components are changed to
 * drop = 0.0 (safe). Larger NaN regions remain as drop hazards.
 */
void filterSmallUnknownHoles(nanogrid::Matrix& drop_mat,
                             const nanogrid::Matrix& elevation_mat,
                             float resolution, float max_safe_unknown_hole_size) {
  if (max_safe_unknown_hole_size <= 0.0f) return;

  Eigen::Index rows = drop_mat.rows();
  Eigen::Index cols = drop_mat.cols();

  // Find cell area
  float cell_area = resolution * resolution;
  int max_cells = static_cast<int>(std::ceil(max_safe_unknown_hole_size / cell_area));

  // Visited matrix
  std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));

  // BFS for connected components
  for (Eigen::Index r = 0; r < rows; ++r) {
    for (Eigen::Index c = 0; c < cols; ++c) {
      if (!visited[r][c] && std::isnan(elevation_mat(r, c)) &&
          drop_mat(r, c) > 0.5f) {  // Unknown and marked as drop
        // BFS to find component
        std::queue<std::pair<Eigen::Index, Eigen::Index>> q;
        q.push({r, c});
        visited[r][c] = true;

        std::vector<std::pair<Eigen::Index, Eigen::Index>> component;
        while (!q.empty()) {
          auto [cr, cc] = q.front();
          q.pop();
          component.push_back({cr, cc});

          // 4-neighbors
          const std::pair<Eigen::Index, Eigen::Index> neighbors[] = {
              {cr - 1, cc}, {cr + 1, cc}, {cr, cc - 1}, {cr, cc + 1}};
          for (auto [nr, nc] : neighbors) {
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                !visited[nr][nc] && std::isnan(elevation_mat(nr, nc)) &&
                drop_mat(nr, nc) > 0.5f) {
              visited[nr][nc] = true;
              q.push({nr, nc});
            }
          }
        }

        // If component is small, mark as safe
        if (static_cast<int>(component.size()) <= max_cells) {
          for (auto [cr, cc] : component) {
            drop_mat(cr, cc) = 0.0f;
          }
        }
      }
    }
  }
}

}  // namespace

void applyDropDetection(ElevationMap& map,
                        const Eigen::Isometry3d& T_world_base,
                        const config::DropDetection& config) {
  if (!config.enabled) return;

  // Validate preconditions
  if (!map.exists(layer::elevation)) {
    spdlog::warn("[DropDetection] Missing required layer: elevation.");
    return;
  }

  float resolution = map.getResolution();
  float reference_z = computeReferenceZ(T_world_base, config);

  // Robot position in map frame (2D)
  Eigen::Vector2f robot_pos =
      T_world_base.translation().head<2>().cast<float>();

  // Initialize output layers (add if not exist)
  if (!map.exists(layer::drop)) map.add(layer::drop);
  if (!map.exists(layer::drop_obstacle_z)) map.add(layer::drop_obstacle_z);

  // Get output layer references
  nanogrid::Matrix& drop_mat = map.get(layer::drop);
  nanogrid::Matrix& obstacle_z_mat = map.get(layer::drop_obstacle_z);

  // Clear and initialize layers with NaN
  drop_mat.setConstant(NAN);
  obstacle_z_mat.setConstant(NAN);

  // Get elevation layer reference
  const nanogrid::Matrix& elevation_mat = map.get(layer::elevation);

  float min_range_sq = config.min_detection_range * config.min_detection_range;
  float max_range_sq = config.max_detection_range * config.max_detection_range;

  // Process all cells
  Eigen::Index rows = map.getSize()(0);
  Eigen::Index cols = map.getSize()(1);

  for (Eigen::Index r = 0; r < rows; ++r) {
    for (Eigen::Index c = 0; c < cols; ++c) {
      // Get cell position
      nanogrid::Index idx(r, c);
      auto pos_opt = map.position(idx);
      if (!pos_opt) continue;

      Eigen::Vector2f cell_pos = pos_opt->cast<float>();
      Eigen::Vector2f delta = cell_pos - robot_pos;
      float dist_sq = delta.squaredNorm();

      // Check range
      if (dist_sq < min_range_sq || dist_sq > max_range_sq) {
        drop_mat(r, c) = 0.0f;
        obstacle_z_mat(r, c) = NAN;
        continue;
      }

      float elevation = elevation_mat(r, c);
      bool is_drop = false;

      if (std::isfinite(elevation)) {
        // Known elevation
        float drop_depth = reference_z - elevation;
        is_drop = drop_depth > config.drop_height_threshold;
      } else {
        // Unknown (NaN) elevation
        is_drop = config.unknown_is_drop;
      }

      // Write drop layer
      drop_mat(r, c) = is_drop ? 1.0f : 0.0f;

      // Write obstacle z layer
      if (is_drop) {
        obstacle_z_mat(r, c) = reference_z + config.virtual_obstacle_height;
      } else {
        obstacle_z_mat(r, c) = NAN;
      }
    }
  }

  // Optional: filter small unknown holes
  if (config.filter_small_unknown_holes) {
    filterSmallUnknownHoles(drop_mat, elevation_mat, resolution,
                            config.max_safe_unknown_hole_size);

    // Update obstacle_z to match filtered drop layer
    for (Eigen::Index r = 0; r < rows; ++r) {
      for (Eigen::Index c = 0; c < cols; ++c) {
        if (drop_mat(r, c) < 0.5f) {
          obstacle_z_mat(r, c) = NAN;
        }
      }
    }
  }

  // Optional: dilate drop mask
  if (config.inflation_radius > 0.0f) {
    dilateDropMask(drop_mat, resolution, config.inflation_radius);

    // Update obstacle_z to match dilated drop layer
    for (Eigen::Index r = 0; r < rows; ++r) {
      for (Eigen::Index c = 0; c < cols; ++c) {
        if (drop_mat(r, c) > 0.5f) {
          if (std::isnan(obstacle_z_mat(r, c))) {
            obstacle_z_mat(r, c) =
                reference_z + config.virtual_obstacle_height;
          }
        } else {
          obstacle_z_mat(r, c) = NAN;
        }
      }
    }
  }
}

}  // namespace fastdem
