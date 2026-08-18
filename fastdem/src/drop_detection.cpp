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
 * @brief Compute reference height for drop detection at a given cell.
 *
 * Without tilt compensation: constant horizontal plane at robot base z.
 * With tilt compensation: tilted plane matching robot pitch/roll, so cells
 * on the continuation of the robot's current stance are not false drops.
 */
float computeReferenceZ(const Eigen::Isometry3d& T_world_base,
                        const config::DropDetection& config,
                        const Eigen::Vector2f& cell_pos) {
  const Eigen::Vector3d robot_pos = T_world_base.translation();
  const float base_z =
      static_cast<float>(robot_pos.z()) + config.reference_z_offset;

  if (!config.compensate_robot_tilt) return base_z;

  // Robot up-vector in world frame (col 2 of rotation matrix)
  const Eigen::Vector3d up = T_world_base.rotation().col(2);
  if (std::abs(up.z()) < 1e-6f) return base_z;  // degenerate tilt, fall back

  // Intersect vertical line at cell_pos with robot's support plane:
  //   up · (p - robot_pos) = 0  →  z = robot_z - (up_x*dx + up_y*dy) / up_z
  const float dx = cell_pos.x() - static_cast<float>(robot_pos.x());
  const float dy = cell_pos.y() - static_cast<float>(robot_pos.y());
  return base_z -
         static_cast<float>((up.x() * dx + up.y() * dy) / up.z());
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

  // Robot position in map frame (2D)
  Eigen::Vector2f robot_pos =
      T_world_base.translation().head<2>().cast<float>();

  // Initialize output layers (add if not exist)
  if (!map.exists(layer::drop)) map.add(layer::drop);
  if (!map.exists(layer::drop_obstacle_z)) map.add(layer::drop_obstacle_z);
  if (!map.exists(layer::drop_source)) map.add(layer::drop_source);

  // Get output layer references
  nanogrid::Matrix& drop_mat = map.get(layer::drop);
  nanogrid::Matrix& obstacle_z_mat = map.get(layer::drop_obstacle_z);
  nanogrid::Matrix& source_mat = map.get(layer::drop_source);

  // Clear and initialize layers with NaN
  drop_mat.setConstant(NAN);
  obstacle_z_mat.setConstant(NAN);
  source_mat.setConstant(NAN);

  // Get elevation layer reference
  const nanogrid::Matrix& elevation_mat = map.get(layer::elevation);

  float min_range_sq = config.min_detection_range * config.min_detection_range;
  float max_range_sq = config.max_detection_range * config.max_detection_range;

  // Precompute forward-sector check (robot x-axis is forward by convention)
  const Eigen::Vector2f forward =
      T_world_base.rotation().block<2, 1>(0, 0).cast<float>().normalized();
  const float cos_half_aperture = std::cos(config.forward_aperture_rad * 0.5f);
  const bool use_sector = config.forward_aperture_rad < (2.0f * static_cast<float>(M_PI));

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
        source_mat(r, c) = NAN;
        safe_mat(r, c) = NAN;
        continue;
      }

      // Check forward sector
      if (use_sector && dist_sq > 0.0f) {
        if (delta.normalized().dot(forward) < cos_half_aperture) {
          drop_mat(r, c) = 0.0f;
          obstacle_z_mat(r, c) = NAN;
          source_mat(r, c) = NAN;
          safe_mat(r, c) = NAN;
          continue;
        }
      }

      const float reference_z = computeReferenceZ(T_world_base, config, cell_pos);
      float elevation = elevation_mat(r, c);
      bool is_drop = false;
      float source_value = 0.0f;  // safe

      if (std::isfinite(elevation)) {
        float drop_depth = reference_z - elevation;
        is_drop = drop_depth > config.drop_height_threshold;
        source_value = is_drop ? 1.0f : 0.0f;
      } else {
        is_drop = config.unknown_is_drop;
        source_value = is_drop ? 2.0f : 0.0f;
      }

      drop_mat(r, c) = is_drop ? 1.0f : 0.0f;
      source_mat(r, c) = source_value;
      safe_mat(r, c) = is_drop ? NAN : 1.0f;

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

    // Update obstacle_z and source to match filtered drop layer
    for (Eigen::Index r = 0; r < rows; ++r) {
      for (Eigen::Index c = 0; c < cols; ++c) {
        if (drop_mat(r, c) < 0.5f) {
          obstacle_z_mat(r, c) = NAN;
          if (source_mat(r, c) > 0.5f) source_mat(r, c) = 0.0f;
          if (std::isnan(safe_mat(r, c))) safe_mat(r, c) = 1.0f;
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
            nanogrid::Index idx(r, c);
            auto pos_opt = map.position(idx);
            const Eigen::Vector2f cell_pos =
                pos_opt ? pos_opt->cast<float>() : robot_pos;
            const float ref_z =
                computeReferenceZ(T_world_base, config, cell_pos);
            obstacle_z_mat(r, c) = ref_z + config.virtual_obstacle_height;
          }
        } else {
          obstacle_z_mat(r, c) = NAN;
        }
      }
    }
  }
}

}  // namespace fastdem
