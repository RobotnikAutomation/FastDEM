// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Ikhyeon Cho <tre0430@korea.ac.kr>

/*
 * drop_detection.hpp
 *
 * Detection of negative obstacles (drops, cliffs, edges) via elevation
 * comparison against robot reference height.
 *
 *  Created on: Aug 2026
 *      Author: FastDEM Contributors
 *   Institute: Korea Univ. ISR (Intelligent Systems & Robotics) Lab
 *       Email: tre0430@korea.ac.kr
 */

#ifndef FASTDEM_POSTPROCESS_DROP_DETECTION_HPP
#define FASTDEM_POSTPROCESS_DROP_DETECTION_HPP

#include <Eigen/Core>

#include "fastdem/config/postprocess.hpp"
#include "fastdem/elevation_map.hpp"

namespace fastdem {

/**
 * @brief Detect and mark negative obstacles (drops/cliffs) in elevation map.
 *
 * For each map cell within configured detection range from the robot:
 * - If elevation is finite: is_drop = (reference_z - elevation > threshold)
 * - If elevation is NaN: is_drop = unknown_is_drop
 * - Outside range: is_drop = false
 *
 * Creates two output layers:
 * - "drop" (visible): binary mask (0.0 = safe, 1.0 = drop hazard)
 * - "_drop_obstacle_z" (internal): z-coordinate for collision obstacle
 *     (reference_z + virtual_obstacle_height for drops, NaN for safe)
 *
 * @param map Height map to update (will add/modify drop and _drop_obstacle_z layers)
 * @param T_world_base Current pose of robot base in map frame
 * @param config Drop detection configuration
 */
void applyDropDetection(ElevationMap& map,
                        const Eigen::Isometry3d& T_world_base,
                        const config::DropDetection& config);

}  // namespace fastdem

#endif  // FASTDEM_POSTPROCESS_DROP_DETECTION_HPP
