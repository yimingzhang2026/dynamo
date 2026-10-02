/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

// Topology holds an opaque compiler name and a chip count.
type Topology struct {
	// ChipCount is the number of LPU chips represented by the topology.
	ChipCount int
	// Raw is the original compiler topology string.
	Raw string
}
