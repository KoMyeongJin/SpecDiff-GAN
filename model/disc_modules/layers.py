# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Score SDE library
# which was released under the Apache License.
#
# Source:
# https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_Apache). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Common layers for defining score networks. 
    Adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/layers.py
"""

import math
import torch
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb