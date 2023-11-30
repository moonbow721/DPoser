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

# Lint as: python3

from configs.default_amass_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'subvpsde'
    training.continuous = True

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # model
    model = config.model
    model.type = 'ScoreModelFC'
    model.HIDDEN_DIM = 1024
    model.EMBED_DIM = 512
    model.N_BLOCKS = 2
    model.dropout = 0.1
    model.fourier_scale = 16
    model.scale_by_sigma = True
    model.ema_rate = 0.9999
    model.nonlinearity = 'swish'
    model.embedding_type = 'positional'    # Or 'fourier'

    return config
