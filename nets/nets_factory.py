# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import alexnet
from nets import cifarnet
from nets import inception
from nets import lenet
from nets import mobilenet_v1
from nets import overfeat
from nets import resnet_v1
from nets import resnet_v2
from nets import vgg
from nets import xception
from nets import root_resnet_base

slim = tf.contrib.slim

networks_map = {'alexnet_v2': alexnet.alexnet_v2,
                'cifarnet': cifarnet.cifarnet,
                'overfeat': overfeat.overfeat,
                'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'inception_v1': inception.inception_v1,
                'inception_v2': inception.inception_v2,
                'inception_v3': inception.inception_v3,
                'inception_v4': inception.inception_v4,
                'inception_resnet_v2': inception.inception_resnet_v2,
                'lenet': lenet.lenet,
                'resnet_v1_50': resnet_v1.resnet_v1_50,
                'resnet_v1_101': resnet_v1.resnet_v1_101,
                'resnet_v1_152': resnet_v1.resnet_v1_152,
                'resnet_v1_200': resnet_v1.resnet_v1_200,
                'resnet_v2_50': resnet_v2.resnet_v2_50,
                'resnet_v2_101': resnet_v2.resnet_v2_101,
                'resnet_v2_152': resnet_v2.resnet_v2_152,
                'resnet_v2_200': resnet_v2.resnet_v2_200,
                'mobilenet_v1': mobilenet_v1.mobilenet_v1,
                'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_075,
                'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
                'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,
                'xception': xception.xception,
                }


arg_scopes_map = {'alexnet_v2': alexnet.alexnet_v2_arg_scope,
                  'cifarnet': cifarnet.cifarnet_arg_scope,
                  'overfeat': overfeat.overfeat_arg_scope,
                  'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'inception_v1': inception.inception_v3_arg_scope,
                  'inception_v2': inception.inception_v3_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'inception_v4': inception.inception_v4_arg_scope,
                  'inception_resnet_v2': inception.inception_resnet_v2_arg_scope,
                  'lenet': lenet.lenet_arg_scope,
                  'resnet_v1_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_101': resnet_v1.resnet_arg_scope,
                  'resnet_v1_152': resnet_v1.resnet_arg_scope,
                  'resnet_v1_200': resnet_v1.resnet_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'resnet_v2_200': resnet_v2.resnet_arg_scope,
                  'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,
                  'xception': xception.xception_arg_scope,
                  }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=True):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images):
        arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn


# TODO: build base network for all feature extractors
base_networks_map = {'inception_resnet_v2': inception.inception_resnet_v2_base,
                     'inception_v1': inception.inception_v1_base,
                     'inception_v2': inception.inception_v2_base,
                     'inception_v3': inception.inception_v3_base,
                     'inception_v4': inception.inception_v4_base,
                     'mobilenet_v1': mobilenet_v1.mobilenet_v1_base_ssd,
                     'vgg_16': vgg.vgg_16_base,
                     'vgg_a': vgg.vgg_a_base,
                     'vgg_19': vgg.vgg_19_base,
                     'resnet_v1_50': resnet_v1.resnet_v1_50_base,

                    'root_resnet_18': root_resnet_base.root_resnet_18_base,
                    'root_resnet_34': root_resnet_base.root_resnet_34_base,
                     }


base_arg_scopes_map = {'vgg_16': vgg.vgg_base_arg_scope,
                       'mobilenet_v1': mobilenet_v1.mobilenet_v1_base_arg_scope,
                       'resnet_v1_50': resnet_v1.resnet_arg_scope,
                       'root_resnet_18': root_resnet_base.root_resnet_arg_scope,
                        'root_resnet_34': root_resnet_base.root_resnet_arg_scope,
                       }


def get_base_network_fn(name, weight_decay=0.0):
    """Returns a base_network_fn such as 'net, end_points = base_network_fn(images)'.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      base_network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          nets, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in base_networks_map:
        raise ValueError('Name of network unknown: %s' % name)
    func = base_networks_map[name]

    @functools.wraps(func)
    def base_network_fn(images):
        arg_scope = base_arg_scopes_map[name](weight_decay=weight_decay)
        # arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images)

    if hasattr(func, 'default_image_size'):
        base_network_fn.default_image_size = func.default_image_size

    return base_network_fn
