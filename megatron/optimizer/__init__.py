# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from apex.optimizers import FusedLAMB as LAMB

from megatron import get_args
from megatron.model import LayerNorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer


def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                     if p is not None])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n == 'bias'])
    return weight_decay_params, no_weight_decay_params


def _get_params_for_sparse_model(modules):
    """Divide params into: 
        1. dense params with wd.
        2. dense params without wd.
        3. pst params with wd.
        4. pst params without wd.
    """
    dense_wd_params = {'params': []}
    dense_no_wd_params = {'params': [], 'weight_decay': 0.0}

    pst_name = set(['weight_U', 'weight_V', 'mask_scores_A', 'mask_scores_B', 'mask_scores_R', 'mask_scores_C'])
    pst_wd_params = {'params': []}
    pst_no_wd_params = {'params': [], 'weight_decay': 0.0}
    
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                dense_no_wd_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                     if p is not None])
            else:
                for name, param in module_._parameters.items():
                    if param is None:
                        continue
                    if name in pst_name:
                        pst_wd_params['params'].append(param)
                    elif len(param.shape) == 4:  # conv.weight
                        pst_wd_params['params'].append(param)
                    elif param.shape[0] == 4:  # conv.bias
                        pst_no_wd_params['params'].append(param)
                    elif name == 'bias':
                        dense_no_wd_params['params'].append(param)
                    else:
                        dense_wd_params['params'].append(param)

    return dense_wd_params, dense_no_wd_params, pst_wd_params, pst_no_wd_params


def get_megatron_optimizer(model):
    args = get_args()

    # Base optimizer.
    if args.enable_sparse_mode:
        param_groups = _get_params_for_sparse_model(model)
    else:
        param_groups = _get_params_for_weight_decay_optimization(model)
    if args.optimizer == 'adam':
        optimizer = Adam(param_groups,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.adam_beta1, args.adam_beta2),
                         eps=args.adam_eps)
    elif args.optimizer == 'sgd':
        optimizer = SGD(param_groups,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        momentum=args.sgd_momentum)
    elif args.optimizer == 'lamb':
        optimizer = LAMB(param_groups,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.lamb_beta1, args.lamb_beta2),
                         eps=args.lamb_eps,
                         bias_correction=True)
    else:
        raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    if args.fp16 or args.bf16:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        return Float16OptimizerWithFloat16Params(optimizer,
                                                 args.clip_grad,
                                                 args.log_num_zeros_in_grad,
                                                 params_have_main_grad,
                                                 args.use_contiguous_buffers_in_local_ddp,
                                                 args.bf16,
                                                 grad_scaler)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad,
                         args.use_contiguous_buffers_in_local_ddp)
