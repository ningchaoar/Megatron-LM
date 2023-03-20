import torch
import torch.nn as nn
import torch.nn.functional as F


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


class SparseBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_scores, sparsity):
        num_prune = int(mask_scores.numel() * sparsity)
        prune_indices = torch.argsort(mask_scores.reshape(-1))[:num_prune]
        mask = mask_scores.clone().fill_(1)
        mask.reshape(-1)[prune_indices] = 0.0
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

class SparseLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias = True,
        pruning_method = "pst",
        weight_rank = 8,
        weight_beta = 1.0,
        mask_rank = 8,
        mask_alpha1 = 1.0,
        mask_alpha2 = 1.0,
        block_size = 1,
        kernel_size = 1,
        stride= 1,
        skip_bias_add=False
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.pruning_method = pruning_method
        
        self.weight_rank = weight_rank
        self.weight_beta = weight_beta
        self.mask_rank = mask_rank
        self.mask_alpha1 = mask_alpha1
        self.mask_alpha2 = mask_alpha2

        self.cur_sparsity = 0.0
        self.skip_bias_add = skip_bias_add

        if self.pruning_method == "pst":
            # create trainable params
            self.weight_U = nn.Parameter(torch.randn(out_features, self.weight_rank, device=torch.cuda.current_device(), dtype=torch.half))
            set_tensor_model_parallel_attributes(self.weight_U, True, 0, 1)

            self.weight_V = nn.Parameter(torch.zeros(self.weight_rank, in_features, device=torch.cuda.current_device(), dtype=torch.half))
            set_tensor_model_parallel_attributes(self.weight_V, True, 0, 1)

            self.mask_scores_A = nn.Parameter(torch.randn(out_features, self.mask_rank, device=torch.cuda.current_device(), dtype=torch.half))
            set_tensor_model_parallel_attributes(self.mask_scores_A, True, 0, 1)

            self.mask_scores_B = nn.Parameter(torch.zeros(self.mask_rank, in_features, device=torch.cuda.current_device(), dtype=torch.half))
            set_tensor_model_parallel_attributes(self.mask_scores_B, True, 0, 1)

            self.mask_scores_R = nn.Parameter(torch.zeros(out_features, device=torch.cuda.current_device(), dtype=torch.half))
            set_tensor_model_parallel_attributes(self.mask_scores_R, True, 0, 1)

            self.mask_scores_C = nn.Parameter(torch.zeros(in_features, device=torch.cuda.current_device(), dtype=torch.half))
            set_tensor_model_parallel_attributes(self.mask_scores_C, True, 0, 1)

            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        # By Angel, test block
        self.block_size = block_size
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, stride=stride, bias=False, device=torch.cuda.current_device(), dtype=torch.half)
        set_tensor_model_parallel_attributes(self.conv.weight, True, 0, 1)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        #self.pooling = nn.MaxPool2d(kernel_size = block_size)
        self.unsampling = nn.UpsamplingNearest2d(scale_factor = self.block_size)
        # Done

    def forward(self, inputs):
        if self.pruning_method == "pst":
            weight = self.weight + self.weight_beta * self.weight_U @ self.weight_V
            mask_scores = weight.abs() + self.mask_alpha1 * self.mask_scores_A @ self.mask_scores_B + \
             self.mask_alpha2 * (self.mask_scores_R.unsqueeze(1) + self.mask_scores_C.unsqueeze(0))

            # By Angel, do pooling
            if self.block_size > 1:
                old_size = mask_scores.size()
                new_size = [1]*(4 - len(old_size)) + list(old_size)
                mask_scores = self.conv(mask_scores.reshape(new_size))
                mask_scores = torch.max(mask_scores, dim = 1, keepdim = True)[0]
                mask_scores = self.pooling(mask_scores)
                #mask_scores = self.pooling(mask_scores.reshape(new_size))
                mask_scores = mask_scores.reshape([dim//self.block_size for dim in new_size[4 - len(old_size):]])
            # Done
            mask = SparseBinarizer.apply(mask_scores, self.cur_sparsity)
            # By Angel, do unsampling
            if self.block_size > 1:
                old_size = mask.size()
                new_size = [1]*(4 - len(old_size)) + list(old_size)
                mask = self.unsampling(mask.reshape(new_size))
                mask = mask.reshape(list(mask.size()[4 - len(old_size):]))
            # Done

            masked_weight = mask * weight
            if self.bias is None or self.skip_bias_add:
                output = F.linear(inputs, masked_weight)
            else:
                output =  F.linear(inputs, masked_weight, self.bias)
            return output, self.bias
        else:
            if self.bias is None or self.skip_bias_add:
                output = F.linear(inputs, self.weight)
            else:
                output =  F.linear(inputs, self.weight, self.bias)
            return output, self.bias
    
    def convert(self):
        if self.pruning_method == "pst":
            weight = self.weight + self.weight_beta * self.weight_U @ self.weight_V
            mask_scores = weight.abs() + self.mask_alpha1 * self.mask_scores_A @ self.mask_scores_B + \
             self.mask_alpha2 * (self.mask_scores_R.unsqueeze(1) + self.mask_scores_C.unsqueeze(0))

            # By Angel, do pooling
            if self.block_size > 1:
                old_size = mask_scores.size()
                new_size = [1]*(4 - len(old_size)) + list(old_size)
                mask_scores = self.conv(mask_scores.reshape(new_size))
                mask_scores = torch.max(mask_scores, dim = 1, keepdim = True)[0]
                mask_scores = self.pooling(mask_scores)
                #mask_scores = self.pooling(mask_scores.reshape(new_size))
                mask_scores = mask_scores.reshape([dim//self.block_size for dim in new_size[4 - len(old_size):]])
            # Done
            mask = SparseBinarizer.apply(mask_scores, self.cur_sparsity)
            # By Angel, do unsampling
            if self.block_size > 1:
                old_size = mask.size()
                new_size = [1]*(4 - len(old_size)) + list(old_size)
                mask = self.unsampling(mask.reshape(new_size))
                mask = mask.reshape(list(mask.size()[4 - len(old_size):]))
            # Done

            masked_weight = mask * weight

            self.old_weight = self.weight.data.clone()
            self.weight.data = masked_weight.data
    
    def restore(self):
        if self.pruning_method == "pst":
            self.weight.data = self.old_weight
            del self.old_weight
