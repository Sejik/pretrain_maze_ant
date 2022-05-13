
import torch
import torch.nn as nn
from torch.autograd import Function


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vector_quantization(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous().view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vector_quantization = VectorQuantization.apply
vector_quantization_st = VectorQuantizationStraightThrough.apply
__all__ = [vector_quantization, vector_quantization_st]


class VQEmbedding(nn.Module):
    """
    Vector Quantization module for VQ-VAE (van der Oord et al., https://arxiv.org/abs/1711.00937)
    This module is compatible with 1D latents only (i.e. with inputs of shape [batch_size, embedding_dim]).
    Adapted from https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py#L70
    Variable names follow those in the paper:
        z_e_x: z_e(x), i.e. the *continuous* encoding emitted by the encoder
        z_q_x: z_q(x), i.e. the decoder input -- the vector-quantized version of z_e(x)  [Eq. 2]
    """
    def __init__(self, codebook_size, code_size, beta):
        """
        :param codebook_size: number of codes in the codebook
        :param code_size: dimensionality of each code
        :param beta: weight for the commitment loss
        """
        super().__init__()

        self.codebook_size = int(codebook_size)
        self.code_size = int(code_size)
        self.beta = float(beta)

        self.embedding = nn.Embedding(self.codebook_size, self.code_size)
        self.embedding.weight.data.uniform_(-1./self.codebook_size, 1./self.codebook_size)

        self.mse_loss = nn.MSELoss(reduction='none')

    def quantize(self, z_e_x):
        return vector_quantization(z_e_x, self.embedding.weight)

    def straight_through(self, z_e_x):
        # Quantized vectors (inputs for the decoder)
        z_q_x, indices = vector_quantization_st(z_e_x, self.embedding.weight.detach())
        # Selected codes from the codebook (for the VQ objective)
        selected_codes = torch.index_select(self.embedding.weight, dim=0, index=indices)
        return z_q_x, selected_codes

    def forward(self, z_e_x, selected_codes=None):
        """
        Compute second and third loss terms in Eq. 3 in the paper
        :param z_e_x: encoder output
        :param selected_codes: (optional) second output from straight_through(); avoids recomputing it
        :return: loss = vq_loss + beta * commitment_loss
        """
        # Recompute z_q(x) if needed
        if selected_codes is None:
            _, selected_codes = self.straight_through(z_e_x)
        # Push VQ codes towards the output of the encoder
        vq_loss = self.mse_loss(selected_codes, z_e_x.detach()).sum(dim=1)
        # Encourage the encoder to commit to a code
        commitment_loss = self.mse_loss(z_e_x, selected_codes.detach()).sum(dim=1)
        # The scale of the commitment loss is controlled with beta [Eq. 3]
        loss = vq_loss + self.beta * commitment_loss
        return loss

    def compute_distances(self, inputs):
        with torch.no_grad():
            embedding_size = self.embedding.weight.size(1)
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(self.embedding.weight ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, self.embedding.weight.t(),
                                    alpha=-2.0, beta=1.0)

            return distances