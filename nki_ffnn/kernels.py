import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

@nki.jit
def nki_transpose(in_tensor):
    """NKI kernel to transpose a 2D tensor.

    Args:
        in_tensor: an input tensor of shape [#rows, #cols]

    Returns:
        out_tensor: an output (transposed) tensor of shape [#cols, #rows]
    """
    i_rows, i_cols = in_tensor.shape
    o_rows, o_cols = i_cols, i_rows

    out_tensor = nl.ndarray((o_rows, o_cols), dtype=in_tensor.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    sz_p = nl.tile_size.pmax

    n_tiles_rows = i_rows // sz_p
    n_tiles_cols = i_cols // sz_p

    rem_rows = i_rows % sz_p
    rem_cols = i_cols % sz_p

    # Full tiles
    for i_tile_col in nl.affine_range(n_tiles_cols):
        for i_tile_row in nl.affine_range(n_tiles_rows):
            in_tile = nl.ndarray((sz_p, sz_p), dtype=in_tensor.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=in_tile, src=in_tensor[nl.ds(i_tile_row*sz_p, sz_p), nl.ds(i_tile_col*sz_p,sz_p)])
            out_tile = nl.ndarray((sz_p, sz_p), dtype=in_tensor.dtype, buffer=nl.sbuf)
            out_tile = nisa.nc_transpose(data=in_tile)
            nisa.dma_copy(dst=out_tensor[nl.ds(i_tile_col*sz_p,sz_p), nl.ds(i_tile_row*sz_p,sz_p)], src=out_tile)

    return out_tensor

@nki.jit
def nki_bias_add_act(A, b, act='relu'):
    """NKI kernel to add a bias vector to each row of a 2D tensor, and apply activation.

    Args:
        A: an input tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
        b: a bias vector of shape [1, HIDDEN_SIZE]
        act: an activation function to apply (e.g., 'relu', 'softmax')
    Returns:
        result: the resulting output tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
    """
    # Gather input shapes
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"

    # Create an output tensor
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    sz_p = nl.tile_size.pmax
    n_tiles_rows = BATCH_SIZE // sz_p
    n_tiles_cols = HIDDEN_SIZE // sz_p

    b_sbuf = nl.ndarray((1, HIDDEN_SIZE), dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sbuf, src=b)
    for i_tile_row in nl.affine_range(n_tiles_rows):
        A_tile = nl.ndarray((sz_p, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=A_tile, src=A[nl.ds(i_tile_row*sz_p, sz_p), :])
        result_tile = nl.ndarray((sz_p, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.sbuf)
        result_tile = nl.add(A_tile, b_sbuf)
        if act == 'relu':
            result_tile = nl.maximum(0, result_tile)
        elif act == 'softmax':
            x_max = nl.max(result_tile, axis=1, keepdims=True)
            x_stable = nl.subtract(result_tile, x_max)
            e_x = nl.exp(x_stable)
            e_x_sum = nl.sum(e_x, axis=1, keepdims=True)
            result_tile = nl.divide(e_x, e_x_sum)
        nisa.dma_copy(dst=result[nl.ds(i_tile_row*sz_p, sz_p), :], src=result_tile)
    return result

@nki.jit
def nki_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel to compute the forward pass of the feedforward neural network with 1 hidden layer.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      probs: the resulting probability output tensor of shape [BATCH_SIZE, OUTPUT_SIZE]
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'
  """
  if matmul_kernel == 'tiled':
    nki_matmul = nki_matmul_tiled_
  elif matmul_kernel == 'hoist_load':
    nki_matmul = nki_matmul_hoist_load_
  elif matmul_kernel == 'block_free_dimension':
    nki_matmul = nki_matmul_block_free_dimension_
  elif matmul_kernel == 'fully_optimized':
    nki_matmul = nki_matmul_fully_optimized_
  else:
    raise ValueError(f"Unsupported matmul kernel: {matmul_kernel}")

  # Layer 1
  # YOUR CODE HERE  
  intermed_result = nki_matmul(nki_transpose(X), W1)
  intermed_result = nki_bias_add_act(intermed_result, b1, 'relu')
  # Layer 2 (output)
  # YOUR CODE HERE
  probs = nki_matmul(nki_transpose(intermed_result), W2)
  probs = nki_bias_add_act(probs, b2, 'softmax')

  return probs


@nki.jit
def nki_predict(
    X,
    W1,
    b1,
    W2,
    b2, matmul_kernel='tiled'
):
    """NKI kernel run forward pass and predict the classes of the input tensor.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'

  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  """
    probs = nki_forward(X, W1, b1, W2, b2, matmul_kernel)
    BATCH_SIZE, OUTPUT_SIZE = probs.shape
    predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)
    sz_p = nl.tile_size.pmax
    for k in nl.affine_range(BATCH_SIZE//sz_p):
        maxes = nl.ndarray((sz_p, 8), dtype=probs.dtype, buffer=nl.hbm)
        for i in nl.affine_range(sz_p):
            temp = nl.load(probs[k*sz_p + i : k*sz_p + i + 1, :])
            m = nl.ndarray((1, 8), dtype=probs.dtype, buffer=nl.sbuf)
            m = nisa.max8(src=temp)
            nisa.dma_copy(dst=maxes[nl.ds(i, 1), :], src=m)
        for j in nl.affine_range(sz_p//8):
            #temp = nl.ndarray((8, OUTPUT_SIZE), dtype=probs.dtype, buffer=nl.sbuf)
            temp = nl.load(probs[k*sz_p + j*8 : k*sz_p + j*8 + 8, :])
            max_tile = nl.load(maxes[j*8: j*8 + 8, :])
            indices = nisa.nc_find_index8(data=temp, vals=max_tile)
            nl.store(predictions[k*sz_p + j*8 : k*sz_p + j*8 + 8], indices[:, 0])
            #nisa.dma_copy(dst=predictions[nl.ds(k*sz_p + j*8)], src=indices[0])
    return predictions
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
