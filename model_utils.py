#@title Set random seed
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import data_utils
import math
import itertools as it

#@markdown Executing `set_seed(seed=seed)` you are setting the seed

# For DL its critical to set the random seed so that students can have a
# baseline to compare their results to expected results.
# Read more here: https://pytorch.org/docs/stable/notes/randomness.html

# Call `set_seed` function in the exercises to ensure reproducibility.

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  # print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


#@title Set device (GPU or CPU). Execute `set_device()`
# especially if torch modules used.

# Inform the user if the notebook uses GPU or CPU.

def set_device():
  """
  Set the device. CUDA if available, CPU otherwise

  Args:
    None

  Returns:
    Nothing
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("GPU is not enabled in this notebook. \n"
          "If you want to enable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `GPU` from the dropdown menu")
  else:
    print("GPU is enabled in this notebook. \n"
          "If you want to disable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `None` from the dropdown menu")

  return device


def ex_initializer_(model, gamma=1e-12, trunc=False):
  """
  In-place Re-initialization of weights

  Args:
    model: torch.nn.Module
      PyTorch neural net model
    gamma: float
      Initialization scale

  Returns:
    Nothing
  """
  for weight in model.parameters():
    n_out, n_in = weight.shape
    sigma = gamma / math.sqrt(n_in + n_out)
    if not trunc:
      nn.init.normal_(weight, mean=0.0, std=sigma)
    else:
      nn.init.trunc_normal_(weight, mean=0.0, std=sigma, a=0.0, b=10.0)


def train(model, inputs, targets, n_epochs, lr, illusory_i=0):
  """
  Training function

  Args:
    model: torch nn.Module
      The neural network
    inputs: torch.Tensor
      Features (input) with shape `[batch_size, input_dim]`
    targets: torch.Tensor
      Targets (labels) with shape `[batch_size, output_dim]`
    n_epochs: int
      Number of training epochs (iterations)
    lr: float
      Learning rate
    illusory_i: int
      Index of illusory feature

  Returns:
    losses: np.ndarray
      Record (evolution) of training loss
    modes: np.ndarray
      Record (evolution) of singular values (dynamic modes)
    rs_mats: np.ndarray
      Record (evolution) of representational similarity matrices
    illusions: np.ndarray
      Record of network prediction for the last feature
  """
  in_dim = inputs.size(1)

  losses = np.zeros(n_epochs)  # Loss records
  modes = np.zeros((n_epochs, in_dim))  # Singular values (modes) records
  rs_mats = []  # Representational similarity matrices
  illusions = np.zeros(n_epochs)  # Prediction for the given feature

  optimizer = optim.SGD(model.parameters(), lr=lr)
  criterion = nn.MSELoss()

  for i in range(n_epochs):
    optimizer.zero_grad()
    predictions, hiddens = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    # Section 2 Singular value decomposition
    U, Σ, V = net_svd(model, in_dim)

    # Section 3 calculating representational similarity matrix
    RSM = net_rsm(hiddens.detach())

    # Section 4 network prediction of illusory_i inputs for the last feature
    pred_ij = predictions.detach()[illusory_i, -1]

    # Logging (recordings)
    losses[i] = loss.item()
    modes[i] = Σ.detach().numpy()
    rs_mats.append(RSM.numpy())
    illusions[i] = pred_ij.numpy()

  return losses, modes, np.array(rs_mats), illusions


def net_svd(model, in_dim):
  """
  Performs a Singular Value Decomposition on
  given model weights

  Args:
    model: torch.nn.Module
      Neural network model
    in_dim: int
      The input dimension of the model

  Returns:
    U: torch.tensor
      Orthogonal Matrix
    Σ: torch.tensor
      Diagonal Matrix
    V: torch.tensor
      Orthogonal Matrix
  """
  W_tot = torch.eye(in_dim)
  for weight in model.parameters():
    W_tot = weight.detach() @ W_tot
  U, SIGMA, V = torch.svd(W_tot)
  return U, SIGMA, V


def net_rsm(hidden_activities, pathway_sizes, corr=True):
  """
  Calculates the Representational Similarity Matrix

  Args:
    h: torch.Tensor
      Activity of a hidden layer

  Returns:
    rsm: torch.Tensor
      Representational Similarity Matrix
  """
  rs_mats = []

  if len(pathway_sizes) != len(hidden_activities):
    raise ValueError('length of hidden activities does not match length of pathway sizes; pathways: ' + str(len(pathway_sizes)) + ', hiddens: ' + str(len(hidden_activities)))

  for l, (pathway_size, hid) in enumerate(zip(pathway_sizes, hidden_activities)):
    layer_rsm = []

    hid = hid.detach()
    hs = torch.split(hid, pathway_size, dim=1)

    for p, h in enumerate(hs):
      if corr:
        rsm = torch.corrcoef(h)
      else:
        rsm = h @ h.T
      rsm_np = rsm.detach().numpy()
      # print(rsm.shape)
      # rs_mats[l][p].append(rsm.numpy())
      layer_rsm.append(rsm_np)
      # rs_mats[l][p].append('(' + str(l) + ',' + str(p) + ')')
      # print(rs_mats)

    rs_mats.append(layer_rsm)

  return rs_mats


def initializer_(model, gamma=1e-12, trunc=False):
  """
  In-place Re-initialization of weights

  Args:
    model: torch.nn.Module
      PyTorch neural net model
    gamma: float
      Initialization scale

  Returns:
    Nothing
  """
  for weight in model.parameters():
    n_out, n_in = weight.shape
    sigma = gamma / math.sqrt(n_in + n_out)
    if not trunc:
      nn.init.normal_(weight, mean=0.0, std=sigma)
    else:
      nn.init.trunc_normal_(weight, mean=0.0, std=sigma, a=0.0, b=10000.0)


def nested_to_nested(list1, list2):

  if all([type(l) == list for l in (list1 + list2)]):
    new_list = [nested_to_nested(list1[i], list2[i]) for i in range(len(list1)) if type(list1[i]) is list]
  else:
    new_list = [l1 + [l2] for l1, l2 in zip(list1, list2)]

  return new_list


def nested_to_numpy(list1):
  if all([type(l) != list for l in list1]):
    new = np.array(list1)
  else:
    new = [nested_to_numpy(list1[i]) for i in range(len(list1))]
  return new


def get_mappings(dim_context):
  n_pathways = 0
  mappings = [[] for _ in range(dim_context)]
  j = 0
  for c in np.arange(dim_context, 0, -1):
    ncr = math.comb(dim_context, c)
    n_pathways += ncr
    a = [comb for comb in it.combinations(np.arange(dim_context), c)]
    for i, ctxs in enumerate(a):
      for ctx in ctxs:
        mappings[ctx].append(i + j)
    j += len(a)

  return mappings, n_pathways