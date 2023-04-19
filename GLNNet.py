import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.optim as optim
import model_utils
import matplotlib.pyplot as plt

class GLNNet(nn.Module):
  """
  A Linear Neural Net with one hidden layer
  """

  def __init__(self, dims, pathways, context_to_pathway_map=None, n_context=1, context_in_input=False, ctx_to_all=False):
    """
    Initialize LNNet parameters

    Args:
      connectivity: list of lists. Each element (list) within list corresponds to a layer. Each item (list) in this list corresponds to a module.
                    Each item in this list corresponds to a module in previous layer, with 1 denoting connection and 0 no connection.
      in_dim: int
        Input dimension
      out_dim: int
        Ouput dimension
      hid_dim: int
        Hidden dimension

    Returns:
      Nothing
    """
    super().__init__()

    self.dims = dims
    self.n_context = n_context
    self.layers = nn.Sequential(OrderedDict([('w' + str(l), nn.Linear(dim_in, dim_out, bias=False))
                                             for l, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:]))]))
    self.context_in_input = context_in_input
    if context_in_input:
      self.ctx_to_all = ctx_to_all
      self.context_input_layer = nn.Linear(n_context, dims[1], bias=False)

    self.input_mask = True
    # self.layers = [nn.Linear(dim_in, dim_out, bias=False) for (dim_in, dim_out) in zip(dims[:-1], dims[1:])]

    if not all(pathways):
      raise ValueError('Cannot have zero pathways in layer')
    elif len(dims) != len(pathways):
      raise ValueError('Length of pathways pattern does not match number of layers')
    elif any([d % p != 0 for d, p in zip(dims, pathways)]):
      raise ValueError('N_units in a layer is not divisible by number of pathways specified')
    else:
      self.pathways = pathways
      self.pathway_sizes = [d // p for d, p in zip(dims, pathways)]
      gateable_layers = np.where([p != 1 for p in pathways])[0].tolist()
      if len(gateable_layers) > 1:
        raise NotImplementedError('Have not implemented gating in network with multiple pathways occurring at multiple layers')
      else:
        self.gateable_layers = gateable_layers
        self.max_pathways = pathways[self.gateable_layers[0]]

    self.context_to_pathway_mapping = None
    self.set_context_to_pathway_mapping(context_to_pathway_map)


  def forward(self, x, context):
    """
    Forward pass of LNNet

    Args:
      x: torch.Tensor
        Input tensor

    Returns:
      hid: torch.Tensor
        Hidden layer activity
      out: torch.Tensor
        Output/Prediction
    """
    pathway_masks = self.get_pathway_masks(context)
    masks = self.get_neuron_masks(pathway_masks)

    # mask input vector with gating pattern
    act = masks[0] * x
    acts = [act]

    for l, (mask, layer) in enumerate(zip(masks[1:], self.layers)):
      # compute layer activations and mask
      act = layer(act)
      if l == 0 and self.context_in_input:
        mask_additional = torch.ones(self.dims[1])
        mask_additional[self.pathway_sizes[1]:] = 0
        act_additional = mask_additional * self.context_input_layer(context)
        act += act_additional
      act = mask * act
      # append activation values to list
      acts.append(act)

    return act, acts, masks


  def compute_pathway_preds(self, x, context):
    pathway_masks = self.get_pathway_masks(context)
    masks = self.get_neuron_masks(pathway_masks)

    # mask input vector with gating pattern
    init = masks[0] * x

    if len(self.gateable_layers) > 1:
      raise NotImplementedError()
    else:
      g = self.gateable_layers[0]

    n_pathways = self.pathways[g]
    preds = torch.zeros((n_pathways, x.shape[0], self.dims[-1]))

    for p in range(n_pathways):
      act = init
      isolation_pathway_masks = [pathway_mask.detach().clone() for pathway_mask in pathway_masks]
      isolation_pathway_masks[g] = torch.zeros_like(isolation_pathway_masks[g])
      isolation_pathway_masks[g][:,p] = 1.
      isolation_masks = self.get_neuron_masks(isolation_pathway_masks)

      for l, (isolation_mask, mask, layer) in enumerate(zip(isolation_masks[1:], masks[1:], self.layers)):
        # compute layer activations and mask
        act = layer(act)

        if l == 0 and self.context_in_input:
          mask_additional = torch.ones(self.dims[1])
          if not self.ctx_to_all:
            print('masking context input to non-shared pathways')
            mask_additional[self.pathway_sizes[1]:] = 0
          act_additional = mask_additional * self.context_input_layer(context)
          act += act_additional

        act = isolation_mask * mask * act
        # append activation values to list
        # acts.append(act)
      preds[p, :, :] = act

    return preds

  def get_pathway_masks(self, context):

    pathway_masks = []

    for l, p in enumerate(self.pathways):

      if l not in self.gateable_layers:
        pathway_mask = torch.ones((context.shape[0], 1), dtype=torch.float)
      else:
        # pathway_mask = context.float()
        context_idxs = torch.argmax(context, axis=1)
        pathway_mask = self.context_to_pathway_mapping[context_idxs, :]
        # if context.shape[1] != p:
        #   print(context.shape, p)
        #   raise NotImplementedError('Have not implemented case where number of contexts does not equal number of gateable pathways in layer')
      pathway_masks.append(pathway_mask)

    return pathway_masks

  def get_neuron_masks(self, pathway_masks):

    # if len(g) != len(self.pathways):
    #   raise ValueError('Gating pattern does not match number of layers')

    neuron_masks = []

    for l, (pathway_mask, pathway_size, pathways) in enumerate(zip(pathway_masks, self.pathway_sizes, self.pathways)):
      neuron_masks.append(torch.repeat_interleave(pathway_mask, pathway_size, dim=1).requires_grad_(False))
      # if (active_pathway == None):
      #   if pathways == 1:
      #     pathway = 0
      #   else:
      #     raise ValueError("Context in layer " + str(l) + " cannot be 'None' since multiple pathways exist in this layer")
      # if pathway >= pathways:
      #   raise ValueError('Context passed at layer ' + str(l) + ' exceeds number of pathways in this layer')
      # masks[l][pathway * pathway_size:(pathway + 1) * pathway_size] = 1

    return neuron_masks

  def set_context_to_pathway_mapping(self, mappings):
    if len(self.gateable_layers) > 0:
      pathways = self.pathways[self.gateable_layers[0]]

      if mappings is None:
        context_to_pathway_mapping = torch.eye(pathways)
      else:
        context_to_pathway_mapping = torch.zeros((len(mappings), pathways))

        for c, c_to_p in enumerate(mappings):
          context_to_pathway_mapping[c, c_to_p] = 1

      self.context_to_pathway_mapping = context_to_pathway_mapping

  def to_relu(self):

    layers = []
    for l, (dim_in, dim_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
      if l in self.gateable_layers:
        dim_in += self.n_context
      lin = ('w' + str(l), nn.Linear(dim_in, dim_out, bias=False))
      relu = ('relu' + str(l), nn.ReLU())
      layers.append(lin)
      layers.append(relu)

    relu_net = nn.Sequential(OrderedDict(layers))

    return relu_net

  def get_svds(self):
    if len(self.gateable_layers) > 1:
      raise NotImplementedError()

    g = self.gateable_layers[0]
    p = self.pathways[g]

    input_dims = [self.dims[0] + self.n_context if (i==0 and self.context_in_input) else self.dims[0] for i in range(p)]
    weights_pre = [torch.eye(input_dim) for input_dim in input_dims]

    for layer in self.layers[1:g-1]:
      new_weight = layer.weight.clone().detach()
      weights_pre = [new_weight @ w_pre for w_pre in weights_pre]

    weights_in = self.layers[g-1].weight.clone().detach()
    weights_out = self.layers[g].weight.clone().detach()

    weights_in_split = torch.split(weights_in, self.pathway_sizes[g], dim=0)
    weights_out_split = torch.split(weights_out, self.pathway_sizes[g], dim=1)

    if self.context_in_input:
      context_weights = self.context_input_layer.weight.clone().detach()
      context_weights_split = torch.split(context_weights, self.pathway_sizes[g], dim=0)
      # print([torch.cat((a, b), dim=1).shape for a,b in zip(context_weights_split, weights_in_split)])
      # for i, w_out, w_in in enumerate(zip(weights_out_split, weights_in_split))
      pathway_weights_gated = [w_out @ (torch.cat((w_context, w_in), dim=1) if i==0 else w_in) for i, (w_out, w_in, w_context)
                               in enumerate(zip(weights_out_split, weights_in_split, context_weights_split))]
    else:
      pathway_weights_gated = [w_out @  w_in for i, (w_out, w_in)
                               in enumerate(zip(weights_out_split, weights_in_split))]

    weights_post = [torch.eye(self.dims[g+1], dtype=torch.float) for _ in range(p)]
    for layer in self.layers[g+1:]:
      new_weight = layer.weight.clone().detach()
      weights_post = [new_weight @ w_post for w_post in weights_post]

    net_pathway_weights = [w_post @ w_pathway @ w_pre for w_post, w_pathway, w_pre in zip(weights_post, pathway_weights_gated, weights_pre)]
    svds = [torch.svd(pathway_weight) for pathway_weight in net_pathway_weights]
    U = [s[0] for s in svds]
    S = [s[1] for s in svds]
    V = [s[2] for s in svds]

    return (U, S, V), net_pathway_weights

def train(model, inputs, targets, context, n_epochs, lr, illusory_i=0, hold_out_i=None, corr=False, rsm_interval=1000, relu=False):
  in_dim = inputs.size(1)

  losses = np.zeros(n_epochs)  # Loss records
  preds = np.zeros((n_epochs, targets.shape[0], targets.shape[1]))
  #   modes = np.zeros((n_epochs, in_dim))  # Singular values (modes) records
  rs_mats = [[[] for _ in range(p)] for p in model.pathways]  # Representational similarity matrices
  pathway_preds = np.zeros((model.pathways[model.gateable_layers[0]], n_epochs//rsm_interval, targets.shape[0], targets.shape[1]))
  weights_mat = [[] for _ in range(model.max_pathways)]
  U_mat = [[] for _ in range(model.max_pathways)]
  S_mat = [[] for _ in range(model.max_pathways)]
  V_mat = [[] for _ in range(model.max_pathways)]

  #   illusions = np.zeros(n_epochs)  # Prediction for the given feature
  optimizer = optim.SGD(model.parameters(), lr=lr)
  criterion = nn.MSELoss(reduction='sum')

  if hold_out_i is not None:
    inputs_held_out = np.delete(inputs, hold_out_i, axis=0)
    targets_held_out = np.delete(targets, hold_out_i, axis=0)
    context_held_out = np.delete(context, hold_out_i, axis=0)

  for i in range(n_epochs):
    optimizer.zero_grad()

    if hold_out_i is not None:
      predictions_held_out, _, _ = model(inputs_held_out, context_held_out)
      loss = 0.5 * criterion(predictions_held_out, targets_held_out)
      predictions, hiddens, _ = model(inputs, context)

    else:
      predictions, hiddens, _ = model(inputs, context)
      loss = 0.5 * criterion(predictions, targets)

    loss.backward()
    optimizer.step()

    # Section 2 Singular value decomposition
    #     U, Σ, V = net_svd(model, in_dim)

    # Section 3 calculating representational similarity matrix
    if i % rsm_interval == 0:
      rsm = model_utils.net_rsm(hiddens, model.pathway_sizes, corr=corr)
      # rs_mats[l].append(RSM.numpy())
      rs_mats = model_utils.nested_to_nested(rs_mats, rsm)
      # print(isinstance(model, GLNNet))
      if not relu:
        pathway_pred = model.compute_pathway_preds(inputs, context).detach().numpy()
        pathway_preds[:, i // rsm_interval, :, :] = pathway_pred
        svds, weights = model.get_svds()
      else:
        comb_idxs, combs = model.get_neuron_gatings(hiddens)
        svds, weights = model.get_svds(comb_idxs)
        print([len(c) for c in comb_idxs])

      weights_mat = [weights_mat[i] + [new_w.numpy()] for i, new_w in enumerate(weights)]
      U_mat = [U_mat[i] + [new_U.numpy()] for i, new_U in enumerate(svds[0])]
      S_mat = [S_mat[i] + [new_S.numpy()] for i, new_S in enumerate(svds[1])]
      V_mat = [V_mat[i] + [new_V.numpy()] for i, new_V in enumerate(svds[2])]

        # fig, axes = plt.subplots(1, len(weights))
        # fig.suptitle(i)
        # for ax, weight in zip(axes, weights):
        #   im=ax.imshow(weight)
        #   plt.colorbar(im)
        #   ax.xaxis.set_visible(False)
        #   ax.yaxis.set_visible(False)

    # Logging (recordings)
    losses[i] = loss.item()
    preds[i, :, :] = predictions.detach().numpy()
    #     modes[i] = Σ.detach().numpy()
    #     illusions[i] = pred_ij.numpy()
  weights_mat = [np.array(w_timeseries) for w_timeseries in weights_mat]
  U_mat = [np.array(U_timeseries) for U_timeseries in U_mat]
  S_mat = [np.array(S_timeseries) for S_timeseries in S_mat]
  V_mat = [np.array(V_timeseries) for V_timeseries in V_mat]

  rs_mats_numpy = model_utils.nested_to_numpy(rs_mats)

  return losses, preds, rs_mats_numpy, pathway_preds, (U_mat, S_mat, V_mat), weights_mat  # , modes, np.array(rs_mats), illusions
