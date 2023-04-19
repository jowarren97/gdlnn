import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.optim as optim
import model_utils
import itertools as it

class ReluNNet(nn.Module):
    """
    A Linear Neural Net with one hidden layer
    """

    def __init__(self, dims, n_context=1, gateable_layers=[], final_relu=False):
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
        self.gateable_layers = gateable_layers
        self.pathways = tuple([1 for _ in dims])
        self.max_pathways = len(list(it.chain.from_iterable([comb for comb in it.combinations(np.arange(n_context), c)]
                                                            for c in np.arange(self.n_context + 1, 0, -1))))
        print(self.max_pathways)
        self.pathway_sizes = list(dims)
        self.final_relu = final_relu

        layers = []
        for l, (dim_in, dim_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            if (l + 1) in self.gateable_layers:
                dim_in += self.n_context
            lin = ('w' + str(l), nn.Linear(dim_in, dim_out, bias=False))
            layers.append(lin)

            if dim_out != self.dims[-1] or final_relu is True:
                if (l + 1) in self.gateable_layers:
                    activation = ('relu' + str(l), nn.ReLU())
                else:
                    activation = ('eye' + str(l), nn.Identity())

                layers.append(activation)

        self.layers = nn.Sequential(OrderedDict(layers))

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

        act = x
        acts = [act]

        for l, layer in enumerate(self.layers):
            if (l/2 + 1) in self.gateable_layers:
                act = torch.cat((context, act), dim=1)
            act = layer(act)
            # append activation values to list
            if type(layer) != nn.Linear:
                acts.append(act)

        if not self.final_relu:
            acts.append(act)

        return act, acts, None


    def get_svds(self, comb_idxs=None):
        if len(self.gateable_layers) > 1:
            raise NotImplementedError()
        lin_layers = [l for l in self.layers if isinstance(l, nn.Linear)]
        net_weight = torch.eye(self.dims[0] + self.n_context)

        if comb_idxs is None:
            for layer in lin_layers:
                net_weight = layer.weight.clone().detach() @ net_weight

            svd = torch.svd(net_weight)
            U = [svd[0]]
            S = [svd[1]]
            V = [svd[2]]
            net_pathway_weights = [net_weight]
        else:
            w_in = lin_layers[0].weight.clone().detach()
            w_out = lin_layers[1].weight.clone().detach()
            net_pathway_weights = [w_out[:, idxs] @ w_in[idxs, :] for idxs in comb_idxs]
            svds = [torch.svd(w) for w in net_pathway_weights]
            U = [s[0] for s in svds]
            S = [s[1] for s in svds]
            V = [s[2] for s in svds]

        return (U, S, V), net_pathway_weights


    def get_neuron_gatings(self, acts, tol=0.001):
        """
        Partition relu hidden neurons based on which task contexts they are active for

        Args:
          self: nn.Module
            The relu network
          acts: torch.Tensor
            The hidden layer activities
          tol: float
            The activity tolerance level for classifying a neuron as 'off'

        Returns:
          comb_idxs: list
            i-th element contains list of hidden neuron ids active in i-th context permutation
          all_combs: list
            all possible context permutations

        """
        # get gateable layers in relu net (there will only be one)
        g = self.gateable_layers[0]
        # detach from graph
        hid = acts[g].detach()
        # get maximum activation of hidden neuron activities
        maxes = torch.max(hid, axis=0)[0]
        # split hidden neurons activity tensor into the separate contexts (dim 0 is batch size)
        hid_split = torch.split(hid, hid.shape[0] // self.n_context, dim=0)
        # sum activities for each context partition, and check if above a threshold (if above, neuron is active in that context)
        hid_sum = [torch.sum(h, dim=0, keepdim=True) >= tol * maxes for h in hid_split]
        # concatenate again; now tensor is shape n_contexts x n_neurons, where each element is binary and denotes if neuron active in that context
        hid = torch.cat(hid_sum, dim=0)

        mainlist = np.arange(self.n_context)
        # get all permutations of combinations of the contexts
        all_combs = [[comb for comb in it.combinations(mainlist, c)] for c in np.arange(self.n_context + 1, 0, -1)]
        all_combs = list(it.chain.from_iterable(all_combs))

        comb_idxs = []
        # iterate through all combinations of contexts
        for comb in all_combs:
            # get on/off hidden neuron tuning only for the contexts in this specific combination
            hid_comb = hid[comb, :]
            # get contexts not in specific combination
            comb_not = tuple(set(np.arange(self.n_context)) - set(comb))
            # get on/off hidden neuron tuning only for the contexts NOT in this specific combination
            hid_comb_not = hid[comb_not, :]
            # get neurons active only in all contexts in this combination
            active_in_all = torch.sum(hid_comb, dim=0) == len(comb)
            # get neurons inactive in all contexts not in this combination
            inactive_in_other = torch.sum(hid_comb_not, dim=0) == 0
            # logical and to find neurons active only in the contexts of this combination, and inactive in other
            active_in_comb_only = torch.logical_and(active_in_all, inactive_in_other)
            # get indices of neurons
            idxs = torch.where(active_in_comb_only)[0]
            comb_idxs.append(idxs)

        return comb_idxs, all_combs


    # def get_neuron_gatings(self, acts, tol=0.001):
    #     """
    #     Partition relu hidden neurons based on which task contexts they contribute to
    #
    #     Args:
    #       self: nn.Module
    #         The relu network
    #       acts: torch.Tensor
    #         The hidden layer activities
    #       tol: float
    #         The tolerance level for classifying a neuron as 'off'
    #
    #     Returns:
    #       comb_idxs: list
    #         i-th element contains list of hidden neuron ids active in i-th context permutation
    #       all_combs: list
    #         all possible context permutations
    #
    #     """
    #     # get gateable layers in relu net (there will only be one)
    #     g = self.gateable_layers[0]
    #     # detach from graph
    #     hid = acts[g].detach()
    #     # get readout weights
    #     readout_weights = self.layers[-1].weight.data
    #     # compute neuron contributions to output
    #     # neuron_contribs = torch.matmul(hid, readout_weights.T)
    #     neuron_contribs = torch.einsum('ij,kj->ijk', hid, readout_weights)
    #     neuron_contribs_l1 = torch.norm(neuron_contribs, p=1, dim=-1, keepdim=False)
    #     maxes = torch.max(neuron_contribs_l1, axis=-1)[0]
    #     neuron_contribs_bin = neuron_contribs_l1 > tol * maxes
    #     print(maxes)
    #     import matplotlib.pyplot as plt
    #     plt.imshow(neuron_contribs_l1.detach().numpy())
    #     plt.show()
    #     # # get maximum contribution of hidden neuron activities
    #     # maxes = torch.max(neuron_contribs, axis=0)[0]
    #     # split hidden neurons contribution tensor into the separate contexts (dim 0 is batch size)
    #     neuron_contribs_split = torch.split(neuron_contribs, neuron_contribs.shape[0] // self.n_context, dim=0)
    #     # sum contributions for each context partition, and check if above a threshold (if above, neuron is active in that context)
    #     neuron_contribs_sum = [torch.sum(c, dim=0, keepdim=True) for c in neuron_contribs_split]
    #     # concatenate again; now tensor is shape n_contexts x n_neurons, where each element is binary and denotes if neuron active in that context
    #     neuron_contribs = torch.cat(neuron_contribs_sum, dim=0)
    #
    #     mainlist = np.arange(self.n_context)
    #     # get all permutations of combinations of the contexts
    #     all_combs = [[comb for comb in it.combinations(mainlist, c)] for c in np.arange(self.n_context + 1, 0, -1)]
    #     all_combs = list(it.chain.from_iterable(all_combs))
    #
    #     comb_idxs = []
    #     # iterate through all combinations of contexts
    #     for comb in all_combs:
    #         # get on/off hidden neuron tuning only for the contexts in this specific combination
    #         contribs_comb = neuron_contribs[comb, :]
    #         # get contexts not in specific combination
    #         comb_not = tuple(set(np.arange(self.n_context)) - set(comb))
    #         # get on/off hidden neuron tuning only for the contexts NOT in this specific combination
    #         contribs_comb_not = neuron_contribs[comb_not, :]
    #         # get neurons active only in all contexts in this combination
    #         active_in_all = torch.sum(contribs_comb, dim=0) == len(comb)
    #         # get neurons inactive in all contexts not in this combination
    #         inactive_in_other = torch.sum(contribs_comb_not, dim=0) == 0
    #         # logical and to find neurons active only in the contexts of this combination, and inactive in other
    #         active_in_comb_only = torch.logical_and(active_in_all, inactive_in_other)
    #         # get indices of neurons
    #         idxs = torch.where(active_in_comb_only)[0]
    #         comb_idxs.append(idxs)
    #
    #     return comb_idxs, all_combs


