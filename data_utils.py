# @title Helper functions
import torch
import numpy as np
import math
import torch.nn as nn
import LNNet
import itertools as it
import matplotlib.pyplot as plt

# def build_tree(n_levels, n_branches, probability,
#                to_np_array=True):
#   """
#   Builds tree
#
#   Args:
#     n_levels: int
#       Number of levels in tree
#     n_branches: int
#       Number of branches in tree
#     probability: float
#       Flipping probability
#     to_np_array: boolean
#       If true, represent tree as np.ndarray
#
#   Returns:
#     tree: dict if to_np_array=False
#           np.ndarray otherwise
#       Tree
#   """
#   assert 0.0 <= probability <= 1.0
#
#   tree = {}
#
#   tree["level"] = [0]
#   for i in range(1, n_levels+1):
#     tree["level"].extend([i]*(n_branches**i))
#
#   tree["pflip"] = [probability]*len(tree["level"])
#
#   tree["parent"] = [None]
#   k = len(tree["level"])-1
#   for j in range(k//n_branches):
#     tree["parent"].extend([j]*n_branches)
#
#   if to_np_array:
#     tree["level"] = np.array(tree["level"])
#     tree["pflip"] = np.array(tree["pflip"])
#     tree["parent"] = np.array(tree["parent"])
#
#   return tree
#
#
# def sample_from_tree(tree, n):
#   """
#   Generates n samples from a tree
#
#   Args:
#     tree: np.ndarray/dictionary
#       Tree
#     n: int
#       Number of levels in tree
#
#   Returns:
#     x: np.ndarray
#       Sample from tree
#   """
#   items = [i for i, v in enumerate(tree["level"]) if v == max(tree["level"])]
#   n_items = len(items)
#   x = np.zeros(shape=(n, n_items))
#   rand_temp = np.random.rand(n, len(tree["pflip"]))
#   flip_temp = np.repeat(tree["pflip"].reshape(1, -1), n, 0)
#   samp = (rand_temp > flip_temp) * 2 - 1
#
#   for i in range(n_items):
#     j = items[i]
#     prop = samp[:, j]
#     while tree["parent"][j] is not None:
#       j = tree["parent"][j]
#       prop = prop * samp[:, j]
#     x[:, i] = prop.T
#   return x
#
#
# def generate_hsd():
#   """
#   Building the tree
#
#   Args:
#     None
#
#   Returns:
#     tree_labels: np.ndarray
#       Tree Labels
#     tree_features: np.ndarray
#       Sample from tree
#   """
#   n_branches = 2  # 2 branches at each node
#   probability = .15  # flipping probability
#   n_levels = 3  # number of levels (depth of tree)
#   tree = build_tree(n_levels, n_branches, probability, to_np_array=True)
#   tree["pflip"][0] = 0.5
#   n_samples = 10000 # Sample this many features
#
#   tree_labels = np.eye(n_branches**n_levels)
#   tree_features = sample_from_tree(tree, n_samples).T
#   return tree_labels, tree_features
#
#
# def linear_regression(X, Y):
#   """
#   Analytical Linear regression
#
#   Args:
#     X: np.ndarray
#       Input features
#     Y: np.ndarray
#       Targets
#
#   Returns:
#     W: np.ndarray
#       Analytical solution
#       W = Y @ X.T @ np.linalg.inv(X @ X.T)
#   """
#   assert isinstance(X, np.ndarray)
#   assert isinstance(Y, np.ndarray)
#   M, Dx = X.shape
#   N, Dy = Y.shape
#   assert Dx == Dy
#   W = Y @ X.T @ np.linalg.inv(X @ X.T)
#   return W

#
# def add_feature(existing_features, new_feature):
#   """
#   Adding new features to existing tree
#
#   Args:
#     existing_features: np.ndarray
#       List of features already present in the tree
#     new_feature: list
#       List of new features to be added
#
#   Returns:
#     New features augmented with existing features
#   """
#   assert isinstance(existing_features, np.ndarray)
#   assert isinstance(new_feature, list)
#   new_feature = np.array([new_feature]).T
#   return np.hstack((tree_features, new_feature))


def hsd(n_items):
  power = int(math.log(n_items, 2))
  assert power % 1 == 0
  n_features = 2 ** (power + 1) - 1

  tree_features = np.zeros((n_items, n_features))
  col = 0

  freqs = np.logspace(0, power, power + 1, base=2, dtype=np.int64)

  for freq in freqs:
    span = int(n_items / freq)

    for blocks in range(freq):
      tree_features[blocks * span: (blocks + 1) * span, col] = 1
      col += 1

  tree_labels = np.eye(n_items)

  return tree_labels, tree_features


def hsd_context(n_items, n_contexts, symmetry=True, random=False):
  tree_labels_all, tree_features_all = [], []
  rng = np.random.default_rng()

  for c in range(n_contexts):
    tree_labels, tree_features = hsd(n_items)

    if random:
      tree_features = np.zeros_like(tree_features)
      tree_features[:,:3] = 1
      for row in tree_features:
        rng.shuffle(row)

    elif symmetry == False:
        while any([np.all(tree_features[:, 1] == prev_features[:, 1]) for prev_features in tree_features_all]):
          rng.shuffle(tree_features)

    tree_features_all.append(tree_features)

  tree_features_all = np.concatenate(tree_features_all, axis=1)
  return tree_labels, tree_features_all


def get_batch(labels, features, n_contexts, masking=True, separation=True):
  if type(labels) != torch.Tensor:
    labels = torch.tensor(labels).float()
  elif labels.dtype != torch.float:
    labels = labels.float()
  if type(features) != torch.Tensor:
    features = torch.tensor(features).float()
  elif features.dtype != torch.float:
    features = features.float()

  label_tensor_expanded = labels.repeat(n_contexts, 1)

  if separation:
    feature_tensor_expanded = features.repeat(n_contexts, 1)

    if masking == True:
      eye = torch.eye(n_contexts)
      mask = torch.repeat_interleave(eye, labels.shape[1], dim=0)
      mask = torch.repeat_interleave(mask, features.shape[1] // n_contexts, dim=1)
      assert mask.shape == feature_tensor_expanded.shape
      feature_tensor_expanded = mask * feature_tensor_expanded

  else:
    features_split = torch.split(features, features.shape[1] // n_contexts, dim=1)
    features_shuffled = [features_split[0]] + [f[:, torch.randperm(f.size()[1])] for f in features_split[1:]]
    feature_tensor_expanded = torch.cat(features_shuffled, axis=0)

  eye = torch.eye(n_contexts)
  context_tensor = torch.repeat_interleave(eye, labels.shape[1], dim=0)
  #     context_tensor = torch.argmax(context_tensor_one_hot, dim=0)

  return label_tensor_expanded, feature_tensor_expanded, context_tensor


def race_reduction(y, x, c, ctx_mode='all', n_iter=20, mult=1, show=True, one_plot=True):

  if ctx_mode not in ['all', 'first', 'on_only']:
    raise ValueError()

  learning_phases, y_contribs, winners = [], [], []
  dim_context = c.shape[1]
  dim_input = x.shape[1]
  n_per_context = int(c.shape[0] / c.shape[1])

  x_c = torch.cat((c, x), dim=1)

  sigma_xx = x_c.T @ x_c
  _, s_xx, _ = torch.svd(sigma_xx)

  x_c_split = torch.split(x_c, n_per_context, dim=0)
  x_split = torch.split(x, n_per_context, dim=0)

  pathway_list = [i for i in range(0, dim_context)]

  # combs = it.combinations(pathway_list, 2)
  all_combs = [[comb for comb in it.combinations(pathway_list, c)] for c in range(1, dim_context + 1)]
  all_combs = list(it.chain.from_iterable(all_combs))

  x_c_concat = [torch.cat([x_c_split[c] for c in comb], dim=0) for comb in all_combs]

  masks = [np.ones(x.shape[1]) for x in x_c_concat]
  all_ctx = set(np.arange(dim_context))
  if ctx_mode == 'all':
    partitions = [dim_context for _ in all_combs]
    pass
  elif ctx_mode == 'first':
    partitions = [dim_context if len(comb) == dim_context else 0 for comb in all_combs]
    for i, comb in enumerate(all_combs):
      if len(all_ctx - set(comb)) != 0:
        masks[i][:dim_context] = 0
  elif ctx_mode == 'on_only':
    partitions = [len(comb) for comb in all_combs]
    for i, comb in enumerate(all_combs):
      off_ctx = all_ctx - set(comb)
      off_ctx = list(off_ctx)
      masks[i][off_ctx] = 0

  x_list_cat = [torch.cat([x_c_split[i] if i in comb else torch.zeros_like(x_c_split[i])
                           for i in range(dim_context)], dim=0)
                for comb in all_combs]
  x_list = [torch.cat([x_c_split[i] for i in range(dim_context) if i in comb], dim=0)
                for comb in all_combs]

  fig, axes = plt.subplots(1, len(x_list))
  for i, x_ in enumerate(x_list):
    axes[i].imshow(x_.numpy())

  fig, axes = plt.subplots(1, len(x_list))
  for i, x_ in enumerate(x_list_cat):
    axes[i].imshow(x_.numpy())

  x_list_cat = [x[:, np.where(m)[0]] for x, m in zip(x_list_cat, masks)]
  x_list = [x[:, np.where(m)[0]] for x, m in zip(x_list, masks)]

  fig, axes = plt.subplots(1, len(x_list))
  for i, x_ in enumerate(x_list):
    axes[i].imshow(x_.numpy())

  fig, axes = plt.subplots(1, len(x_list))
  for i, x_ in enumerate(x_list_cat):
    axes[i].imshow(x_.numpy())

  y_split = torch.split(y, n_per_context, dim=0)
  y_concat = [torch.cat([y_split[c] for c in comb]) for comb in all_combs]

  sigmas_all = [(y_.T @ x_) for x_, y_ in zip(x_list, y_concat)]

  n_col = len(sigmas_all) + 3
  #
  all_titles = [''.join([str(i) for i in comb]) for comb in all_combs]
  all_titles = [r'$\Sigma^{' + '{}'.format(i) + '}$' for i in all_titles]

  y_sum = torch.zeros_like(y).T
  if show and one_plot:
    fig, axes = plt.subplots(n_iter, n_col, layout='constrained', figsize=(8, 2*n_iter))

  for j in range(n_iter):
    if show and not one_plot:
      fig, axes = plt.subplots(1, n_col, layout='constrained', figsize=(8, 14))
    #     fig.suptitle(j)
    #     for i, sig in enumerate(sigmas_all):
    #     lim = np.maximum(np.abs(np.min(sig.numpy())), np.max(sig.numpy()))
    #     show_svd(sig, lim)
    #     plt.tight_layout()

    s_max = 0
    for i, sig in enumerate(sigmas_all):
      ax = axes if not one_plot else axes[j]
      for s in ax[i].spines.values():
        s.set_alpha(0.3)
      U, S, V = torch.svd(sig)

      if show:
        lim = np.maximum(np.abs(np.min(sig.numpy())), np.max(sig.numpy()))
        im = ax[i].imshow(sig.numpy(), cmap='bwr', vmin=-lim, vmax=lim)
        # plt.colorbar(im)
        ax[i].xaxis.set_visible(False)
        if i == 0:
          ax[i].set_ylabel('race ' + str(j))
          ax[i].set_yticks([])
        else:
          ax[i].yaxis.set_visible(False)
        title = all_titles[i] + "\n%.2f" % np.max(S.numpy()) if j == 0 else "%.2f" % np.max(S.numpy())
        ax[i].set_title(title, fontdict={'fontsize':9})

      if torch.max(S) >= s_max:
        s_max = torch.max(S)
        U_max = U[:, 0]
        V_max = V[:, 0]
        winner = i

    if show:
      winner_title = (all_titles[winner] + "\n%.2f" % s_max) if j == 0 else "%.2f" % s_max
      ax[winner].set_title(winner_title, fontdict={'fontsize':9, 'fontweight':'bold'})
      for s in ax[winner].spines.values():
        s.set_alpha(1)

    sigma_contrib = mult * s_max * torch.outer(U_max, V_max)
    # if sigma_contrib.shape[1] != dim_input + dim_context:
      # sigma_contrib = torch.cat((torch.zeros(sigma_contrib.shape[0], dim_context), sigma_contrib), dim=1)
    if show:
      im = ax[-3].imshow(sigma_contrib.numpy())
      ax[-3].set_title(r'$\Sigma$ contrib', fontdict={'fontsize': 9})
      ax[-3].xaxis.set_visible(False)
      ax[-3].yaxis.set_visible(False)
      plt.colorbar(im)

    # sigma_c_contrib = torch.any(sigma_contrib[:, :partitions[winner]] > 1e-6 * mult)
    sigma_c_contrib = torch.any(torch.abs(V_max[:dim_context]) > 1e-6 * mult)
    sigma_x_contrib = torch.any(torch.abs(V_max[dim_context:]) > 1e-6 * mult)
    print(V_max)
    # sigma_x_contrib = torch.any(sigma_contrib[:, partitions[winner]:] > 1e-6 * mult)
    if sigma_c_contrib and sigma_x_contrib:
      print('both')
      div = s_xx[0]
    elif sigma_c_contrib:
      print('c')
      div = s_xx[1]
    elif sigma_x_contrib:
      print('x')
      div = s_xx[1+dim_context]
    else:
      raise ValueError()

    # if j == 0:
    #   div = dim_context + dim_input
    # elif j == 1:
    #   div = dim_context
    # elif j in [2, 3]:
    #   div = dim_input
    # elif j in [4, 5]:
    #   div = dim_context
    # #     elif j==4:
    # #         div = 3
    # else:
    #   div = 4

    y_contrib = (1 / div) * sigma_contrib @ x_list_cat[winner].T
    y_sum += y_contrib
    if show:
      im = ax[-2].imshow(y_contrib.numpy())
      ax[-2].set_title(r'$y$ contrib', fontdict={'fontsize': 9})
      ax[-2].xaxis.set_visible(False)
      ax[-2].yaxis.set_visible(False)
      plt.colorbar(im)

      lim = np.maximum(np.abs(np.min(y_sum.numpy())), np.max(y_sum.numpy()))
      im = ax[-1].imshow(y_sum.numpy(), vmin=-lim, vmax=lim)
      ax[-1].title.set_text(r'$y$')
      ax[-1].xaxis.set_visible(False)
      ax[-1].yaxis.set_visible(False)
      plt.colorbar(im)

    y_contrib_split = torch.split(y_contrib, n_per_context, dim=1)
    y_contrib_concat = [torch.cat([y_contrib_split[c] for c in comb], dim=1) for comb in all_combs]
    sigmas_explained = [y_ @ x_ for x_, y_ in zip(x_list, y_contrib_concat)]
    sigmas_all = [s - s_explained for s, s_explained in zip(sigmas_all, sigmas_explained)]

    learning_phases.append((s_max, div))
    y_contribs.append(y_contrib)
    winners.append(winner)

  return learning_phases, y_contribs, winners


from matplotlib.animation import FuncAnimation


def race_reduction_anim(y, x, c, ctx_mode='all', n_iter=20, mult=1, show=True, one_plot=True, fps=1):
  if ctx_mode not in ['all', 'first', 'on_only']:
    raise ValueError()

  learning_phases, y_contribs, winners = [], [], []
  dim_context = c.shape[1]
  dim_input = x.shape[1]
  n_per_context = int(c.shape[0] / c.shape[1])

  x_c = torch.cat((c, x), dim=1)

  sigma_xx = x_c.T @ x_c
  _, s_xx, _ = torch.svd(sigma_xx)

  x_c_split = torch.split(x_c, n_per_context, dim=0)
  x_split = torch.split(x, n_per_context, dim=0)

  pathway_list = [i for i in range(0, dim_context)]

  # combs = it.combinations(pathway_list, 2)
  all_combs = [[comb for comb in it.combinations(pathway_list, c)] for c in range(1, dim_context + 1)]
  all_combs = list(it.chain.from_iterable(all_combs))

  x_c_concat = [torch.cat([x_c_split[c] for c in comb], dim=0) for comb in all_combs]

  masks = [np.ones(x.shape[1]) for x in x_c_concat]
  all_ctx = set(np.arange(dim_context))
  if ctx_mode == 'all':
    partitions = [dim_context for _ in all_combs]
    pass
  elif ctx_mode == 'first':
    partitions = [dim_context if len(comb) == dim_context else 0 for comb in all_combs]
    for i, comb in enumerate(all_combs):
      if len(all_ctx - set(comb)) != 0:
        masks[i][:dim_context] = 0
  elif ctx_mode == 'on_only':
    partitions = [len(comb) for comb in all_combs]
    for i, comb in enumerate(all_combs):
      off_ctx = all_ctx - set(comb)
      off_ctx = list(off_ctx)
      masks[i][off_ctx] = 0

  x_list_cat = [torch.cat([x_c_split[i] if i in comb else torch.zeros_like(x_c_split[i])
                           for i in range(dim_context)], dim=0)
                for comb in all_combs]
  x_list = [torch.cat([x_c_split[i] for i in range(dim_context) if i in comb], dim=0)
                for comb in all_combs]

  fig, axes = plt.subplots(1, len(x_list))
  for i, x_ in enumerate(x_list):
    axes[i].imshow(x_.numpy())

  fig, axes = plt.subplots(1, len(x_list))
  for i, x_ in enumerate(x_list_cat):
    axes[i].imshow(x_.numpy())

  x_list_cat = [x[:, np.where(m)[0]] for x, m in zip(x_list_cat, masks)]
  x_list = [x[:, np.where(m)[0]] for x, m in zip(x_list, masks)]

  fig, axes = plt.subplots(1, len(x_list))
  for i, x_ in enumerate(x_list):
    axes[i].imshow(x_.numpy())

  fig, axes = plt.subplots(1, len(x_list))
  for i, x_ in enumerate(x_list_cat):
    axes[i].imshow(x_.numpy())

  y_split = torch.split(y, n_per_context, dim=0)
  y_concat = [torch.cat([y_split[c] for c in comb]) for comb in all_combs]

  sigmas_all = [(y_.T @ x_) for x_, y_ in zip(x_list, y_concat)]

  n_col = len(sigmas_all) + 3
  #
  all_titles = [''.join([str(i) for i in comb]) for comb in all_combs]
  all_titles = [r'$\Sigma^{' + '{}'.format(i) + '}$' for i in all_titles]

  y_sum = torch.zeros_like(y).T

  fig, axes = plt.subplots(1, n_col, layout='constrained', figsize=(8, 2 * n_iter))

  def update(j):
    nonlocal axes
    nonlocal fig
    nonlocal y_sum
    nonlocal winners
    nonlocal learning_phases
    nonlocal y_contribs
    nonlocal sigmas_all

    #     fig.suptitle(j)
    #     for i, sig in enumerate(sigmas_all):
    #     lim = np.maximum(np.abs(np.min(sig.numpy())), np.max(sig.numpy()))
    #     show_svd(sig, lim)
    #     plt.tight_layout()

    s_max = 0
    for i, sig in enumerate(sigmas_all):
      ax = axes
      for s in ax[i].spines.values():
        s.set_alpha(0.3)
      U, S, V = torch.svd(sig)

      lim = np.maximum(np.abs(np.min(sig.numpy())), np.max(sig.numpy()))
      im = ax[i].imshow(sig.numpy(), cmap='bwr', vmin=-lim, vmax=lim)
      # plt.colorbar(im)
      ax[i].xaxis.set_visible(False)
      if i == 0:
        ax[i].set_ylabel('race ' + str(j))
        ax[i].set_yticks([])
      else:
        ax[i].yaxis.set_visible(False)
      title = all_titles[i] + "\n%.2f" % np.max(S.numpy()) if j == 0 else "%.2f" % np.max(S.numpy())
      ax[i].set_title(title, fontdict={'fontsize': 9})

      if torch.max(S) >= s_max:
        s_max = torch.max(S)
        U_max = U[:, 0]
        V_max = V[:, 0]
        winner = i

    if show:
      winner_title = (all_titles[winner] + "\n%.2f" % s_max) if j == 0 else "%.2f" % s_max
      ax[winner].set_title(winner_title, fontdict={'fontsize': 9, 'fontweight': 'bold'})
      for s in ax[winner].spines.values():
        s.set_alpha(1)

    sigma_contrib = mult * s_max * torch.outer(U_max, V_max)
    # if sigma_contrib.shape[1] != dim_input + dim_context:
    # sigma_contrib = torch.cat((torch.zeros(sigma_contrib.shape[0], dim_context), sigma_contrib), dim=1)
    im = ax[-3].imshow(sigma_contrib.numpy())
    ax[-3].set_title(r'$\Sigma$ contrib', fontdict={'fontsize': 9})
    ax[-3].xaxis.set_visible(False)
    ax[-3].yaxis.set_visible(False)
    # plt.colorbar(im)

    # sigma_c_contrib = torch.any(sigma_contrib[:, :partitions[winner]] > 1e-6 * mult)
    sigma_c_contrib = torch.any(torch.abs(V_max[:dim_context]) > 1e-6 * mult)
    sigma_x_contrib = torch.any(torch.abs(V_max[dim_context:]) > 1e-6 * mult)
    print(V_max)
    # sigma_x_contrib = torch.any(sigma_contrib[:, partitions[winner]:] > 1e-6 * mult)
    if sigma_c_contrib and sigma_x_contrib:
      print('both')
      div = s_xx[0]
    elif sigma_c_contrib:
      print('c')
      div = s_xx[1]
    elif sigma_x_contrib:
      print('x')
      div = s_xx[1 + dim_context]
    else:
      raise ValueError()

    # if j == 0:
    #   div = dim_context + dim_input
    # elif j == 1:
    #   div = dim_context
    # elif j in [2, 3]:
    #   div = dim_input
    # elif j in [4, 5]:
    #   div = dim_context
    # #     elif j==4:
    # #         div = 3
    # else:
    #   div = 4

    y_contrib = (1 / div) * sigma_contrib @ x_list_cat[winner].T
    y_sum += y_contrib

    im = ax[-2].imshow(y_contrib.numpy())
    ax[-2].set_title(r'$y$ contrib', fontdict={'fontsize': 9})
    ax[-2].xaxis.set_visible(False)
    ax[-2].yaxis.set_visible(False)
    # plt.colorbar(im)

    lim = np.maximum(np.abs(np.min(y_sum.numpy())), np.max(y_sum.numpy()))
    im = ax[-1].imshow(y_sum.numpy(), vmin=-lim, vmax=lim)
    ax[-1].title.set_text(r'$y$')
    ax[-1].xaxis.set_visible(False)
    ax[-1].yaxis.set_visible(False)
    # plt.colorbar(im)

    y_contrib_split = torch.split(y_contrib, n_per_context, dim=1)
    y_contrib_concat = [torch.cat([y_contrib_split[c] for c in comb], dim=1) for comb in all_combs]
    sigmas_explained = [y_ @ x_ for x_, y_ in zip(x_list, y_contrib_concat)]
    sigmas_all = [s - s_explained for s, s_explained in zip(sigmas_all, sigmas_explained)]

    learning_phases.append((s_max, div))
    y_contribs.append(y_contrib)
    winners.append(winner)

    return axes

  ani = FuncAnimation(fig, update, frames=range(n_iter), blit=False, repeat=False)
    # plt.show()

  ani.save('a.gif', writer='mencoder', fps=fps)

  return learning_phases, y_contribs, winners