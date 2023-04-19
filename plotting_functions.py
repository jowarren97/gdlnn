# @title Plotting functions
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from matplotlib.animation import FuncAnimation
import math

def plot_inputs_targets_contexts(inputs, targets, context, held_out=(), labels=None, features=None, plot_specs=None, name='data'):
  cmap = matplotlib.colors.ListedColormap(['cyan', 'yellow', 'magenta'])

  dim_input = inputs.shape[1]
  dim_output = targets.shape[1]
  dim_context = context.shape[1]
  batch_size = inputs.shape[0]

  if labels is not None and features is not None:
    gs_kw = dict(width_ratios=[dim_input, dim_output, dim_context], height_ratios=[dim_input, batch_size])
    fig, axd = plt.subplot_mosaic([['labels', 'features', '.'],
                                   ['inputs', 'targets', 'context']],
                                  gridspec_kw=gs_kw,
                                  layout="tight")
  else:
    gs_kw = dict(width_ratios=[dim_input, dim_output, dim_context])
    fig, axd = plt.subplot_mosaic([['inputs', 'targets', 'context']],
                                  gridspec_kw=gs_kw,
                                  layout="tight")

  for ax in axd.items():
    #     ax[-1].xaxis.set_visible(False)
    #     ax[-1].yaxis.set_visible(False)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
    ax[-1].title.set_text(ax[0])

    if ax[0] == 'targets':
      ax[-1].set_xlabel('feature')
    #         ax[-1].set_xticks([0,7,14])
    if ax[0] == 'inputs':
      ax[-1].set_xlabel('item')
      ax[-1].set_ylabel('samples')
    #         ax[-1].set_yticks([0,4,8])
    if ax[0] == 'context':
      ax[-1].set_xlabel('context')

  if type(targets) == torch.Tensor:
    targets = targets.clone().detach().numpy()
  if type(inputs) == torch.Tensor:
    inputs = inputs.clone().detach().numpy()
  if type(context) == torch.Tensor:
    context = context.clone().detach().numpy()

  for i in held_out:
    targets[i, :] = 0.5 * targets[i, :]
    inputs[i, :] = 0.5 * inputs[i, :]
    context[i, :] = 0.5 * context[i, :]

  if labels is not None and features is not None:
    axd['labels'].imshow(labels, cmap=cmap)
    axd['features'].imshow(features, cmap=cmap)
  axd['targets'].imshow(targets, cmap=cmap)
  axd['inputs'].imshow(inputs, cmap=cmap)
  axd['context'].imshow(context, cmap=cmap)

  if plot_specs is not None:
    if plot_specs['show']:
      plt.show()
    if plot_specs['save']:
      fig.savefig(plot_specs['save_dir'] + name + '.png', format='png')


def plot_preds_targets(preds, targets, separate_contexts, dim_context, plot_specs=None, name='preds'):
  lim_pred = np.maximum(np.abs(np.min(preds)), np.max(preds))
  lim_target = np.maximum(np.abs(np.min(targets)), np.max(targets))

  fig, axes = plt.subplots(1, 2, figsize=(10,4))
  im=axes[0].imshow(preds, vmin=-lim_pred, vmax=lim_pred)
  plt.colorbar(im)
  axes[0].title.set_text('Predictions')
  im=axes[1].imshow(targets, vmin=-lim_target, vmax=lim_target)
  plt.colorbar(im)
  axes[1].title.set_text('Targets')

  for ax in axes:
      if not separate_contexts:
          ax.set_yticks(ticks=np.arange(-.5, targets.shape[0], targets.shape[0] // dim_context), labels=[], minor=False)
          ax.set_xticks(ticks=[], minor=False)
      else:
          ax.set_xticks(ticks=np.arange(-.5, targets.shape[1], targets.shape[1] // dim_context), labels=[], minor=False)
          ax.set_yticks(ticks=[], minor=False)
      ax.grid(which='major', color='w', linestyle='-', linewidth=1.5)

  if plot_specs is not None and plot_specs['save']:
    fig.savefig(plot_specs['save_dir'] + name + '.png', format='png')


def animate_preds(preds_timeseries, end_epoch, separate_contexts, dim_context, duration=10, fps=10, plot_specs=None, name='preds_anim'):
  dim_output = preds_timeseries.shape[-1]
  dim_input = preds_timeseries.shape[1] // dim_context

  n_frames = duration * fps
  interval = math.ceil(end_epoch / n_frames)
  vmin, vmax = np.min(preds_timeseries), np.max(preds_timeseries)

  fig = plt.figure()
  ax = plt.gca()

  # creating a plot
  image_plotted = ax.imshow(preds_timeseries[0,:,:], vmin=vmin, vmax=vmax)

  if not separate_contexts:
      ax.set_yticks(ticks=np.arange(-.5, dim_context * dim_input, dim_input), labels=[], minor=False)
      ax.set_xticks(ticks=[], minor=False)
  else:
      ax.set_xticks(ticks=np.arange(-.5, dim_output, dim_output // dim_context), labels=[], minor=False)
      ax.set_yticks(ticks=[], minor=False)
  ax.grid(which='major', color='w', linestyle='-', linewidth=2)

  def AnimationFunction(frame):
      image_plotted.set_data(preds_timeseries[frame * interval, :, :])

  ani = FuncAnimation(fig, AnimationFunction, frames=n_frames, interval=25)

  if plot_specs is not None and plot_specs['save']:
    ani.save(plot_specs['save_dir'] + name + '.gif', writer='mencoder', fps=fps)


# def animate_pathway_preds(pathway_preds, dim_context, duration=10, fps=10, plot_specs=None, data_interval=None, name='pathway_preds_anim'):
#   dim_output = pathway_preds.shape[-1]
#   dim_input = pathway_preds.shape[-2] // dim_context
#   # n_frames = duration * fps
#   n_frames = pathway_preds.shape[1]
#   interval = (duration * 1000) // n_frames
#   print(interval)
#   # interval = math.ceil(end_epoch / n_frames)
#
#   vmin, vmax = np.min(pathway_preds), np.max(pathway_preds)
#
#   images_plotted = []
#
#   n_panels = pathway_preds.shape[0] + 1
#   fig, axes = plt.subplots(n_panels, 1, figsize=(8, 10))
#
#   if data_interval is not None:
#     title = fig.suptitle('0')
#
#   for p in range(n_panels):
#     if p < n_panels - 1:
#       axes[p].title.set_text('Pathway ' + r'${}$'.format(p) + ' contribution to output')
#       images_plotted.append(axes[p].imshow(pathway_preds[p,0,:,:], vmin=vmin, vmax=vmax))
#     else:
#       axes[p].title.set_text('Sum of contributions')
#       images_plotted.append(axes[p].imshow(np.sum(pathway_preds, axis=0)[0,:,:], vmin=vmin, vmax=vmax))
#     axes[p].xaxis.set_visible(False)
#     axes[p].yaxis.set_visible(False)
#
#   def AnimationFunction(frame):
#     for p in range(n_panels):
#       if p < n_panels - 1:
#         images_plotted[p].set_data(pathway_preds[p, frame, :, :])
#       else:
#         images_plotted[p].set_data(np.sum(pathway_preds, axis=0)[frame, :, :])
#
#     if data_interval is not None:
#       title.set_text(frame * data_interval)
#
#   ani = FuncAnimation(fig, AnimationFunction, frames=n_frames, interval=interval, blit=False)
#
#   if plot_specs is not None and plot_specs['save']:
#     ani.save(name + '.gif', writer='mencoder', fps=fps)

def animate_pathway_preds(pathway_preds, dim_context, duration=10, fps=10, additional_data_dict={}, plot_specs=None,
                          data_interval=None, name='pathway_preds_anim', hor=False):
  dim_output = pathway_preds.shape[-1]
  dim_input = pathway_preds.shape[-2] // dim_context
  # n_frames = duration * fps
  n_frames = pathway_preds.shape[1]
  interval = (duration * 1000) // n_frames
  # interval = math.ceil(end_epoch / n_frames)
  fixed_lims = True
  pred_lim = np.maximum(np.abs(np.min(pathway_preds)), np.max(pathway_preds))
  pred_lim = np.maximum(np.max(np.sum(pathway_preds, axis=0)), pred_lim)
  preds_plotted = []

  gridspec = {'width_ratios':[]}
  if 'weights' in additional_data_dict:
    weights = additional_data_dict['weights']
    weights_plotted = []
    W_lim = [np.maximum(np.abs(np.min(W_pathway)), np.max(W_pathway)) for W_pathway in weights]

  if 'U' in additional_data_dict:
    U = additional_data_dict['U']
    U_plotted = []
    U_lim = [np.maximum(np.abs(np.min(U_pathway)), np.max(U_pathway)) for U_pathway in U]

  if 'S' in additional_data_dict:
    S = additional_data_dict['S']
    S_plotted = []
    S_labels = []
    S_lim = [np.maximum(np.abs(np.min(S_pathway)), np.max(S_pathway)) for S_pathway in S]

  if 'V' in additional_data_dict:
    V = additional_data_dict['V']
    V_plotted = []
    V_lim = [np.maximum(np.abs(np.min(V_pathway)), np.max(V_pathway)) for V_pathway in V]

  n_col = 1 + len(additional_data_dict)
  n_rows = pathway_preds.shape[0] + 1
  if hor:
    n_col, n_rows = n_rows, n_col
  n_panel = n_rows if not hor else n_col

  def get_ax(p, idx):
      if hor:
          return axes[p] if n_rows == 1 else axes[idx, p]
      else:
          return axes[p] if n_col == 1 else axes[p, idx]

  # gridspec = dict(width_ratios=[dim_input, dim_output, dim_context], height_ratios=[dim_input, batch_size]
  fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, figsize=(2 * n_col, 3 * n_rows))
  plt.tight_layout(rect=[0, 0.03, 1, 0.90])

  if data_interval is not None:
    title = fig.suptitle('0')

  for p in range(n_rows if not hor else n_col):
    idx = 0
    ax = get_ax(p, idx)
    if p < n_panel - 1:
      ax.title.set_text(r'Output ${}$'.format(p))
      preds_plotted.append(ax.imshow(pathway_preds[p,0,:,:].T, vmin=-pred_lim, vmax=pred_lim))
    else:
      ax.title.set_text('Net')
      im = ax.imshow(np.sum(pathway_preds, axis=0)[0,:,:].T, vmin=-pred_lim, vmax=pred_lim)
      preds_plotted.append(im)
      # plt.colorbar(im)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if 'weights' in additional_data_dict:
      idx += 1
      ax = get_ax(p, idx)
      if p < n_panel - 1:
        ax.title.set_text(r'$W_{}$'.format(p))
        init_weight = weights[p][0,:,:]
        im = ax.imshow(init_weight)
        if fixed_lims:
          im.set_clim(vmin=-W_lim[p], vmax=W_lim[p])
        plt.colorbar(im)
        weights_plotted.append(im)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
      else:
        ax.set_axis_off()

    if 'U' in additional_data_dict:
      idx += 1
      ax = get_ax(p, idx)
      if p < n_panel - 1:
        ax.title.set_text(r'$U_{}$'.format(p))
        init_U = U[p][0,:,:]
        im = ax.imshow(init_U, cmap='bwr')
        if fixed_lims:
          im.set_clim(vmin=-U_lim[p], vmax=U_lim[p])
        plt.colorbar(im)
        U_plotted.append(im)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
      else:
        ax.set_axis_off()

    if 'S' in additional_data_dict:
      idx += 1
      ax = get_ax(p, idx)
      if p < n_panel - 1:
        ax.title.set_text(r'$S_{}$'.format(p))
        init_S = np.diag(S[p][0,:])
        im = ax.imshow(init_S, cmap='bwr')
        if fixed_lims:
          im.set_clim(vmin=-S_lim[p], vmax=S_lim[p])
        plt.colorbar(im)
        S_plotted.append(im)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        S_pathway_labels = []
        for (j, i), label in np.ndenumerate(init_S):
          if label != 0:
            S_pathway_labels.append(ax.text(i, j, ("%.1f" % label), ha='center', va='center', fontsize=5))
        S_labels.append(S_pathway_labels)
      else:
        ax.set_axis_off()

    if 'V' in additional_data_dict:
      idx += 1
      ax = get_ax(p, idx)
      if p < n_panel - 1:
        ax.title.set_text(r'$V^T_{}$'.format(p))
        init_V = V[p][0,:,:].T
        im = ax.imshow(init_V, cmap='bwr')
        if fixed_lims:
          im.set_clim(vmin=-V_lim[p], vmax=V_lim[p])
        plt.colorbar(im)
        V_plotted.append(im)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
      else:
        ax.set_axis_off()

  def AnimationFunction(frame):
    n_panel = n_rows if not hor else n_col
    for p in range(n_rows if not hor else n_col):
      if p < n_panel - 1:
        preds_plotted[p].set_data(pathway_preds[p, frame, :, :].T)
      else:
        preds_plotted[p].set_data(np.sum(pathway_preds, axis=0)[frame, :, :].T)

      if 'weights' in additional_data_dict:
        if p < n_panel - 1:
          new_w = weights[p][frame, :, :]
          weights_plotted[p].set_data(new_w)
          if not fixed_lims:
            lim = np.maximum(np.abs(np.min(new_w)), np.max(new_w))
            weights_plotted[p].set_clim(vmin=-lim, vmax=lim)
        # else:
          # w_sum = weights[0][frame, :, :]
          # w_array = np.array(weights)
          # w_to_add = np.sum(w_array[1:,:,:,:], dim=0)
          # w_sum  += np.concatenate([np.zeros([w_sum.shape[0], w_sum.shape[1]-w_to_add.shape[1], w_to_add])])


      if 'U' in additional_data_dict and p < n_rows - 1:
        new_U = U[p][frame, :, :]
        U_plotted[p].set_data(new_U)
        if not fixed_lims:
          lim = np.maximum(np.abs(np.min(new_U)), np.max(new_U))
          U_plotted[p].set_clim(vmin=-lim, vmax=lim)

      if 'S' in additional_data_dict and p < n_rows - 1:
        new_S = np.diag(S[p][frame, :])
        S_plotted[p].set_data(new_S)
        if not fixed_lims:
          lim = np.maximum(np.abs(np.min(new_S)), np.max(new_S))
          S_plotted[p].set_clim(vmin=-lim, vmax=lim)
        for i, label in enumerate(S[p][frame, :]):
          if label != 0:
            S_labels[p][i].set_text("%.1f" % label)

      if 'V' in additional_data_dict and p < n_rows - 1:
        new_V = V[p][frame, :, :].T
        V_plotted[p].set_data(new_V)
        if not fixed_lims:
          lim = np.maximum(np.abs(np.min(new_V)), np.max(new_V))
          V_plotted[p].set_clim(vmin=-lim, vmax=lim)

    if data_interval is not None:
      title.set_text(frame * data_interval)

  ani = FuncAnimation(fig, AnimationFunction, frames=n_frames, interval=interval, blit=False)

  if plot_specs is not None and plot_specs['save']:
    # FFwriter = matplotlib.animation.FFMpegWriter(fps=10)
    ani.save(plot_specs['save_dir'] + name + '.mp4', fps=fps, writer='ffmpeg')
    ani.save(plot_specs['save_dir'] + name + '.gif', writer='pillow', fps=fps)

def plot_x_y_hier_data(im1, im2, subplot_ratio=[1, 2]):
  """
  Plot hierarchical data of labels vs features
  for all samples

  Args:
    im1: np.ndarray
      Input Dataset
    im2: np.ndarray
      Targets
    subplot_ratio: list
      Subplot ratios used to create subplots of varying sizes

  Returns:
    Nothing
  """
  fig = plt.figure(figsize=(12, 5))
  gs = gridspec.GridSpec(1, 2, width_ratios=subplot_ratio)
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1])
  ax0.imshow(im1, cmap="cool")
  ax1.imshow(im2, cmap="cool")
  ax0.set_title("Labels of all samples")
  ax1.set_title("Features of all samples")
  ax0.set_axis_off()
  ax1.set_axis_off()
  plt.tight_layout()
  plt.show()


def plot_x_y_hier_one(im1, im2, subplot_ratio=[1, 2]):
  """
  Plot hierarchical data of labels vs features
  for a single sample

  Args:
    im1: np.ndarray
      Input Dataset
    im2: np.ndarray
      Targets
    subplot_ratio: list
      Subplot ratios used to create subplots of varying sizes

  Returns:
    Nothing
  """
  fig = plt.figure(figsize=(12, 1))
  gs = gridspec.GridSpec(1, 2, width_ratios=subplot_ratio)
  ax0 = plt.subplot(gs[0])
  ax1 = plt.subplot(gs[1])
  ax0.imshow(im1, cmap="cool")
  ax1.imshow(im2, cmap="cool")
  ax0.set_title("Labels of a single sample")
  ax1.set_title("Features of a single sample")
  ax0.set_axis_off()
  ax1.set_axis_off()
  plt.tight_layout()
  plt.show()


def plot_tree_data(label_list = None, feature_array = None, new_feature = None):
  """
  Plot tree data

  Args:
    label_list: np.ndarray
      List of labels [default: None]
    feature_array: np.ndarray
      List of features [default: None]
    new_feature: string
      Enables addition of new features

  Returns:
    Nothing
  """
  cmap = matplotlib.colors.ListedColormap(['cyan', 'magenta'])
  n_features = 10
  n_labels = 8
  im1 = np.eye(n_labels)
  if feature_array is None:
    im2 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1],
                      [1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 1, 1, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 0],
                      [0, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 1]]).T
    im2[im2 == 0] = -1
    feature_list = ['can_grow',
                    'is_mammal',
                    'has_leaves',
                    'can_move',
                    'has_trunk',
                    'can_fly',
                    'can_swim',
                    'has_stem',
                    'is_warmblooded',
                    'can_flower']
  else:
    im2 = feature_array
  if label_list is None:
    label_list = ['Goldfish', 'Tuna', 'Robin', 'Canary',
                  'Rose', 'Daisy', 'Pine', 'Oak']
  fig = plt.figure(figsize=(12, 7))
  gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.35])
  ax1 = plt.subplot(gs[0])
  ax2 = plt.subplot(gs[1])
  ax1.imshow(im1, cmap=cmap)
  if feature_array is None:
    implt = ax2.imshow(im2, cmap=cmap, vmin=-1.0, vmax=1.0)
  else:
    implt = ax2.imshow(im2[:, -n_features:], cmap=cmap, vmin=-1.0, vmax=1.0)
  divider = make_axes_locatable(ax2)
  cax = divider.append_axes("right", size="5%", pad=0.1)
  cbar = plt.colorbar(implt, cax=cax, ticks=[-0.5, 0.5])
  cbar.ax.set_yticklabels(['no', 'yes'])
  ax1.set_title("Labels")
  ax1.set_yticks(ticks=np.arange(n_labels))
  ax1.set_yticklabels(labels=label_list)
  ax1.set_xticks(ticks=np.arange(n_labels))
  ax1.set_xticklabels(labels=label_list, rotation='vertical')
  ax2.set_title("{} random Features".format(n_features))
  ax2.set_yticks(ticks=np.arange(n_labels))
  ax2.set_yticklabels(labels=label_list)
  if feature_array is None:
    ax2.set_xticks(ticks=np.arange(n_features))
    ax2.set_xticklabels(labels=feature_list, rotation='vertical')
  else:
    ax2.set_xticks(ticks=[n_features-1])
    ax2.set_xticklabels(labels=[new_feature], rotation='vertical')
  plt.tight_layout()
  plt.show()


def plot_loss(loss_array,
              title="Training loss (Mean Squared Error)",
              c="r"):
  """
  Plot loss function

  Args:
    c: string
      Specifies plot color
    title: string
      Specifies plot title
    loss_array: np.ndarray
      Log of MSE loss per epoch

  Returns:
    Nothing
  """
  plt.figure(figsize=(10, 5))
  plt.plot(loss_array, color=c)
  plt.xlabel("Epoch")
  plt.ylabel("MSE")
  plt.title(title)
  plt.show()


def plot_loss_sv(loss_array, sv_array):
  """
  Plot loss function

  Args:
    sv_array: np.ndarray
      Log of singular values/modes across epochs
    loss_array: np.ndarray
      Log of MSE loss per epoch

  Returns:
    Nothing
  """
  n_sing_values = sv_array.shape[1]
  sv_array = sv_array / np.max(sv_array)
  cmap = plt.cm.get_cmap("Set1", n_sing_values)

  _, (plot1, plot2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
  plot1.set_title("Training loss (Mean Squared Error)")
  plot1.plot(loss_array, color='r')

  plot2.set_title("Evolution of singular values (modes)")
  for i in range(n_sing_values):
    plot2.plot(sv_array[:, i], c=cmap(i))
  plot2.set_xlabel("Epoch")
  plt.show()


def plot_loss_sv_twin(loss_array, sv_array):
  """
  Plot learning dynamics

  Args:
    sv_array: np.ndarray
      Log of singular values/modes across epochs
    loss_array: np.ndarray
      Log of MSE loss per epoch

  Returns:
    Nothing
  """
  n_sing_values = sv_array.shape[1]
  sv_array = sv_array / np.max(sv_array)
  cmap = plt.cm.get_cmap("winter", n_sing_values)

  fig = plt.figure(figsize=(10, 5))
  ax1 = plt.gca()
  ax1.set_title("Learning Dynamics")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Mean Squared Error", c='r')
  ax1.tick_params(axis='y', labelcolor='r')
  ax1.plot(loss_array, color='r')

  ax2 = ax1.twinx()
  ax2.set_ylabel("Singular values (modes)", c='b')
  ax2.tick_params(axis='y', labelcolor='b')
  for i in range(n_sing_values):
    ax2.plot(sv_array[:, i], c=cmap(i))

  fig.tight_layout()
  plt.show()


def plot_ills_sv_twin(ill_array, sv_array, ill_label):
  """
  Plot network training evolution
  and illusory correlations

  Args:
    sv_array: np.ndarray
      Log of singular values/modes across epochs
    ill_array: np.ndarray
      Log of illusory correlations per epoch
    ill_label: np.ndarray
      Log of labels associated with illusory correlations

  Returns:
    Nothing
  """
  n_sing_values = sv_array.shape[1]
  sv_array = sv_array / np.max(sv_array)
  cmap = plt.cm.get_cmap("winter", n_sing_values)

  fig = plt.figure(figsize=(10, 5))
  ax1 = plt.gca()
  ax1.set_title("Network training and the Illusory Correlations")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel(ill_label, c='r')
  ax1.tick_params(axis='y', labelcolor='r')
  ax1.plot(ill_array, color='r', linewidth=3)
  ax1.set_ylim(-1.05, 1.05)

  ax2 = ax1.twinx()
  ax2.set_ylabel("Singular values (modes)", c='b')
  ax2.tick_params(axis='y', labelcolor='b')
  for i in range(n_sing_values):
    ax2.plot(sv_array[:, i], c=cmap(i))

  fig.tight_layout()
  plt.show()


def plot_loss_sv_rsm(loss_array, sv_array, rsm_array, i_ep, item_names=None):
  """
  Plot learning dynamics

  Args:
    sv_array: np.ndarray
      Log of singular values/modes across epochs
    loss_array: np.ndarray
      Log of MSE loss per epoch
    rsm_array: torch.tensor
      Representation similarity matrix
    i_ep: int
      Which epoch to show

  Returns:
    Nothing
  """
  n_ep = loss_array.shape[0]
  rsm_array = rsm_array / np.max(rsm_array)
  sv_array = sv_array / np.max(sv_array)

  n_sing_values = sv_array.shape[1]
  cmap = plt.cm.get_cmap("winter", n_sing_values)

  fig = plt.figure(figsize=(14, 5))
  gs = gridspec.GridSpec(1, 2, width_ratios=[5, 3])

  ax0 = plt.subplot(gs[1])
  ax0.yaxis.tick_right()
  implot = ax0.imshow(rsm_array[i_ep], cmap="Purples", vmin=0.0, vmax=1.0)
  divider = make_axes_locatable(ax0)
  cax = divider.append_axes("right", size="5%", pad=0.9)
  cbar = plt.colorbar(implot, cax=cax, ticks=[])
  cbar.ax.set_ylabel('Similarity', fontsize=12)
  ax0.set_title("RSM at epoch {}".format(i_ep), fontsize=16)
  ax0.set_yticks(ticks=np.arange(n_sing_values))
  ax0.set_yticklabels(labels=item_names)
  ax0.set_xticks(ticks=np.arange(n_sing_values))
  ax0.set_xticklabels(labels=item_names, rotation='vertical')

  ax1 = plt.subplot(gs[0])
  ax1.set_title("Learning Dynamics", fontsize=16)
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Mean Squared Error", c='r')
  ax1.tick_params(axis='y', labelcolor='r', direction="in")
  ax1.plot(np.arange(n_ep), loss_array, color='r')
  ax1.axvspan(i_ep-2, i_ep+2, alpha=0.2, color='m')

  ax2 = ax1.twinx()
  ax2.set_ylabel("Singular values", c='b')
  ax2.tick_params(axis='y', labelcolor='b', direction="in")
  for i in range(n_sing_values):
    ax2.plot(np.arange(n_ep), sv_array[:, i], c=cmap(i))
  ax1.set_xlim(-1, n_ep+1)
  ax2.set_xlim(-1, n_ep+1)

  plt.show()


def show_svd(sigma, sigma_lim, sigma_xx=None, cbar=True, name='svd'):
  fig, axes = plt.subplots(1, 4 if sigma_xx is None else 6, figsize=(10, 6))
  U, S, V = torch.svd(sigma)
  offset = 0

  if sigma_xx is not None:
    _, S_xx, _ = torch.svd(sigma_xx, compute_uv=False)
    S_norm = S / S_xx
    S_norm[S_xx < 1e-5] = 0.0

    sigma_xx = sigma_xx.numpy()

    lim = np.maximum(np.abs(np.min(sigma_xx)), np.max(sigma_xx))
    im = axes[0].imshow(sigma_xx, cmap='bwr', vmin=-lim, vmax=lim)
    axes[0].title.set_text(r'$\Sigma_{xx}$')
    if cbar: plt.colorbar(im)
    offset = 1

    S_norm, idx_sorted = torch.sort(S_norm, descending=True)
    S = S[idx_sorted]
    U = U[:, idx_sorted]
    V = V[:, idx_sorted]

    S_norm = np.diag(S_norm.numpy())

  sigma, U, S, V = sigma.numpy(), U.numpy(), np.diag(S.numpy()), V.numpy()

  #     lim = np.maximum(np.abs(np.min(sigma1.numpy())), np.max(sigma.numpy()))

  im = axes[0 + offset].imshow(sigma, cmap='bwr', vmin=-sigma_lim, vmax=sigma_lim)
  if cbar: plt.colorbar(im)
  axes[0 + offset].title.set_text(r'$\Sigma_{yx}$')

  lim = np.maximum(np.abs(np.min(U)), np.max(U))
  axes[1 + offset].imshow(U, cmap='bwr', vmin=-lim, vmax=lim)
  axes[1 + offset].title.set_text(r'$U$')

  lim = np.maximum(np.abs(np.min(S)), np.max(S))
  axes[2 + offset].imshow(S, cmap='bwr', vmin=-lim, vmax=lim)
  axes[2 + offset].title.set_text(r'$S_{yx}$')
  for (j, i), label in np.ndenumerate(S):
    if label != 0:
      axes[2 + offset].text(i, j, ("%.1f" % label), ha='center', va='center', fontsize=6)

  if sigma_xx is not None:
    offset += 1
    lim = np.maximum(np.abs(np.min(S_norm)), np.max(S_norm))
    axes[2 + offset].imshow(S_norm, cmap='bwr', vmin=-lim, vmax=lim)
    axes[2 + offset].title.set_text(r'$S$')
    axes[2 + offset].title.set_text(r'$\frac{S_{yx}}{S_{xx}}$')
    for (j, i), label in np.ndenumerate(S_norm):
      if label != 0:
        axes[2 + offset].text(i, j, ("%.1f" % label), ha='center', va='center', fontsize=6)

  lim = np.maximum(np.abs(np.min(V)), np.max(V))
  axes[3 + offset].imshow(V.T, cmap='bwr', vmin=-lim, vmax=lim)
  axes[3 + offset].title.set_text(r'$V^T$')

  for ax in list(axes):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

  plt.tight_layout()
  fig.savefig(name + '.png', format='png')


def plot_pathway_weights(weights, svds, plot_specs=None):
  for k, (svd, weight) in enumerate(zip(svds, weights)):
    U, S, V = svd
    S = np.diag(S)
    #     weight = weight.numpy()
    #     U, S, V, weight = U.numpy(), np.diag(S.numpy()), V.numpy(), weight.numpy()
    fig, axes = plt.subplots(1, 4)

    lim = np.maximum(np.abs(np.min(weight)), np.max(weight))
    im = axes[0].imshow(weight, cmap='bwr', vmin=-lim, vmax=lim)
    plt.colorbar(im)
    axes[0].title.set_text(r'$W$')

    lim = np.maximum(np.abs(np.min(U)), np.max(U))
    axes[1].imshow(U, cmap='bwr', vmin=-lim, vmax=lim)
    axes[1].title.set_text(r'$U$')

    lim = np.maximum(np.abs(np.min(S)), np.max(S))
    axes[2].imshow(S, cmap='bwr', vmin=-lim, vmax=lim)
    axes[2].title.set_text(r'$S$')

    lim = np.maximum(np.abs(np.min(V)), np.max(V))
    axes[3].imshow(V.T, cmap='bwr', vmin=-lim, vmax=lim)
    axes[3].title.set_text(r'$V^T$')
    for (j, i), label in np.ndenumerate(S):
      if label != 0:
        axes[2].text(i, j, ("%.1f" % label), ha='center', va='center', fontsize=6)

    #     title = ''.join([str(i) for i in comb])
    #     fig.suptitle(r'$\Sigma^{' + '{}'.format(title) + '}$', fontsize=14)

    #     fig2, axes2 = plt.subplots(1, len(S_))
    #     for ax, U_col, V_col in zip(axes2, U_.T, V_.T):
    #         outer = torch.outer(U_col, V_col).numpy()
    #         contrib = outer @ x.T.numpy()
    #         ax.imshow(contrib)

    for ax in list(axes):
      ax.xaxis.set_visible(False)
      ax.yaxis.set_visible(False)

    plt.tight_layout()

    if plot_specs is not None and plot_specs['save'] is True:
      fig.savefig(plot_specs['save_dir'] + 'weights_' + str(k) + '.png', format='png')