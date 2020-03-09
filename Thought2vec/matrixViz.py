import matplotlib.pyplot as plt
import numpy as np

def visualize_weights(matrix, contrast = None, ret = False, title = None):
    if title is None:
        title = 'Weight Matrix Visualization'
        
    if contrast is not None:
        matrix -= np.mean(matrix)
        matrix = matrix / (contrast * (1 + np.abs(matrix / contrast)))

    max_size = np.max(np.abs(matrix))

    num_dims = matrix.shape[1]
    fig, ax = plt.subplots()
    fig.set_size_inches(h = 8, w = 16)
    fig.suptitle(title, fontsize = 24)
    fig.subplots_adjust(top = 0.93, bottom = 0.1)

    plot = ax.pcolormesh(matrix, cmap='PiYG', vmin=-max_size, vmax=max_size, snap=True)

    ax.set_xlabel('Dimension', fontsize = 20)
    ax.set_xticks(np.arange(0, num_dims + 1, 1))
    ax.set_xticks(np.arange(0.5, num_dims, 1), minor = True)
    ax.set_xticklabels([])
    ax.set_xticklabels(np.arange(1, 1 + num_dims), minor = True)

    ax.set_ylabel('Question', fontsize = 20)
    ax.set_yticks(np.linspace(0, 150, 31))
    ax.set_yticklabels([])
    ax.set_yticks(np.linspace(3, 147, 30), minor = True)
    ax.set_yticklabels(np.arange(1, 31), minor = True)

    ax.tick_params(axis = 'y', which = 'minor', length = 0, labelsize = 12)
    ax.tick_params(axis = 'x', which='minor', length=0, labelsize=12)

    ax.grid(axis='y', which='major', color='black',linewidth=0.5, linestyle='dashed')
    ax.grid(axis='x', which='major', color='black',linewidth=1)

    cbar = fig.colorbar(plot, ax = ax, pad = 0.01, extend = 'neither')
    cbar.ax.tick_params(labelsize=14)

    fig.canvas.set_window_title(title)

    if ret:
        return fig
        
    plt.show()

def visualize_weights_expanded(matrix, contrast=None, Q=30, title = None):
    if title is None:
        title = 'Weight Matrix Visualization'
    if contrast is not None:
        matrix -= np.mean(matrix)
        matrix = matrix / (contrast * (1 + np.abs(matrix / contrast)))

    max_size = np.max(np.abs(matrix))

    num_responses, num_dims = matrix.shape
    num_subcolumns = num_responses // Q
    expanded_matrix = (np.eye(num_subcolumns)[:, :, None] * matrix.reshape(Q, num_subcolumns, 1, num_dims)).transpose(0, 1, 3, 2).reshape(num_responses, num_subcolumns * num_dims)

    fig, ax = plt.subplots()
    fig.set_size_inches(h = 8, w = 16)
    fig.suptitle(title, fontsize=24)
    fig.subplots_adjust(top=0.93, bottom=0.1)

    plot = ax.pcolormesh(expanded_matrix, cmap='PiYG', vmin = -max_size, vmax = max_size, snap = True)

    ax.set_ylabel('Question', fontsize=20)

    ax.set_yticks(np.linspace(0, 150, 31))
    ax.set_yticklabels(np.arange(1, 31), verticalalignment = 'bottom')

    #ax.set_yticks(np.linspace(0, 180, 181), minor = True)
    #ax.set_yticklabels([], minor=True)

    ax.set_xlabel('Dimension', fontsize=20)

    ax.set_xticks(np.arange(0, num_subcolumns * num_dims + 1, num_subcolumns))
    ax.set_xticklabels(np.arange(1, 1 + num_dims), horizontalalignment = 'left')

    ax.set_xticks(np.arange(num_subcolumns * num_dims + 1), minor=True)
    ax.set_xticklabels([], minor=True)

    ax.tick_params(axis='y', which='major', length=3, labelsize=12)
    ax.tick_params(axis='x', which='major', length=6, labelsize=12)

    ax.grid(axis='y', which='major', color='black', linewidth=0.5, linestyle = 'dashed')
    ax.grid(axis='x', which='major', color = 'black', linewidth = 1)
    ax.grid(axis='x', which='minor', color='gray', linewidth=0.5, linestyle = 'dotted')

    cbar = fig.colorbar(plot, ax=ax, pad=0.01, extend='neither')
    cbar.ax.tick_params(labelsize=14)

    fig.canvas.set_window_title(title)
    plt.show()