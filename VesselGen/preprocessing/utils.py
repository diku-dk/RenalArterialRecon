import matplotlib.pyplot as plt
import mcubes
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from skimage import measure
import os

def marching_cubes(res, name='vessel', level=1, spacing=(0.3, 0.3, 0.3), gradient_direction='descent', offset=0):

    verts, faces, normals, values = measure.marching_cubes_lewiner(volume=res,
                                                                   level=level,
                                                                   spacing=spacing,
                                                                   gradient_direction=gradient_direction
                                                                   )

    mcubes.export_obj(verts-offset, faces, name+'.obj')

def safe_mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)

def max_projection(img, axis=0):
    return np.max(img, axis=axis)


def mean_projection(img, axis=0):
    return np.mean(img, axis=axis)


def center_projection(img, axis=0):
    m, n, k = img.shape

    assert axis in (0, 1, 2)

    if axis == 0:
        return img[m // 2, :, :]
    elif axis == 1:
        return img[:, n // 2, :]
    else:
        return img[:, :, k // 2]



def points_to_sparse_img(points, orig_nifty, path='center_points_img.nii.gz'):
    x, y, z = points
    img = np.zeros(orig_nifty.shape, dtype=np.uint8)
    img[x, y, z] = 1
    nib.save(nib.Nifti1Image(img,
                             orig_nifty.affine), path)


def save_nifty(img, affine, path):
    nib.save(nib.Nifti1Image(img, affine), path)


def decompose_bifurcations(connectivity, include_self=True):
    connectivity_final = []
    bifurcation_indices = []
    for i, neighbors in enumerate(connectivity):
        if len(neighbors) == 0:
            continue
        for n in neighbors:
            if include_self:
                connectivity_final.append([i, n])
            else:
                connectivity_final.append([n])
        if len(neighbors) > 2:
            bifurcation_indices.append(i)

    connectivity_final = np.array(connectivity_final)
    bifurcation_indices = np.array(bifurcation_indices)

    return connectivity_final, bifurcation_indices


def get_line_for_pv_from_adj_list(adj_list):
    connectivity_final, _ = decompose_bifurcations(adj_list)
    lines = np.hstack([np.ones((len(connectivity_final), 1), dtype=np.uint8) * 2,
                       connectivity_final])
    lines = np.hstack(lines)
    return lines

def get_adj_matrix_from_pv(lines):
    n_points = np.max(lines) + 1
    adj_matrix = np.zeros((n_points, n_points), dtype=np.uint8)
    i = 1
    j = 2
    while j < len(lines):
        adj_matrix[lines[i], lines[j]] = 1
        i += 3
        j += 3

    return adj_matrix

def adj_list_to_matrix(adj_list):
    n = len(adj_list)
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in adj_list[i]:
            adj_matrix[i, j] = 1
    return adj_matrix


def adj_matrix_to_list(adj_matrix, include_self=False, keep_dim=False):
    adj_list = []
    for i, cur_nb_indices in enumerate(adj_matrix):
        neighbors = list(np.where(cur_nb_indices)[0])
        if len(neighbors) > 0 or keep_dim:
            neighbors = [i] + neighbors if include_self else neighbors
            adj_list.append(neighbors)
    return adj_list


def compute_length(coords):
    diffs = np.diff(coords, axis=0)
    diffs = diffs ** 2
    diffs = diffs.sum(axis=1)
    diffs = np.sqrt(diffs)
    final_length = np.sum(diffs)
    return final_length


def compute_radius(eucl_trans_grid, coords):
    radius = eucl_trans_grid(coords)
    final_radius = np.mean(radius)
    return final_radius


def ccd_largest_part(img):
    res = np.zeros(img.shape)

    for i in np.unique(img):
        if i == 0: continue

        labels_out, num_labels = label((img == i).astype('uint8'))

        lab_list, lab_count = np.unique(labels_out, return_counts=True)

        if lab_list[0] == 0:
            lab_list = lab_list[1:]
            lab_count = lab_count[1:]

        largest_ind = np.argmax(lab_count)
        lab = lab_list[largest_ind]

        res += (i * (labels_out == lab)).astype(np.uint8)

    res = res.astype('uint8')

    return res



