import sys

import nibabel as nib
import numpy as np
import os
root_folder = os.getcwd()


from skimage.morphology import ball
import scipy.ndimage as nd

def binary_erode_label(img, radius=1, bg_val=0):

    img = img.astype(np.uint8)

    max_radius = 11

    indiv_radius = radius // max_radius
    iter = radius // indiv_radius
    remain = radius % max_radius

    print(f'erode {radius}, with {iter} iters and {indiv_radius} indiv radius')

    res = nd.binary_erosion(img, ball(indiv_radius), iterations=iter)

    res = res.astype('uint8')

    if remain >0:
        res = nd.binary_erosion(res, ball(remain))
        res = res.astype('uint8')

    return res


def filter_label_from_root(shape, root_pos, radius):

    h, w, d = shape
    
    Y, X, Z = np.ogrid[:h, :w, :d]
    dist_field = (X - root_pos[0])**2/radius**2 + (Y-root_pos[1])**2/radius**2 + (Z-root_pos[2])**2/radius**2
    mask = dist_field <= 1
    return mask


def filter_ellipsoid_from_root(shape, root_pos, radius):

    h, w, d = shape
    a, b, c = radius
    

    Y, X, Z = np.ogrid[:h, :w, :d]
    
    dist_field = (X - root_pos[0])**2/a**2 + (Y-root_pos[1])**2/b**2 + (Z-root_pos[2])**2/c**2
    
    mask = dist_field <= 1
    return mask


def ccd_largest_part(img):

    from scipy.ndimage import label
    import numpy as np

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



if __name__ == '__main__':


    path = '/Users/px/Downloads/final_kidney_Sep/cropped_whole_struct.nii.gz'
    save_path = '/Users/px/Downloads/final_kidney_Sep/estimated_cortex.nii.gz'

    radius = 89
    dist = 250

    root_loc = np.array([585.18, 206.25, 641])

    root_loc = np.array([520.18, 223, 560])

    label = nib.load(path)
    label_data = label.get_fdata().astype('uint8')
    affine = label.affine

    res = label - binary_erode_label(label, radius)
    res = res.astype('uint8')

    mask = filter_label_from_root(label.shape, [root_loc[1], root_loc[0], root_loc[2]], dist)

    filtered_res = np.logical_and(res, np.logical_not(mask))

    filtered_res = ccd_largest_part(filtered_res)

    filtered_res = nd.binary_erosion(filtered_res, ball(5), iterations=4)
    filtered_res = ccd_largest_part(filtered_res)
    filtered_res = nd.binary_dilation(filtered_res, ball(5), iterations=4)
    filtered_res = ccd_largest_part(filtered_res)

    nib.save(nib.Nifti1Image(filtered_res.astype('uint8'), affine),save_path)

    sys.exit()