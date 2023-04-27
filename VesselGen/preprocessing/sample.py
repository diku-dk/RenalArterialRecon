
import igl
import os
root_folder = os.getcwd()
import numpy as np
import poisson_disc as pd
import nibabel as nib
import pyvista

if __name__ == '__main__':

    file = 'estimated_cortex.nii.gz'
    img = nib.load(file).get_fdata()
    volume = np.sum(img)
    dims3d = np.array(img.shape)
    box_volume = np.prod(dims3d + 1)
    n = 30000

    box_n = int(n * box_volume/volume)
    radius = np.mean(dims3d/np.power(box_n, 1/3))
    points3d = pd.Bridson_sampling(dims3d, radius=radius*0.82, k=100)

    print(points3d.shape)
    a = points3d.astype(int)
    a = np.minimum(a, np.array(img.shape) - 1)
    valid_idx = [i for i in range(len(a)) if img[a[i][0], a[i][1], a[i][2]]]

    valid_points = points3d[valid_idx]
    print(valid_points.shape)


    pyvista.PolyData(valid_points).save('sampled_terminal_nodes.vtk')


