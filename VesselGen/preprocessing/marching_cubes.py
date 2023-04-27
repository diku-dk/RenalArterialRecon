from skimage import measure
import nibabel as nib


import mcubes


def marching_cubes(res, name='vessel', level=0.9, spacing=1, method=None, smooth=False, show=False):
    if smooth:
        res = mcubes.smooth(res)

    if method == 'skimage':
        verts, faces, normals, values = measure.marching_cubes(volume=res,
                                                               
                                                               
                                                                       )
    else:
        verts, faces = mcubes.marching_cubes(res, 0)

    mcubes.export_obj(verts, faces, name+'.obj')


if __name__ == '__main__':

    img_path = 'artery_seg.nii.gz'
    save_name = 'artery_mesh'

    img = nib.load(img_path)
    data = img.get_fdata()
    affine_func = img.affine

    marching_cubes(data, name=save_name, level=0.9, smooth=True, method='mcube', show=False)
