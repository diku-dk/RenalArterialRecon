from pygel3d import hmesh
from utils import *
from center_points import CenterPoints
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pygel3d import hmesh, gl_display as gl


if __name__ == '__main__':

    input_path = 'artery_mesh.obj'
    output_path = 'artery_centerline_raw'

    original_file = 'artery_seg.nii.gz'

    saved_vtk = 'artery_centerline_filtered.vtk'


    m = hmesh.load(input_path)

    print("vertices before simplification :", m.no_allocated_vertices())

    hmesh.close_holes(m)
    hmesh.triangulate(m)

    while m.no_allocated_vertices() > 2e5:
        hmesh.quadric_simplify(m, 0.5)

    m.cleanup()

    print("vertices after simplification :", m.no_allocated_vertices())
    

    from pygel3d import graph
    g = graph.from_mesh(m)
    s = graph.LS_skeleton(g)

    graph.save(output_path, s)



    # from pygel3d import jupyter_display as jd
    #
    # jd.set_export_mode(True)
    # jd.display(s)

    print(f'num of points orig = {len(s.positions())}')

    cp = CenterPoints(s, original_file,)

    print(f'num of points = {len(cp.valid_pos)}')

    print(f'len of points = {np.unique([len(i) for i in cp.adj_list], return_counts=True)}')

    cp.set_adj_list()
    cp.set_adj_matrix()
    cp.sanity_check()

    n_branches = cp.set_separate_branch(force_recompute=True)
    print(f'num of n_branches = {n_branches}')
    cp.remove_duplicate_branch()
    cp.set_non_visited_nodes()

    cp.label_bifur_point()
    cp.set_distance_transforms()
    cp.set_distance_interp()
    cp.set_branch_properties()
    cp.save_all_branch(path=saved_vtk)
