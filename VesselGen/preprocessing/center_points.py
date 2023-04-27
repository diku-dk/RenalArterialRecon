import json
import os.path
import pickle
from collections import defaultdict, OrderedDict

import pygel3d.gl_display
import pyvista
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from utils import *
from pygel3d import graph
from pygel3d import hmesh
import numpy as np

class CenterPoints:
    def __init__(self, s, original_file='1.nii.gz',
                 distance_file=None, save_root='./', to_RAS=False):

        self.s = s
        self.original_file = original_file
        self.orig_nifty = nib.load(original_file)
        self.save_root = save_root
        safe_mkdir(save_root)

        self.shape = self.orig_nifty.shape
        self.pos = s.positions()

        if to_RAS:
            self.pos = np.array([-self.pos[:, 0], -self.pos[:, 1], self.pos[:, 2]]).T


        valid_indices = np.where(~np.isnan(self.pos).any(axis=1))[0]
        valid_pos = self.pos[valid_indices, :]

        self.valid_indices = valid_indices
        self.valid_pos = valid_pos
        self.valid_pos_rounded = np.round(valid_pos).astype(np.uint16)

        self.get_remove_disconnected()

        self.n_points = len(self.valid_pos)

        self.adj_list = None
        self.adj_matrix = None
        self.global_dict = defaultdict(lambda: [])
        
        self.n_branches = 0
        self.distance_transform_img = None
        self.distance_interp = None
        self.length_list = None
        self.radius_list = None
        self.distance_file = distance_file

        self.disconnected_indices = []
        self.line_to_break_loop = []
        self.non_visited = []

        self.set_adj_list()

        self.cycles = []
        self.duplicate_branch = []

        self.val_radius = None

    def set_non_visited_nodes(self):
        visited = np.unique(np.hstack(self.global_dict.values()))
        visited = np.sort(visited)
        all_nodes = np.arange(self.n_points)
        non_visited = np.setdiff1d(all_nodes, visited)

        self.non_visited = non_visited

    def get_remove_disconnected(self):
        disconnected_indices = []

        for i in range(len(self.valid_pos)):
            orig_index = self.valid_indices[i]
            neighbor_list = [j for j in self.s.neighbors(orig_index)]
            if len(neighbor_list) == 0:
                disconnected_indices.append(self.valid_indices[i])

        self.valid_pos = np.delete(self.valid_pos, disconnected_indices, axis=0)
        self.valid_indices = np.delete(self.valid_indices, disconnected_indices, axis=0)
        self.valid_pos_rounded = np.delete(self.valid_pos_rounded, disconnected_indices, axis=0)

        self.disconnected_indices = disconnected_indices

    def remove_duplicate_branch(self):
        sets = [set(i) for i in (list(self.global_dict.values()))]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if sets[i] == sets[j]:
                    self.duplicate_branch.append(sets[i])
                    self.global_dict.pop(j, None)

    def set_adj_list(self):
        connectivity = []
        for i in range(len(self.valid_pos)):
            orig_index = self.valid_indices[i]
            neighbor_list = [np.where(self.valid_indices == j)[0][0] for j in self.s.neighbors(orig_index)]
            

            if len(neighbor_list) == 0:
                print(f'found node {i} with empty neighbors')

            connectivity.append(neighbor_list)

        self.adj_list = connectivity

    def save_points_inside_image(self, root=None, save=True):
        r, c, d = self.valid_pos_rounded.T
        img = np.zeros(self.orig_nifty.shape)
        img[r, c, d] = 1
        assert img.sum() == len(self.valid_pos_rounded)
        if root is None:
            nib.save(nib.Nifti1Image(img, self.orig_nifty.affine), 'points_1.nii.gz')

    def set_adj_matrix(self):
        assert self.adj_list is not None
        adj_matrix = adj_list_to_matrix(self.adj_list)
        self.adj_matrix = adj_matrix

    def save_adj_list(self, root):
        np.save(root, self.adj_list)

    def sanity_check(self):
        assert np.all(np.diag(self.adj_matrix) == 0)
        assert np.all(self.adj_matrix == self.adj_matrix.T)

    def set_separate_branch(self, force_recompute=False, begin_index=None):

        if begin_index is None:
            n_neighbor_list = np.array([len(i) for i in self.adj_list])
            begin_index = np.where(n_neighbor_list == 1)[0][0]
            begin_index = int(begin_index)

        if len(self.global_dict) > 0:
            if force_recompute:
                
                self.global_dict = defaultdict(lambda: [])

            else:
                print('global_dict already exists, set "global_dict" to True to recompute')
                return

        n_branches = self.label_branch(i=begin_index, branch_id=0)
        self.n_branches = n_branches


        return n_branches

    def save_separate_branch(self, root):
        assert 'json' in root
        
        
        with open(root, 'w') as file:
            json.dump(self.global_dict, file, separators=(',', ':'), indent=10)

    def label_branch(self, i, branch_id, visited=set(), incoming_node=None, cur_path=[]):

        

        
        self.global_dict[branch_id].append(i)

        if i in visited:
            if i in cur_path:
                self.cycles.append(cur_path)
            return branch_id

        if incoming_node is not None:
            assert incoming_node in self.adj_list[i]

        i_neighbors = [k for k in self.adj_list[i] if k != incoming_node]

        if len(i_neighbors) == 0:
            visited.add(i)

            assert i not in cur_path

            return branch_id

        elif len(i_neighbors) == 1:
            if i in cur_path:
                print(cur_path)
            visited.add(i)
            return self.label_branch(i_neighbors[0], branch_id, visited, incoming_node=i, cur_path=cur_path + [i])

        else:
            visited.add(i)
            for _, j in enumerate(i_neighbors):
                branch_id += 1

                
                self.global_dict[branch_id].append(i)

                branch_id = self.label_branch(j, branch_id, visited, incoming_node=i, cur_path=cur_path + [i])

            return branch_id

    def pyvisa_show(self):

        connectivity, _ = decompose_bifurcations(self.adj_list, )
        lines = np.hstack([np.ones((len(connectivity), 1), dtype=np.uint8) * 2, connectivity])
        lines = np.hstack(lines)

        mesh = pyvista.PolyData(self.valid_pos,
                                lines=lines
                                )

        mesh.plot()

    def set_distance_transforms(self):
        if self.distance_file is not None:
            distance_transform_img = nib.load(self.distance_file).get_fdata()

        else:
            img = self.orig_nifty.get_fdata()
            distance_transform_img = distance_transform_edt(img)
        self.distance_transform_img = distance_transform_img

        self.save_distance_transforms(os.path.join(self.save_root, 'distance.nii.gz'))

    def save_distance_transforms(self, root):
        nib.save(nib.Nifti1Image(self.distance_transform_img.astype(np.float32),
                                 self.orig_nifty.affine), root)

    def set_distance_interp(self):
        shape = self.orig_nifty.shape
        coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

        bg_val = 0
        im_intrps = RegularGridInterpolator(coords, self.distance_transform_img,
                                            method="linear",
                                            bounds_error=False,
                                            fill_value=bg_val)

        self.distance_interp = im_intrps

    def set_branch_properties(self):

        self.length_list, self.val_radius, self.radius_list = None, None, None

        val_radius = []
        for coords in self.valid_pos:
            radius = compute_radius(self.distance_interp, coords)
            val_radius.append(radius)

        self.val_radius = np.array(val_radius)

        length_list = []
        radius_list = []

        for i, points in enumerate(self.global_dict.values()):
            points = np.array(points)
            coords = self.valid_pos[points]
            length = compute_length(coords)
            
            radius = np.mean(self.val_radius[points])
            length_list.append(length)
            radius_list.append(radius)

        self.length_list = np.array(length_list)
        self.radius_list = np.array(radius_list)

    def label_end_point(self):
        endpoints = [i for i, j in enumerate(self.adj_list) if len(j) == 1]
        endpoints = np.array(endpoints)
        self.endpoints = endpoints

    def save_end_points(self, path='end_points.vtk'):
        path = os.path.join(self.save_root, path)

        points = np.empty(3)
        for i in self.endpoints:
            points = np.hstack([points, self.valid_pos[i]])

        mesh = pyvista.PolyData(points,
                                )
        mesh.save(path)

    def save_given_points(self, points, path='loop_points.vtk'):
        path = os.path.join(self.save_root, path)

        if len(points) == 0:
            return

        points = np.hstack(points)
        mesh = pyvista.PolyData(self.valid_pos[points],
                                )
        mesh.save(path)

    def save_given_branch(self, points, path='loop_points.vtk'):
        path = os.path.join(self.save_root, path)

        if len(points) == 0:
            return

        points = np.hstack(points)
        lines = [len(points)] + list(points)
        mesh = pyvista.PolyData(self.valid_pos[points],
                                lines=lines
                                )
        mesh.save(path)

    def label_end_branch(self):
        end_branch = {}
        for k, v in self.global_dict.items():
            if np.any(np.isin(self.endpoints, np.array(v))):
                end_branch[k] = v
        self.end_branch = end_branch

    def save_end_branch(self, path='end_branch.vtk'):
        path = os.path.join(self.save_root, path)

        points = np.empty(3)
        for i in self.endpoints:
            points = np.hstack([points, self.valid_pos[i]])

        items = self.end_branch.values()

        lines = []
        for i in items:
            cur_line = [len(i)] + i
            lines = lines + cur_line
        lines = np.array(lines)

        mesh = pyvista.PolyData(self.valid_pos,
                                lines=lines
                                )
        mesh.save(path)

    def save_all_branch(self, path='all_branch.vtk', seperate_edge=False):
        path = os.path.join(self.save_root, path)
        vertices = self.valid_pos
        lines = self.get_line_for_pv()
        mesh = pyvista.PolyData(vertices,
                                lines=lines
                                )

        if self.val_radius is not None:
            mesh.point_data['radius'] = self.val_radius

        mesh.save(path)


    def filter_radius(self, level=7):
        global_dict = defaultdict(lambda: [])

        for i, points in enumerate(self.global_dict.values()):
            if self.radius_list[i] < level:
                global_dict[i] = points

        self.global_dict = global_dict

    def save_branch_from_global_dict(self, path='all_branch.vtk'):
        path = os.path.join(self.save_root, path)
        vertices = self.valid_pos
        lines = self.get_line_from_global_dict_for_pv()

        
        
        
        


        mesh = pyvista.PolyData(vertices,
                                lines=lines
                                )

        if self.val_radius is not None:
            mesh.cell_data['radius'] = self.radius_list

        if (self.length_list is not None) and len(self.length_list) == mesh.n_lines:
            mesh.cell_data['length'] = self.length_list

        mesh.save(path)




    def label_bifur_point(self):
        bifur_point = [i for i, j in enumerate(self.adj_list) if len(j) > 2]
        bifur_point = np.array(bifur_point)
        self.bifur_point = bifur_point

    def get_line_for_pv(self):
        connectivity_final, _ = decompose_bifurcations(self.adj_list)
        lines = np.hstack([np.ones((len(connectivity_final), 1), dtype=np.uint8) * 2,
                           connectivity_final])
        lines = np.hstack(lines)
        return lines

    def get_line_from_global_dict_for_pv(self):
        lines = []
        for i, points in enumerate(self.global_dict.values()):
            lines = lines + [len(points)] + list(points)

        lines = np.array(lines)

        return lines

    def get_line_from_separate_branch(self, end_branch=False):
        lines = []
        items = self.end_branch.items() if end_branch else self.global_dict.items()
        for _, i in items:
            cur_line = [len(i)] + i
            lines = lines + cur_line

        lines = np.array(lines)
        return lines

    def find_cycle(self, v, visited, parent, label_cycles=False):
        visited[v] = True

        for i in self.adj_list[v]:
            if not visited[i]:
                if self._isCyclicUtil(i, visited, v, label_cycles):
                    if label_cycles:
                        self.line_to_break_loop.append([v, i])

                    return True

            
            
            else:
                if i != parent:
                    self.line_to_break_loop.append([parent, i])
                    return True
                else:
                    continue

        return False

    def _isCyclicUtil(self, v, visited, parent, label_cycles=False):
        visited[v] = True

        for i in self.adj_list[v]:
            if not visited[i]:
                if self._isCyclicUtil(i, visited, v, label_cycles):
                    if label_cycles:
                        self.line_to_break_loop.append([v, i])

                    return True

            
            
            else:
                if i != parent:
                    self.line_to_break_loop.append([parent, i])
                    return True
                else:
                    continue

        return False

    def isTree(self, begin_index=None, label_abnormal=False, label_cycles=False):

        non_visited = []
        cycles = []

        n_nodes = len(self.valid_pos)
        visited = [False] * n_nodes

        if begin_index is None:
            n_neighbor_list = np.array([len(i) for i in self.adj_list])
            begin_index = np.where(n_neighbor_list == 1)[0][0]
            begin_index = int(begin_index)

        if self._isCyclicUtil(begin_index, visited, None, label_cycles=label_cycles):
            if not label_cycles:
                return False

        for i, v in enumerate(visited):
            if not v:
                if not label_abnormal:
                    return False
                else:
                    non_visited.append(i)

        if len(non_visited) > 0 or len(cycles) > 0:
            self.non_visited = non_visited
            self.cycles = cycles

            return False

        return True

    def save_object(self, filename):
        with open(filename, 'wb') as out:  
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    def dfs_cycle(self, u, p, color: list,
                  mark: list, par: list):

        if color[u] == 2:
            return

        if color[u] == 1:

            self.cyclenumber += 1
            cur = p
            mark[cur] = self.cyclenumber

            while cur != u:
                cur = par[cur]
                mark[cur] = self.cyclenumber

            return

        par[u] = p

        
        color[u] = 1

        
        for v in self.adj_list[u]:

            
            if v == par[u]:
                continue
            self.dfs_cycle(v, u, color, mark, par)

        
        color[u] = 2

    
    def printCycles(self):

        self.cyclenumber = 0

        edges = sum([len(i) for i in self.adj_list])

        

        N = 2 * edges

        self.cycles = [[] for _ in range(N)]

        color = [0] * N
        par = [-1] * N
        mark = [0] * N

        
        self.dfs_cycle(0, -1, color, mark, par)

        
        
        for i in range(1, edges + 1):
            if mark[i] != 0:
                self.cycles[mark[i]].append(i)

        cycles = defaultdict(lambda: [])
        for i in range(1, self.cyclenumber + 1):

            
            print("Cycle Number %d:" % i, end=" ")
            for x in self.cycles[i]:
                print(x, end=" ")
                cycles[i].append(x)

            print()

        return cycles

    def filter_too_many_branches(self, save_level=10):

        bifur_points = [i for i, j in enumerate(self.adj_list) if len(j) > save_level]
        bifur_points = np.array(bifur_points)

        all_indices = []
        for bifur_point in bifur_points:
            
            neighbor_branches = []
            indices = []
            for i, points in self.global_dict.items():
                points = np.array(points)
                if bifur_point in points:
                    neighbor_branches.append(points)
                    indices.append(i)
            n = len(neighbor_branches)
            delete_branches = np.argsort([len(i) for i in neighbor_branches])[:n - 5]
            indices = np.array(indices)
            indices = indices[delete_branches]
            all_indices += list(indices)

        for j in all_indices:
            self.global_dict.pop(j, None)

    def get_adj_list_from_global_dict(self):
        n = self.n_points
        adj_matrix = np.zeros((n, n), dtype=np.uint8)

        for points in self.global_dict.values():
            if len(points) == 1:
                continue
            for i in range(len(points)-1):
                adj_matrix[points[i], points[i+1]] = 1
                adj_matrix[points[i+1], points[i]] = 1

        self.adj_matrix = adj_matrix
        self.adj_list = adj_matrix_to_list(adj_matrix, keep_dim=True)

    def save_points_to_img(self, path):
        path = os.path.join(self.save_root, path)

        x, y, z = self.valid_pos_rounded.T
        img = np.zeros(self.orig_nifty.shape, dtype=np.uint8)
        img[x, y, z] = 1

        nib.save(nib.Nifti1Image(img,
                                 self.orig_nifty.affine), path)


    def plot_adj_neighbor_distribution(self):
        n_neighbors, count = np.unique([len(i) for i in self.adj_list], return_counts=True)
        plt.bar(n_neighbors, count)
        print(count)
        plt.xlabel('number of neighbors')
        plt.ylabel('count')
        
        
        for index, data in zip(n_neighbors, count):
            plt.text(x=index, y=data + 1, s=f"{data}", fontdict=dict(fontsize=10),
                     va='center'
                     )
        plt.tight_layout()
        plt.show()




if __name__ == '__main__':

    file = 'Rat_37_zoom_seg_clean_254'
    m = hmesh.load(file)
    a = pygel3d.gl_display.Viewer()
    a.display(m)

