import sys

import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
from scipy.ndimage import label
from skimage import measure
import os
import networkx as nx
import copy
import pyvista

class VtkNetwork:

    def __init__(self, pt_file=None, root_loc=None, vsize=22.6):

        

        self.pt_file = pt_file
        self.root_loc = root_loc
        self.vsize = vsize

    def build(self):
        mesh = pyvista.read(self.pt_file,
                            
                                      )
        points = mesh.points


        line = mesh.lines

        if 'radius' in mesh.point_data:
            node_radius = mesh.point_data['radius']
        elif 'node_radius' in mesh.point_data:
            node_radius = mesh.point_data['node_radius']
        elif 'radius' in mesh.cell_data:
            node_radius = None

        self.tree = nx.Graph()

        for i, p in enumerate(points):
            self.tree.add_node(i, loc=np.array(p), fixed=False, root=False,
                               node_radius=node_radius[i] if node_radius is not None else None
                               )

        i = 1
        while i < len(line):
            node1, node2 = line[i], line[i + 1]
            dis = np.linalg.norm(self.tree.nodes[node1]['loc'] - self.tree.nodes[node2]['loc']
                                 ) * self.vsize
            if node_radius is not None:
                radius = np.mean([node_radius[node1], node_radius[node2]])
            else:
                radius = mesh.cell_data['radius'][i // 3]
            self.tree.add_edge(node1, node2, radius=radius, length=dis)
            i += 3

        for edge in self.tree.edges:
            a, b = edge
            self.tree[a][b]['radius_neg'] = - self.tree[a][b]['radius']

        if self.root_loc is not None:
            self.root = self.find_n_with_coord(self.root_loc)
            self.tree.nodes[self.root]['fixed'] = True
            self.tree.nodes[self.root]['root'] = True

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

    def mst(self):

        tree = nx.minimum_spanning_tree(self.tree, weight='radius_neg')
        root = self.root

        self.tree = tree
        self.root = self.find_n_with_coord(self.root_loc)

        self.tree.nodes[root]['fixed'] = True
        self.tree.nodes[root]['root'] = True

        self.update_order()
        self.to_directed()


        

    def to_directed(self):
        def bfs_to_direct(node):
            neighbors = list(self.tree.neighbors(node))

            if self.tree.nodes[node]['root']:
                children = neighbors
            else:
                neighbor_order = np.array([self.tree.nodes[n]['level'] for n in neighbors])
                if -1 in neighbor_order:
                    root_idx = np.where(neighbor_order == -1)[0][0]
                else:
                    root_idx = np.argmax(neighbor_order)

                root_idx = neighbors[root_idx]

                children = [i for i in neighbors if i != root_idx]

            if len(children) == 0:
                return

            for child in children:
                attrs = self.tree[node][child]
                self.di_tree.add_edge(node, child, **attrs)

            for child in children:
                bfs_to_direct(child)

        self.di_tree = nx.DiGraph()

        for n in self.tree.nodes:
            attrs = self.tree.nodes[n]
            self.di_tree.add_node(n, **attrs)

        root = [i for i in self.tree.nodes if self.tree.nodes[i]['root']][0]

        bfs_to_direct(root)

    def get_depth(self, root=None):
        if root is None:
            all_nodes = np.array([i for i in self.tree.nodes])

            orders = np.array([self.tree.nodes[n]['level'] for n in all_nodes])

            root = all_nodes[np.where(orders == -1)[0][0]]

            neighbors = np.array(list(self.tree.neighbors(root)))

            depth = 1 + max([self.get_depth(i) for i in neighbors])
            self.tree.nodes[root]['depth'] = depth
            return depth

        neighbors = np.array(list(self.tree.neighbors(root)))
        if len(neighbors) == 1:
            self.tree.nodes[root]['depth'] = 1
            return 1
        else:
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            if -1 in neighbor_orders:
                root_idx = np.where(neighbor_orders == -1)[0][0]
            else:
                root_idx = np.argmax(neighbor_orders)

            depth = 1 + max([self.get_depth(self.tree, i) for i in neighbors if i != neighbors[root_idx]])
            self.tree.nodes[root]['depth'] = depth
            return depth

    def save(self, file='sanity_check3.vtk'):

        for i in self.tree.nodes:
            self.tree.nodes[i]['ind'] = i

        nodes_inds = self.tree.nodes
        indices = sorted(nodes_inds)

        

        points = [nodes_inds[i]['loc'] for i in indices]
        points = np.array(points)

        edges_inds = self.tree.edges
        edges_inds = np.array(edges_inds)

        radius_list = [self.tree[n1][n2]['radius'] for n1, n2 in edges_inds]
        radius_list = np.array(radius_list)

        if np.max(edges_inds) >= len(nodes_inds):
            map = dict(zip(indices, np.arange(len(indices))))
            edges_inds_new = np.vectorize(map.get)(edges_inds)
            
        else:
            edges_inds_new = edges_inds

        lines = np.hstack([np.ones((len(edges_inds_new), 1), dtype=np.uint8) * 2, edges_inds_new])
        lines = np.hstack(lines)
        mesh = pyvista.PolyData(points,
                                lines=lines
                                )

        for key in self.tree[edges_inds[0][0]][edges_inds[0][1]]:
            edge_feature = [self.tree[n1][n2][key] for n1, n2 in edges_inds]
            edge_feature = np.array(edge_feature)

            assert len(edge_feature) > 0 and len(edge_feature) == mesh.n_lines
            mesh.cell_data[key] = edge_feature

        for key in self.tree.nodes[indices[0]]:
            if key in ('loc', 'fixed', 'root'):
                continue
            node_feature = [self.tree.nodes[i][key] for i in indices]
            assert len(node_feature) > 0 and len(node_feature) == mesh.n_points
            mesh.point_data[key] = np.array(node_feature)

        mesh.save(file)

        

    def update_order(self, mode='level'):
        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 and not self.tree.nodes[n]['fixed'] else 0
            if self.tree.nodes[n]['fixed']:
                self.tree.nodes[n][mode] = -1
        count_no_label = len(self.tree.nodes)
        cur_order = 1
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
                if -1 in neighbor_orders and np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue
                if -1 not in neighbor_orders and np.count_nonzero(neighbor_orders == 0) > 1:
                    continue
                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)


                self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def update_final_order(self, mode='HS'):
        """
        Update order when no further operations will be applied.
        """

        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] == 1 else 0
        count_no_label = len(self.tree.nodes)
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                if np.count_nonzero(neighbor_orders == 0) > 1:  
                    continue
                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)
                self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1
            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

        node = [n for n in self.tree.nodes if self.tree.nodes[n][mode] == -1][0]

        neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
        max_order = np.max(neighbor_orders)
        max_count = np.count_nonzero(neighbor_orders == max_order)
        self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

    def prune_by_distance_to_end(self, thresh=5):

        nodes_to_delete = []

        neighbor_nums = np.array([len(list(self.tree.neighbors(i))) for i in self.tree.nodes])
        all_nodes = np.array([i for i in self.tree.nodes])
        all_node_to_check = all_nodes

        for node in all_node_to_check:

            neighbors = list(self.tree.neighbors(node))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)

            if root == -1 or self.tree.nodes[node]['level'] == -1:
                continue

            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])

            children_dist = self.max_children_distances(node)

            if children_dist < thresh:
                nodes_to_delete.append(node)

        nodes_to_delete = np.unique(nodes_to_delete)
        self.tree.remove_nodes_from(nodes_to_delete)

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

        return self.tree

    def prune_too_dense(self, thresh=5):

        nodes_to_delete = []

        neighbor_nums = np.array([len(list(self.tree.neighbors(i))) for i in self.tree.nodes])
        all_nodes = np.array([i for i in self.tree.nodes])
        indices = np.where(neighbor_nums > thresh)[0]

        all_node_to_check = all_nodes[indices]

        for node in all_node_to_check:
            neighbors = list(self.tree.neighbors(node))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)

            if root == -1 or self.tree.nodes[node]['level'] == -1:
                continue

            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])

            children_dist = [self.sum_children_distances(i) for i in children]

            children_to_delete = children[np.argsort(children_dist)[:-(thresh - 1)]]

            nodes_to_delete += list(children_to_delete)

        nodes_to_delete = np.unique(nodes_to_delete)
        self.tree.remove_nodes_from(nodes_to_delete)

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

        return self.tree

    def prune_asymmetric(self, thresh=0.2):

        nodes_to_delete = []

        neighbor_nums = np.array([len(list(self.tree.neighbors(i))) for i in self.tree.nodes])
        all_nodes = np.array([i for i in self.tree.nodes])
        indices = np.where(neighbor_nums > 2)[0]

        all_node_to_check = all_nodes[indices]

        for node in all_node_to_check:

            neighbors = list(self.tree.neighbors(node))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)

            if root == -1 or self.tree.nodes[node]['level'] == -1 or len(neighbors) <= 2:
                continue

            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])

            children_dist = np.array([self.max_children_distances(i) for i in children])
            orders = np.argsort(children_dist)
            if children_dist[orders[0]]/children_dist[orders[1]] <= thresh:
                nodes_to_delete.append(children[orders[0]])

        nodes_to_delete = np.unique(nodes_to_delete)
        self.tree.remove_nodes_from(nodes_to_delete)

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

        return self.tree

    def prune_too_close(self):

        all_nodes = np.array([i for i in self.tree.nodes])

        orders = np.array([self.tree.nodes[n]['level'] for n in all_nodes])

        root = all_nodes[np.where(orders == -1)[0][0]]
        neighbors = np.array(list(self.tree.neighbors(root)))

        orders = np.array([self.tree.nodes[i]['level'] for i in neighbors])
        orders = np.argsort(orders)

        

        self.tree.remove_nodes_from(neighbors[orders[:-1]])

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

        root = self.find_n_with_coord(self.root_loc)
        self.tree.nodes[root]['fixed'] = True

    def prune_too_thin(self, thresh=26):

        all_nodes = np.array([i for i in self.tree.nodes])

        for (i, j) in self.tree.edges:
            if self.tree[i][j]['radius'] < thresh:
                self.tree.remove_edge(i, j)

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))


    def sum_children_distances(self, n):

        neighbors = np.array(list(self.tree.neighbors(n)))
        neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
        root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
        children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])

        root = neighbors[np.where(neighbor_orders == root)[0][0]]
        if len(neighbors) == 1:
            return np.linalg.norm(self.tree.nodes[root]['loc'] - self.tree.nodes[n]['loc'])

        cur = 0

        for i in children:
            cur += self.sum_children_distances(i)

        return cur

    def label_dist_from_root(self, node=None):
        if node is None:
            node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']][0]
            self.tree.nodes[node]['sub_length_root'] = 0
            children = np.array(list(self.tree.neighbors(node)))
        else:
            neighbors = np.array(list(self.tree.neighbors(node)))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])
            root = neighbors[np.where(neighbor_orders == root)[0][0]]
            self.tree.nodes[node]['sub_length_root'] = self.tree.nodes[root]['sub_length_root'] + np.linalg.norm(
                self.tree.nodes[root]['loc'] - self.tree.nodes[node]['loc'])

        for i in children:
            self.label_dist_from_root(i)

    def label_dist_to_end(self, node=None):
        if node is None:
            node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']][0]
            self.tree.nodes[node]['sub_length_end'] = 0
            children = np.array(list(self.tree.neighbors(node)))
        else:
            neighbors = np.array(list(self.tree.neighbors(node)))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])
            root = neighbors[np.where(neighbor_orders == root)[0][0]]
            self.tree.nodes[node]['sub_length_root'] = self.tree.nodes[root]['sub_length_root'] + np.linalg.norm(
                self.tree.nodes[root]['loc'] - self.tree.nodes[node]['loc'])

        for i in children:
            self.label_dist_from_root(i)

    def prune_dist_from_root(self, dist=400):

        nodes_to_delete = []

        for edge in list(self.tree.edges):
            a, b = edge
            if self.tree.nodes[a]['level'] < self.tree.nodes[b]['level']:
                a, b = b, a

            if self.tree.nodes[a]['sub_length_root'] > dist and self.tree.nodes[b]['sub_length_root'] > dist:
                nodes_to_delete.append(b)
            elif self.tree.nodes[a]['sub_length_root'] <= dist and self.tree.nodes[b]['sub_length_root'] <= dist:
                continue
            else:
                exceed = dist - self.tree.nodes[a]['sub_length_root']
                self.tree.nodes[b]['loc'] = self.tree.nodes[a]['loc'] + \
                                            exceed * (self.tree.nodes[b]['loc'] -
                                                      self.tree.nodes[a]['loc']) / np.linalg.norm(
                    self.tree.nodes[b]['loc'] - self.tree.nodes[a]['loc'])

        self.tree.remove_nodes_from(nodes_to_delete)

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

    def max_children_distances(self, n):

        neighbors = np.array(list(self.tree.neighbors(n)))
        neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
        root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
        children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])

        root = neighbors[np.where(neighbor_orders == root)[0][0]]
        if len(neighbors) == 1:
            return np.linalg.norm(self.tree.nodes[root]['loc'] - self.tree.nodes[n]['loc'])

        cur = np.max([self.max_children_distances(i) for i in children]) + np.linalg.norm(
            self.tree.nodes[root]['loc'] - self.tree.nodes[n]['loc'])

        return cur

    def save_results(self, work_dir):
        """
        Stores the resulting network structure.
        """

        if not os.path.exists(work_dir):
            os.mkdir(work_dir)

        coord_file = os.path.join(work_dir, 'test_1_result_coords.npy')
        connection_file = os.path.join(work_dir, 'test_1_result_connections.npy')
        radius_file = os.path.join(work_dir, 'test_1_result_radii.npy')
        order_file = os.path.join(work_dir, 'test_1_result_HS_order.npy')
        level_file = os.path.join(work_dir, 'test_1_result_level_order.npy')

        nodes = dict()
        coords = list()
        connections = list()
        radii = list()
        order = list()
        l_order = list()
        
        

        for edge in list(self.tree.edges):
            node1, node2 = edge
            for node in edge:
                if not node in nodes:
                    nodes[node] = len(coords)
                    coords.append(self.tree.nodes[node]['loc'])
                    order.append(self.tree.nodes[node]['HS'])
                    l_order.append(self.tree.nodes[node]['level'])
            connections.append([nodes[node1], nodes[node2]])
            radii.append(abs(self.tree[node1][node2]['radius']))

        np.save(coord_file, coords)
        np.save(connection_file, connections)
        np.save(radius_file, radii)
        print("Save coords, edges and radius.")
        np.save(order_file, order)
        np.save(level_file, l_order)
        print("Save orders.")

    def find_n_with_coord(self, coords=np.array([1, 2, 3])):
        all_nodes = [i for i in self.tree.nodes]
        all_loc = np.array([self.tree.nodes[i]['loc'] for i in all_nodes])
        all_dist = np.sum((all_loc - np.array(coords)) ** 2, axis=1)
        return all_nodes[np.argmin(all_dist)]

    def remove_intermediate(self):

        max_orders = np.max(np.array([self.tree.nodes[n]['level'] for n in self.tree.nodes]))
        root = [n for n in self.tree.nodes if self.tree.nodes[n]['level'] == max_orders]
        assert len(root) == 1
        root = root[0]
        all_nodes = [n for n in self.tree.nodes]
        for n in all_nodes:
            neighbors = list(self.tree.neighbors(n))
            if len(neighbors) == 2 and n != root:
                left = neighbors[0]
                right = neighbors[1]

                dis = np.linalg.norm(self.tree.nodes[left]['loc'] - self.tree.nodes[right]['loc'])
                r = np.mean([self.tree[left][n]['radius'], self.tree[n][right]['radius']])
                self.tree.add_edge(left, right, radius=r, radius_neg=-r, length=dis)

                self.tree.remove_node(n)

        self.update_order()

        return

    def prune_too_deep(self, level=8):

        
        self.tree_node_before = np.array(self.tree.nodes)

        all_orders = np.array([self.tree.nodes[n]['level'] for n in self.tree.nodes])
        nodes_to_delete = np.logical_and(all_orders <= level, all_orders != -1)
        nodes_to_remain = np.logical_or(all_orders > level, all_orders != -1)

        points = [self.tree.nodes[i]['loc'] for i in self.tree_node_before[nodes_to_delete]]
        points = np.array(points)

        mesh = pyvista.PolyData(points)

        mesh.save(f'pruned_points_{level}.vtk')

        self.tree.remove_nodes_from(self.tree_node_before[nodes_to_delete])

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

    def prune_by_distance(self, dist=8, thresh=1):

        def get_dist(a, b):
            all_locs = (a - b) ** 2
            all_locs = np.sqrt(np.sum(all_locs))
            return all_locs

        def binary_search(low, high, root, dist):
            mid = (low + high) / 2
            cur_dist = get_dist(mid, root)
            if np.abs(cur_dist - dist) < thresh:
                return mid
            elif cur_dist > dist:
                mid = (low + mid) / 2
                return binary_search(low, mid, root, dist)
            else:
                mid = (high + mid) / 2
                return binary_search(mid, high, root, dist)

        
        self.tree_node_before = np.array(self.tree.nodes)

        nodes_to_delete = []

        for edge in list(self.tree.edges):
            node1, node2 = edge
            if get_dist(self.tree.nodes[node1]['loc'], self.root_loc) > dist and \
                    get_dist(self.tree.nodes[node2]['loc'], self.root_loc) > dist:
                nodes_to_delete.append(node1)
                nodes_to_delete.append(node2)

                self.tree.remove_edge(node1, node2)

            elif get_dist(self.tree.nodes[node1]['loc'], self.root_loc) <= dist and \
                    get_dist(self.tree.nodes[node2]['loc'], self.root_loc) <= dist:
                continue
            else:
                if get_dist(self.tree.nodes[node1]['loc'], self.root_loc) > dist:
                    self.tree.nodes[node1]['loc'] = binary_search(self.tree.nodes[node2]['loc'],
                                                                  self.tree.nodes[node1]['loc'],
                                                                  self.root_loc,
                                                                  dist)
                else:
                    self.tree.nodes[node2]['loc'] = binary_search(self.tree.nodes[node1]['loc'],
                                                                  self.tree.nodes[node2]['loc'],
                                                                  self.root_loc,
                                                                  dist)

        for node in list(self.tree.nodes):
            if len(list(self.tree.neighbors(node))) == 0:
                self.tree.remove_node(node)


        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

    def prune_too_deep_by_depth(self, level=8):

        
        self.tree_node_before = np.array(self.tree.nodes)

        all_orders = np.array([self.tree.nodes[n]['depth'] for n in self.tree.nodes])
        nodes_to_delete = all_orders <= level
        nodes_to_remain = all_orders > level

        points = [self.tree.nodes[i]['loc'] for i in self.tree_node_before[nodes_to_delete]]
        points = np.array(points)

        mesh = pyvista.PolyData(points)

        mesh.save(f'pruned_points_{level}.vtk')

        self.tree.remove_nodes_from(self.tree_node_before[nodes_to_delete])

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

        self.save(file=f'pruned_{level}.vtk')

    def prune_too_short(self, level=20):

        
        self.tree_node_before = np.array(self.tree.nodes)
        all_nodes = self.tree.nodes
        all_nodes = np.array([n for n in self.tree.nodes if len(list(self.tree.neighbors(n))) == 1])

        all_nodes = np.array([n for n in all_nodes if
                              np.linalg.norm(
                                  self.tree.nodes[n]['loc'] - self.tree.nodes[list(self.tree.neighbors(n))[0]][
                                      'loc']) <= level])

        nodes_to_delete = all_nodes

        self.tree.remove_nodes_from(nodes_to_delete)

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.update_order()

        self.save(file=f'pruned_{level}.vtk')


    def make_node_radius(self):

        for n in self.tree.edges:
            a, b = n[0], n[1]
            self.tree.nodes[b]['node_radius'] = max(self.tree[a][b]['radius'], self.tree.nodes[b]['node_radius'])
            self.tree.nodes[a]['node_radius'] = max(self.tree[a][b]['radius'], self.tree.nodes[a]['node_radius'])

        root = [i for i in self.tree.nodes if self.tree.nodes[i]['node_radius'] == 0]
        print(len(root))

        for n in self.tree.edges:
            for i in n:
                self.tree.nodes[i]['node_radius_scaled'] = self.tree.nodes[i]['node_radius'] / self.vsize

if __name__ == '__main__':


    root_loc = [526, 224, 608]
    pt_file = 'artery_centerline_filtered.vtk'
    root_loc = [590, 202, 638]


    vt = VtkNetwork(pt_file=pt_file, root_loc=root_loc)
    vt.build()

    vt.mst()
    
    vt.remove_intermediate()

    vt.save(file='remove_intermediate.vtk')

    vt.prune_too_dense(3)
    vt.prune_asymmetric(0.3)
    vt.prune_too_deep(level=4)
    vt.prune_asymmetric(0.3)

    vt.update_order()
    vt.label_dist_from_root()
    vt.prune_dist_from_root(450) 

    vt.remove_intermediate()

    vt.save(file='final_main_artery.vtk')


    
