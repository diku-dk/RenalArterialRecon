"""
Author: Junhong Shen (jhshen@g.ucla.edu) and Peidi Xu (peidi@di.ku.dk)

Description: Using a directed graph for speedup.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pyvista
import warnings
import functools
import nibabel as nib
from scipy.ndimage import label
from skimage import measure
import os
import copy

class VascularNetwork:
    """
    Given a set of fixed nodes and leaf nodes, generates a representation of vascular networks that
    connects all leaf nodes with the root nodes.
    """

    def __init__(self, fixed, leaves, r_init, f_init, p_init, edge_list=[], r_leaf=0.5, vsize=1,
                 split=False, no_branch=[], pt_file=None, root_loc=None, only_connect_non_branching=False, t_flow=1,
                 mu=3.6 * 1e-3):

        self.tree = nx.Graph()

        if not isinstance(r_init, list):
            r_init = len(edge_list) * [r_init]
        assert len(r_init) == len(edge_list)

        self.root_r = r_init
        self.only_connect_non_branching = only_connect_non_branching
        self.edge_list = edge_list
        self._r_0 = r_leaf
        self.f_0 = t_flow
        self.p_0 = p_init
        self.node_count = 0

        self.vsize = vsize
        
        self.k = 1
        self.c = 3
        self.mu = 3.6 * 1e-3
        self.alpha = 1
        self.no_branch = no_branch

        self.leaves_loc = leaves
        self.root_loc = root_loc
        self.pt_file = pt_file
        

        if pt_file is None:
            self.fixed = range(len(fixed))
            self.leaves = range(len(fixed), len(fixed) + len(leaves))

            for i, fixed_loc in enumerate(fixed):
                self.add_branching_point(fixed_loc, fixed=True, root= i == 0, no_branch=i in no_branch)

            for leaf_loc in leaves:
                self.add_branching_point(leaf_loc, leaf=True)

            self.initialize_nodes(split=split)
            self.update_order_undirect()

        else:
            mesh = pyvista.read(self.pt_file)
            points = mesh.points

            self.fixed = range(len(points))
            self.leaves = range(len(points), len(points) + len(leaves))

            self.init_from_vtk(split=split)

        self.to_directed()

        self.tree = self.di_tree

        print('to directed')

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

    def find_n_with_coord(self, coords=np.array([1, 2, 3])):
        all_nodes = [i for i in self.tree.nodes]
        all_loc = np.array([self.tree.nodes[i]['loc'] for i in all_nodes])
        all_dist = np.sum((all_loc - np.array(coords)) ** 2, axis=1)
        return all_nodes[np.argmin(all_dist)]

    @property
    def r_0(self):
        if np.isscalar(self._r_0):
            return self._r_0
        else:
            mins, maxs = scipy.stats.norm.ppf([.01, .99], self._r_0[0], self._r_0[1])

            return min(max(np.random.normal(loc=self._r_0[0], scale=self._r_0[1]), mins), maxs)

    @r_0.setter
    def r_0(self, value):
        self._r_0 = value

    def init_from_vtk(self, split=True):

        mesh = pyvista.read(self.pt_file)
        points = mesh.points
        line = mesh.lines

        
        edge_radius = mesh.cell_data['radius']

        for p in points:
            self.add_branching_point(p, fixed=True, root=False, no_branch=False)

        i = 1
        while i < len(line):
            node1, node2 = line[i], line[i + 1]
            
            

            radius = edge_radius[i // 3]

            self.add_vessel(node1, node2, radius=radius * self.vsize, flow=self.f_0)

            i += 3

        root = self.find_n_with_coord(self.root_loc)

        self.tree.nodes[root]['fixed'] = True
        self.tree.nodes[root]['root'] = True

        if self.only_connect_non_branching:
            for n in self.tree.nodes:
                self.tree.nodes[n]['no_branch'] = True if len(list(self.tree.neighbors(n))) > 2 or \
                                                          self.tree.nodes[n]['root'] else False

        for leaf_loc in self.leaves_loc:
            self.add_branching_point(leaf_loc, leaf=True)

        for node in list(self.tree.nodes):
            if not self.tree.nodes[node]['fixed']:
                closest = self.find_nearest_fixed(node)
                self.add_vessel(node, closest, self.r_0, sub=closest)

        self.update_order_undirect()

        nodes_to_remove = []
        fix_remove = []
        for node in list(self.tree.nodes):
            if self.tree.nodes[node]['fixed']:
                leaf_neighbors = [i for i in self.tree.neighbors(node) if self.tree.nodes[i]['leaf']]
                if len(leaf_neighbors) < 20 and len(leaf_neighbors) > 0:
                    nodes_to_remove += leaf_neighbors
                    fix_remove += [node]
                    flag = True

        nodes_to_remove = sorted(nodes_to_remove)
        nodes_to_remove = np.unique(nodes_to_remove)
        self.tree.remove_nodes_from(nodes_to_remove)
        self.reorder_removed_leaves(nodes_to_remove)


        print(f'removed leaves = {nodes_to_remove}')
        print('ok')

        self.node_count = len(self.tree.nodes)

        return

        nodes_to_remove = []
        for node in list(self.tree.nodes):
            if self.tree.nodes[node]['fixed'] and self.tree.degree[node] == 2 and not \
                    self.tree.nodes[node]['root']:
                node_del = [node]
                cur_node = list(self.tree.neighbors(node))[0]

                while self.tree.degree[node] == 2:
                    node_del.append(cur_node)
                    cur_node = list(self.tree.neighbors(cur_node))[0]

                nodes_to_remove += node_del

        nodes_to_remove = sorted(nodes_to_remove)

        nodes_to_remove = np.unique(nodes_to_remove)


        all_nodes = list(self.tree.nodes)

        for node in all_nodes:
            if self.tree.nodes[node]['fixed'] and self.tree.degree[node] == 2 and not self.tree.nodes[node]['root']:
                flag = True
                
                left = list(self.tree.predecessors(node))[0]
                right = list(self.tree.neighbors(node))[0]

                r = np.mean([self.tree[left][node]['radius'], self.tree[node][right]['radius']])
                
                self.add_vessel(left, right, radius=r, flow=self.tree[left][node]['flow'])

                self.tree.remove_node(node)
                nodes_to_remove.append(node)

        self.node_count = len(self.tree.nodes)

    def initialize_nodes(self, split=True):
        """
        Connects the fixed nodes with the information given. Creates intermediate nodes for an edge if necessary.
        Then connects each leaf node with the closest fixed node.

        Parameters
        --------------------
        split -- if True, creates intermediate nodes (only between fixed) for an edge if necessary
        """

        for r, edge in zip(self.root_r, self.edge_list):
            n1, n2 = edge[0], edge[1]
            
            self.add_vessel(n1, n2, self.r_0, flow=self.f_0)

        

        if split:
            count = 0
            level_map = {}
            for i in range(5):
                for edge in list(self.tree.edges):
                    node1, node2 = edge

                    if not (self.tree.nodes[node1]['fixed'] and self.tree.nodes[node2]['fixed']):
                        continue

                    if np.linalg.norm(self.tree.nodes[node1]['loc'] -
                                      self.tree.nodes[node2]['loc']) * self.vsize <= 6:
                        continue

                    loc = (self.tree.nodes[node1]['loc'] + self.tree.nodes[node2]['loc']) / 2
                    self.split(node1, node2_loc=loc, nodes_to_split=[node2])
                    level_map[self.node_count - 1] = len(self.fixed) + count
                    self.tree.nodes[self.node_count - 1]['fixed'] = True

                    assert np.all(self.tree.nodes[self.node_count - 1]['loc'] == loc)

                    count += 1
            
            for i in self.fixed:
                level_map[i] = i
            for i in self.leaves:
                level_map[i] = i + count

            

            self.tree = nx.relabel_nodes(self.tree, level_map)

            self.fixed = range(len(self.fixed) + count)
            self.leaves = range(len(self.fixed), len(self.fixed) + len(self.leaves))
            self.edge_list = []
            for edge in list(self.tree.edges):
                node1, node2 = edge
                self.edge_list.append([node1, node2])

        for node in list(self.tree.nodes):
            if not self.tree.nodes[node]['fixed']:
                closest = self.find_nearest_fixed(node)
                self.add_vessel(node, closest, self.r_0)

        for node in list(self.tree.nodes):
            if self.tree.nodes[node]['fixed'] and not self.tree.nodes[node]['root']:
                neighbors = list(self.tree.neighbors(node))
                if len(neighbors) == 2:
                    self.remove_middle(node, neighbors)

    def remove_middle(self, *args):
        print('middle point that does not have nearest leaves needs to be deleted')

    def rescale_leaf_radius(self):

        max_root_level = np.max([self.tree.nodes[node]['root_level'] for node in self.tree.nodes
                                 if self.tree.nodes[node]['fixed']])

        for i in range(1, max_root_level + 1):
            for node in list(self.tree.nodes):
                if self.tree.nodes[node]['root']:
                    continue
                if self.tree.nodes[node]['fixed']:
                    neighbors = list(self.tree.neighbors(node))
                    root_idx = list(self.tree.predecessors(node))[0]

                    if self.tree.nodes[node]['root_level'] == i:
                        r_p = self.tree[root_idx][node]['radius']
                        r_child_leaf = np.array([self.tree[node][n]['radius'] for n in neighbors
                                                 if not self.tree.nodes[n]['fixed']])
                        r_child_fixed = np.array([self.tree[node][n]['radius'] for n in neighbors
                                                  if self.tree.nodes[n]['fixed']])

                        if r_p == 0:
                            r_root = np.power(np.sum(r_child_leaf ** 3) + np.sum(r_child_fixed ** 3), 1 / 3)
                            self.tree[root_idx][node]['radius'] = r_root
                        else:
                            r_leaf = r_p ** 3 - np.sum(r_child_fixed ** 3)
                            leaf_num = len([n for n in neighbors if not self.tree.nodes[n]['fixed']])
                            r_leaf = np.power(r_leaf / leaf_num, 1 / 3)
                            for n in neighbors:
                                if not self.tree[n]['fixed']:
                                    self.tree[node][n]['radius'] = r_leaf

    def rescale_fixed_radius(self, mode='level'):

        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 else 0
            if self.tree.nodes[n]['root']:
                self.tree.nodes[n][mode] = -1

        
        count_no_label = len(self.tree.nodes)


        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue

                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                root = list(self.tree.predecessors(node))[0]

                if np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue

                max_order = np.max(neighbor_orders)
                self.tree.nodes[node][mode] = max_order + 1

                r_child = np.array([self.tree[node][n]['radius'] for n in self.tree.neighbors(node)])

                f_child = np.array([self.tree[node][n]['flow'] for n in self.tree.neighbors(node)])

                r_root = np.power(np.sum(r_child ** 3), 1 / 3)
                f_root = np.sum(f_child)

                self.tree[root][node]['radius'] = r_root
                self.tree[root][node]['flow'] = f_root

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def radius_check(self):

        self.update_order()

        points = []

        for node in list(self.tree.nodes):
            if self.tree.nodes[node]['root']:
                continue

            if self.tree.nodes[node]['fixed']:
                neighbors = list(self.tree.neighbors(node))
                neighbors = np.array(neighbors)
                neighbor_order = np.array([self.tree.nodes[n]['level'] for n in self.tree.neighbors(node)])

                root = list(self.tree.predecessors(node))[0]

                if len(neighbors) == 1 or len(neighbors) == 0:
                    continue

                r_p = self.tree[root][node]['radius']
                r_child = np.array([self.tree[node][n]['radius'] for n in neighbors])

                max_r_child = np.max(r_child)

                if max_r_child > r_p and (not np.isclose(max_r_child, r_p)):
                    print('got child raidus > parent vessel')
                    points.append(self.tree.nodes[node]['loc'])


        if len(points) != 0:
            points = np.array(points)
            mesh = pyvista.PolyData(points)
            mesh.save('wrong radiuses.vtk')
        else:
            print('ok, all parent radius > child radius')

    def rescale_flow(self):
        self.update_order(mode='level')

        max_order = np.max([self.tree.nodes[node]['level'] for node in self.tree.nodes])

        for i in range(1, max_order + 1):
            for node in list(self.tree.nodes):
                if self.tree.nodes[node]['root_level'] == i and i == 1:
                    pass
                elif self.tree.nodes[node]['root_level'] == i:
                    neighbors = list(self.tree.neighbors(node))
                    root_idx = list(self.tree.predecessors(node))[0]
                    f_children = np.array([self.tree[node][n]['flow'] for n in neighbors])
                    self.tree[root_idx][node]['flow'] = np.sum(f_children)

    def add_branching_point(self, loc, fixed=False, pres=None, root=False, no_branch=False, level=None,
                            pseudo_fixed=False, leaf=False):
        """
        Adds a branching point.

        Parameters
        --------------------
        loc -- location of the branching point
        """

        self.tree.add_node(self.node_count, loc=np.array(loc), pressure=pres, HS=None, level=level, fixed=fixed,
                           root=root, root_level=None, no_branch=no_branch, pseudo_fixed=pseudo_fixed, parent=None,
                           forest_level=None, forest_HS=None, leaf=leaf)

        self.node_count += 1
        return self.node_count - 1

    def add_vessel(self, node, neighbor_node, radius=None, flow=None, sub=-1):
        """
        Adds a vessel between two nodes.

        Parameters
        --------------------
        node -- one endpoint
        neighbor_node -- the other endpoint
        """

        r = self.r_0 if radius is None else radius
        f = self.f_0 if flow is None else flow
        dis = np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[neighbor_node]['loc']
                             ) * self.vsize
        self.tree.add_edge(node, neighbor_node, radius=r, flow=f, length=dis, sub=sub)

    
    def merge(self, node1, node2):
        """
        Merges node2 with node1.
        """
        all_neighbors = list(self.tree.neighbors(node2))
        if list(self.tree.predecessors(node1))[0] == node2:
            n = list(self.tree.predecessors(node2))[0]
            self.add_vessel(n, node1, self.tree[n][node2]['radius'],
                            self.tree[n][node2]['flow'])
            self.tree.remove_edge(n, node2)

        for n in all_neighbors:
            if n != node1:
                self.add_vessel(node1, n, self.tree[node2][n]['radius'], self.tree[node2][n]['flow'])
                self.tree.remove_edge(node2, n)

        if list(self.tree.predecessors(node1))[0] == node2:
            self.tree.remove_edge(node2, node1)
        else:
            self.tree.remove_edge(node1, node2)

        assert self.tree.degree[node2] == 0

        self.tree.remove_node(node2)

    def split(self, node1, node2_loc, nodes_to_split):
        """
        Splits nodes_to_split from node1.
        """

        level = np.max([self.tree.nodes[n]['level'] for n in nodes_to_split])

        node2 = self.add_branching_point(node2_loc, level=level)

        for n in nodes_to_split:
            self.add_vessel(node2, n, self.tree[node1][n]['radius'], self.tree[node1][n]['flow'])

        r_sum, f_sum = self.split_radius(node1, nodes_to_split)

        for n in nodes_to_split:
            self.tree.remove_edge(node1, n)
        self.add_vessel(node1, node2, r_sum, f_sum)
        

    def split_radius(self, node, nodes_to_split, remaining_nodes=None, root=None):
        """
        Finds the radius connecting the old node and the new node after splitting.
        """

        if len(nodes_to_split) == 1:
            n = nodes_to_split[0]
            return self.tree[node][n]['radius'], self.tree[node][n]['flow']

        if remaining_nodes is None and root is None:
            remaining_nodes = list(self.tree.neighbors(node))
            root = list(self.tree.predecessors(node))[0]

        remaining_nodes = set(remaining_nodes)
        nodes_to_split = set(nodes_to_split)

        if root not in nodes_to_split:
            remaining_nodes = nodes_to_split
        else:
            remaining_nodes = remaining_nodes - nodes_to_split

        r_sum = [self.tree[node][n]['radius'] ** self.c for n in remaining_nodes]
        f_sum = [self.tree[node][n]['flow'] for n in remaining_nodes]

        r_sum = np.sum(r_sum)
        f_sum = np.sum(f_sum)

        if r_sum > 0:
            final_radius = r_sum ** (1 / self.c)
        else:
            print('something wrong')
            final_radius = r_sum ** (1 / self.c)

        return final_radius, f_sum

    def prune(self, l, mode='level'):
        """
        Remove nodes by predefined criteria.
        """
        total_pruned = 0

        if l == 0:
            return 0

        self.update_order(mode)
        for edge in list(self.tree.edges):
            node1, node2 = edge
            edge1 = [node1, node2]
            edge2 = [node2, node1]
            if edge1 in self.edge_list or edge2 in self.edge_list:
                continue

            if self.tree.nodes[node1]['fixed'] and self.tree.nodes[node2]['fixed']:
                continue
            if self.tree.nodes[node1]['fixed'] and self.tree.nodes[node2]['pseudo_fixed'] or \
                    self.tree.nodes[node1]['pseudo_fixed'] and self.tree.nodes[node2]['fixed']:
                continue

            if self.tree.nodes[node1][mode] == -1 or self.tree.nodes[node2][mode] == -1:
                order = max(self.tree.nodes[node1][mode], self.tree.nodes[node2][mode])
            else:
                order = min(self.tree.nodes[node1][mode], self.tree.nodes[node2][mode])

            
            if order <= l:
                self.tree.remove_edge(node1, node2)
                
                total_pruned += 1
        
        for node in list(self.tree.nodes):
            if len(list(self.tree.neighbors(node))
                   + list(self.tree.predecessors(node))) == 0 and not self.tree.nodes[node]['leaf'] and not \
            self.tree.nodes[node]['fixed']:
                self.tree.remove_node(node)

        return total_pruned

    def get_child_helper(self, all_edges, node, parent):
        all_neighbors = np.array([n for n in self.tree.neighbors(node)])
        neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in all_neighbors])
        assert -1 not in neighbor_orders

        self.tree.nodes[node]['parent'] = parent

        if len(all_neighbors) == 0:
            return all_edges

        elif len(all_neighbors) == 1:
            self.tree.nodes[all_neighbors[0]]['parent'] = parent
            all_edges.append([node, all_neighbors[0]])
            return all_edges

        else:
            for n in all_neighbors:
                self.tree.nodes[n]['parent'] = parent
                all_edges.append([node, n])
                all_edges = self.get_child_helper(all_edges, n, parent=parent)

        return all_edges

    def get_all_child_edges(self, node):

        all_edges = []
        all_neighbors = np.array([n for n in self.tree.neighbors(node) if not self.tree.nodes[n]['fixed']])

        for n in all_neighbors:
            self.tree.nodes[n]['parent'] = node
            all_edges.append([node, n])
            all_edges = self.get_child_helper(all_edges, n, parent=node)
        return all_edges

    def forest_prune(self, l, mode='level', prune_threshold=0, too_few_prune=10):
        """
        Remove nodes by predefined criteria.
        """
        total_pruned = 0
        original_mode = mode
        mode = 'forest_' + mode
        if l == 0:
            return 0

        for node in self.tree.nodes:
            cur_max = self.tree.nodes[node][mode]
            original_max = self.tree.nodes[node][original_mode]

            if self.tree.nodes[node]['fixed'] and not self.tree.nodes[node]['root'] and (
                    original_max >= prune_threshold and cur_max > l or cur_max <= too_few_prune):

                all_leaf_neighbors = [i for i in self.tree.neighbors(node) if not self.tree.nodes[i]['fixed']]
                

                all_edges = []

                for n in all_leaf_neighbors:

                    all_edges.append([node, n])

                    self.tree.nodes[n]['parent'] = node

                    all_descentants = nx.descendants(self.tree, n)
                    assert len([i for i in all_descentants if self.tree.nodes[i]['fixed']]) == 0

                    for i in all_descentants:
                        self.tree.nodes[i]['parent'] = node

                    all_edges += [[list(self.tree.predecessors(i))[0], i] for i in all_descentants]

                for edge in all_edges:
                    node1, node2 = edge

                    if edge in self.edge_list:
                        continue
                    assert not (self.tree.nodes[node1]['fixed'] and self.tree.nodes[node2]['fixed'])

                    if self.tree.nodes[node1][mode] == -1 or self.tree.nodes[node2][mode] == -1:
                        order = max(self.tree.nodes[node1][mode], self.tree.nodes[node2][mode])
                    else:
                        order = min(self.tree.nodes[node1][mode], self.tree.nodes[node2][mode])

                    if l >= prune_threshold:
                        if order == 1:
                            self.tree.remove_edge(node1, node2)
                            total_pruned += 1

                    elif order <= self.tree.nodes[node][mode] - l:

                        self.tree.remove_edge(node1, node2)
                        
                        total_pruned += 1

                
                

        for node in list(self.tree.nodes):
            if self.tree.degree[node] == 0 and not self.tree.nodes[node]['leaf']:
                assert not self.tree.nodes[node]['fixed']

                self.tree.remove_node(node)

        return total_pruned

    def remove_negative_raidus(self):

        for n1, n2 in self.tree.edges:
            if self.tree[n1][n2]['radius'] <= 0:
                print(n1, n2)
                self.tree.remove_edge(n1, n2)

        for node in list(self.tree.nodes):
            if len(list(self.tree.neighbors(node))
                   + list(self.tree.predecessors(node))) == 0 and not self.tree.nodes[node]['leaf'] and not \
            self.tree.nodes[node]['fixed']:
                self.tree.remove_node(node)

    def forest_reconnect(self, avoid_root_recon=False):
        """
        Reconnect leaf nodes to the nearest existing node.
        """
        for leaf in self.tree.nodes:
            if not self.tree.nodes[leaf]['leaf'] or self.tree.degree[leaf] != 0:
                continue

            assert self.tree.nodes[leaf]['parent'] is not None

            all_nodes = [n for n in self.tree.nodes
                         if not self.tree.nodes[n]['leaf'] and not self.tree.nodes[n]['no_branch']
                         and self.tree.nodes[n]['parent'] == self.tree.nodes[leaf]['parent']]

            

            if avoid_root_recon:
                all_nodes = [n for n in self.tree.nodes
                             if not self.tree.nodes[n]['leaf'] and not self.tree.nodes[n]['root']
                             and not self.tree.nodes[n]['no_branch']
                             and self.tree.nodes[n]['parent'] == self.tree.nodes[leaf]['parent']]

            dis_list = np.array([np.linalg.norm(
                self.tree.nodes[leaf]['loc'] - self.tree.nodes[n]['loc']) * self.vsize
                                 for n in all_nodes])

            ranges = np.argmin(dis_list)
            nearest_node = all_nodes[ranges]
            self.add_vessel(nearest_node, leaf, self.r_0, self.f_0)

        self.rescale_pruned_leaves()

    def rescale_pruned_leaves(self):
        for node in self.tree.nodes:
            if not self.tree.nodes[node]['leaf'] and len(list(self.tree.successors(node))) == 0:
                pre = list(self.tree.predecessors(node))[0]
                self.tree[pre][node]['radius'] = self.r_0
                self.tree[pre][node]['flow'] = self.f_0

        self.rescale_fixed_radius()


    def reconnect(self, avoid_root_recon=False):
        """
        Reconnect leaf nodes to the nearest existing node.
        """
        all_nodes = [n for n in self.tree.nodes
                     if not self.tree.nodes[n]['leaf'] and not self.tree.nodes[n]['no_branch']]

        if avoid_root_recon:
            all_nodes = [n for n in self.tree.nodes
                         if not self.tree.nodes[n]['leaf'] and not self.tree.nodes[n]['root']
                         and not self.tree.nodes[n]['no_branch']]

        for leaf in self.tree.nodes:
            if not self.tree.nodes[leaf]['leaf']: continue
            if self.tree.degree[leaf] != 0: continue

            dis_list = np.array([np.linalg.norm(
                self.tree.nodes[leaf]['loc'] - self.tree.nodes[n]['loc']) * self.vsize
                                 for n in all_nodes])

            ranges = np.argmin(dis_list)
            nearest_node = all_nodes[ranges]
            self.add_vessel(nearest_node, leaf, self.r_0, self.f_0)

    def reorder_removed_leaf(self, n):
        """
        Reorder nodes by level.
        """

        level_map = {}
        for i in self.tree.nodes:
            if i < n:
                level_map[i] = i
            else:
                level_map[i] = i - 1

        self.tree = nx.relabel_nodes(self.tree, level_map)

    def reorder_removed_leaves(self, n):
        """
        Reorder nodes by level.
        """

        level_map = {}
        j = 0
        for i in self.tree.nodes:

            while j < len(n) and i > n[j]:
                j += 1

            level_map[i] = i - j

        assert len(level_map) == len(self.tree.nodes)

        self.tree = nx.relabel_nodes(self.tree, level_map)

    def reorder_nodes(self, undirect=False):
        """
        Reorder nodes by level.
        """
        if undirect:
            self.update_order_undirect()
        else:
            self.update_order()
        level_list = []
        node_list = list(self.tree.nodes)
        for node in node_list:
            level_list.append(self.tree.nodes[node]['level'])
        level_idices = np.argsort(-1 * np.array(level_list))
        level_map = {}
        for i in self.fixed:
            level_map[i] = i
        for i in self.leaves:
            level_map[i] = i
        idx = len(self.leaves) + len(self.fixed)

        fix_leaf_indices = np.hstack((np.array(self.fixed), np.array(self.leaves)))
        remaining_indices = np.setdiff1d(level_idices, fix_leaf_indices)

        
        
        

        for i in remaining_indices:
            level_map[node_list[i]] = idx
            idx += 1

        self.tree = nx.relabel_nodes(self.tree, level_map)

    def update_forest_order(self, mode='level'):
        mode = 'forest_' + mode
        for n in self.tree.nodes:

            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 else 0

            if self.tree.nodes[n]['fixed']:
                self.tree.nodes[n][mode] = -1

        count_no_label = self.node_count
        cur_order = 1

        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
                root = list(self.tree.predecessors(node))[0]
                root_order = self.tree.nodes[root][mode]

                if np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue

                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)

                if len(neighbor_orders) == 1:
                    self.tree.nodes[node][mode] = max_order
                else:
                    self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'forest_HS' else max_order + 1

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

        for node in self.tree.nodes:

            if self.tree.nodes[node]['fixed'] and not self.tree.nodes[node]['root']:
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)
                                            if not self.tree.nodes[n]['fixed']])
                if len(neighbor_orders) == 0:
                    continue

                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)

                if len(neighbor_orders) == 1:
                    self.tree.nodes[node][mode] = max_order
                else:
                    self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'forest_HS' else max_order + 1

    def update_order(self, mode='level'):
        for n in self.tree.nodes:
            

            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 else 0

            if self.tree.nodes[n]['fixed'] and self.tree.nodes[n]['root']:
                self.tree.nodes[n][mode] = -1

        count_no_label = self.node_count
        cur_order = 1
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                root = list(self.tree.predecessors(node))[0]
                root_order = self.tree.nodes[root][mode]

                if np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue

                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)

                if len(neighbor_orders) == 1:
                    self.tree.nodes[node][mode] = max_order
                else:
                    self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def update_order_undirect(self, mode='level'):
        for n in self.tree.nodes:
            

            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 else 0

            if self.tree.nodes[n]['fixed'] and self.tree.nodes[n]['root']:
                self.tree.nodes[n][mode] = -1

        count_no_label = self.node_count
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

                if len(neighbor_orders) == 2:
                    self.tree.nodes[node][mode] = max_order
                else:
                    self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def update_final_order(self, mode='HS'):
        """
        Update order when no further operations will be applied.
        """

        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] == 1 else 0
        count_no_label = self.node_count
        cur_order = 1
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])

                if np.count_nonzero(neighbor_orders == 0) >= 1:  
                    continue
                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)
                self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1
            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

    def final_merge(self):
        """
        Remove unneeded nodes on edges when no further operations will be applied.
        """

        merge_count = 0
        for n in list(self.tree.nodes):
            if n not in self.tree.nodes: continue
            neighbors = list(self.tree.neighbors(n))
            if len(neighbors) == 2:
                n1, n2 = neighbors[0], neighbors[1]
                s1 = (self.tree.nodes[n]['loc'] - self.tree.nodes[n1]['loc'])[1] / \
                     (self.tree.nodes[n]['loc'] - self.tree.nodes[n1]['loc'])[0]
                s2 = (self.tree.nodes[n]['loc'] - self.tree.nodes[n2]['loc'])[1] / \
                     (self.tree.nodes[n]['loc'] - self.tree.nodes[n2]['loc'])[0]
                if s1 == s2 and self.tree[n][n1]['radius'] == self.tree[n][n2]['radius']:
                    merge_count += 1
                    self.merge(n2, n)

        print(f'final merged {merge_count} nodes')

    
    
    

    def find_nearest_fixed(self, node):

        dis_list = np.array([np.linalg.norm(
            self.tree.nodes[node]['loc'] - self.tree.nodes[n]['loc']) * self.vsize if not self.tree.nodes[n][
            'no_branch'] else np.inf
                             for n in self.fixed])

        ranges = np.argsort(dis_list)

        for n in ranges:
            if not self.tree.nodes[n]['no_branch']:
                return n

        print('something wrong, all fixed node cannot branch?')

        return ranges[0]

    def move_node(self, node, loc_new):
        self.tree.nodes[node]['loc'] = loc_new
        for n in self.tree.neighbors(node):
            self.tree[node][n]['length'] = np.linalg.norm(self.tree.nodes[node]['loc'] -
                                                          self.tree.nodes[n]['loc']) * self.vsize

        n = list(self.tree.predecessors(node))[0]
        self.tree[n][node]['length'] = np.linalg.norm(self.tree.nodes[node]['loc'] -
                                                      self.tree.nodes[n]['loc']) * self.vsize

    def update_radius_and_flow(self, edge, r_new, flow=0):
        node1, node2 = edge
        self.tree[node1][node2]['radius'] = r_new
        self.tree[node1][node2]['flow'] = self.k * (r_new ** self.c) if flow == 0 else flow

    def get_max_level(self, mode='level'):
        return np.max(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]))

    def get_max_forest_level(self, mode='level'):
        mode = 'forest_' + mode
        return np.max(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]))

    def get_conductance(self, edge):
        node1, node2 = edge
        return self.alpha * np.pi * (self.tree[node1][node2]['radius'] ** 4) / (
                8 * self.mu * self.tree[node1][node2]['length'])

    def get_pressure_diff(self, edge):
        node1, node2 = edge
        return self.tree[node1][node2]['flow'] / self.get_conductance(edge)

    def get_power_loss(self, edge):
        node1, node2 = edge
        return (self.tree[node1][node2]['flow'] ** 2) * (1 / self.get_conductance(edge))

    def get_max_stress(self, edge):
        node1, node2 = edge
        return (4 * self.mu * self.tree[node1][node2]['flow']) / (np.pi * self.tree[node1][node2]['radius'] ** 3)

    def remove_intermediate(self):
        max_orders = np.max(np.array([self.tree.nodes[n]['level'] for n in self.tree.nodes]))
        root = [n for n in self.tree.nodes if self.tree.nodes[n]['root']]
        assert len(root) == 1
        root = root[0]
        all_nodes = [n for n in self.tree.nodes]
        for n in all_nodes:
            neighbors = list(self.tree.neighbors(n))
            if len(neighbors) == 1 and n != root:
                left = list(self.tree.predecessors(n))[0]
                right = neighbors[0]
                
                r = np.mean([self.tree[left][n]['radius'], self.tree[n][right]['radius']])
                

                self.add_vessel(left, right, radius=r, flow=self.tree[left][right]['flow'])

                self.tree.remove_node(n)

        return

    def update_HS_order(self, mode='level'):
        for n in self.tree.nodes:
            self.tree.nodes[n][mode] = 1 if self.tree.degree[n] <= 1 and not self.tree.nodes[n]['root'] else 0
            if self.tree.nodes[n]['root']:
                self.tree.nodes[n][mode] = -1
        count_no_label = len(self.tree.nodes)
        cur_order = 1
        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue
                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
                if np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue
                root = list(self.tree.predecessors(node))[0]
                root_order = self.tree.nodes[root][mode]

                max_order = np.max(neighbor_orders)
                max_count = np.count_nonzero(neighbor_orders == max_order)
                if len(neighbor_orders) == 1:
                    self.tree.nodes[node][mode] = max_order
                else:
                    self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

        node = [n for n in self.tree.nodes if self.tree.nodes[n][mode] == -1]
        assert len(node) == 1
        node = node[0]

        neighbor_orders = np.array([self.tree.nodes[n][mode] for n in self.tree.neighbors(node)])
        max_order = np.max(neighbor_orders)
        max_count = np.count_nonzero(neighbor_orders == max_order)
        self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

    def make_node_radius(self):

        for n in self.tree.nodes:
            self.tree.nodes[n]['node_radius'] = 0

        for n in self.tree.edges:
            a, b = n[0], n[1]
            self.tree.nodes[b]['node_radius'] = max(self.tree[a][b]['radius'], self.tree.nodes[b]['node_radius'])
            self.tree.nodes[a]['node_radius'] = max(self.tree[a][b]['radius'], self.tree.nodes[a]['node_radius'])

        root = [i for i in self.tree.nodes if self.tree.nodes[i]['node_radius'] == 0]
        

        for n in self.tree.edges:
            for i in n:
                self.tree.nodes[i]['node_radius_scaled'] = self.tree.nodes[i]['node_radius'] / 22.6

    def label_sub_regions(self):

        all_fixed = [node for node in self.tree.nodes if self.tree.nodes[node]['fixed']]
        for node in all_fixed:

            all_neighbors = [i for i in self.tree.neighbors(node)]
            all_neighbors_fixed = [i for i in self.tree.neighbors(node) if self.tree.nodes[i]['fixed']]

            if len(all_neighbors_fixed) > 0:
                continue

            all_descendants = nx.descendants(self.tree, node)
            for i in all_descendants:
                j = list(self.tree.predecessors(i))[0]
                self.tree[j][i]['sub'] = node



    def save_vtk(self, file='vascularTree.vtk', include_line=True, ignore_order=False):

        if not ignore_order:
            self.make_node_radius()
            self.update_HS_order('HS')

            max_hs = np.max([self.tree.nodes[i]['HS'] for i in self.tree.nodes])
        else:
            max_hs = 0

        for i in self.tree.nodes:
            self.tree.nodes[i]['ind'] = i

        self.label_sub_regions()

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

        file = file[:-4] + '_' + str(max_hs) + '_hs' + '.vtk'

        mesh.save(file)

