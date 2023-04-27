"""
Author: Junhong Shen (jhshen@g.ucla.edu) and Peidi Xu (peidi@di.ku.dk)

Description:
    Implementation of GCO Forest, a modified version of Global Constructive Optimization (GCO)
    algorithm with multiple root nodes.

References:
    Georg, Manfred; Preusser, Tobias; and Hahn, Horst K., "Global Constructive Optimization of
    Vascular Systems" Report Number: WUCSE-2010-11 (2010). All Computer Science and Engineering Research.
    http://openscholarship.wustl.edu/cse_research/36
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from VascularNetworkDi import VascularNetwork
from SimulatedAnnealingOptimizer import SA_Optimizer
from GradientDescentOptimizer import GD_Optimizer
from GradientDescentOptimierUnit import GD_OptimizerUnit

import pyvista

import pickle
import multiprocessing
from collections import OrderedDict

from functools import partial

from loggers import *

log = logging.getLogger(__name__)
log.debug('My message with %s', 'variable data')


class GCO:
    """
    Given a set of fixed nodes (root nodes) and leaf nodes, generates an optimized vascular network that
    connects all leaf nodes with the root nodes using Global Constructive Optimization (GCO).
    """

    def __init__(self, fixed_locs, leaf_locs, r_init, f_init, p_init, edge_list, work_dir='./', cost_mode='PC',
                 use_C=True, remove_negative=False, r_leaf=0.5, max_l=3, optimize_r=True, prune_threshold=8,
                 min_r=0.5, max_r=2.0,
                 w1=642.0, w2=1.0, w3=5e3, global_weighting=True, n_outer_loop=15, n_inner_loop=0,
                 include_surface_area=False,
                 save_interm_res=False, vsize=1.0, merge_threshold=0.1, optimizer=None, no_branch=[],
                 prune_mode='level', avoid_root_recon=False, pt_file=None, root_loc=None, rescale_radius=False,
                 only_save_last=False, split_fixed=False, too_few_prune=3, only_connect_non_branching=False,
                 mu=1, t_flow=1
                 ):
        self.mu = mu
        self.t_flow = t_flow
        self.VN = VascularNetwork(fixed_locs, leaf_locs, r_init, f_init, p_init, edge_list, r_leaf=r_leaf, vsize=vsize,
                                  no_branch=no_branch, pt_file=pt_file, root_loc=root_loc,
                                  only_connect_non_branching=only_connect_non_branching,
                                  mu=mu, t_flow=t_flow
                                  )
        self.only_save_last = only_save_last
        self.vsize = vsize
        
        self.avoid_root_recon = avoid_root_recon
        self.stable_nodes = self.VN.fixed
        self.leaves = self.VN.leaves
        self.split_fixed = split_fixed
        
        
        self.optimizer = optimizer
        self.use_C = use_C
        self.cost_mode = cost_mode
        self.only_connect_non_branching = only_connect_non_branching
        
        self.max_l = max_l
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.max_iter_1 = n_outer_loop
        self.max_iter_2 = np.log2(len(self.leaves)) * 3 + 5 if n_inner_loop == 0 else n_inner_loop

        

        print(f"Max iter: {self.max_iter_1} outer and {self.max_iter_2} inner")
        print("Num fixed: %d  Num leaf: %d" % (len(self.stable_nodes), len(self.leaves)))
        assert prune_mode in ('level', 'HS')
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        self.total_loss = []
        self.VN.save_vtk(os.path.join(work_dir, 'whole_structure_before_intialize.vtk'))
        self.remove_negative = remove_negative
        self.optimize_r = optimize_r
        self.min_r = min_r
        self.max_r = max_r

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.global_weighting = global_weighting
        self.include_surface_area = include_surface_area
        self.save_interm_res = save_interm_res
        self.prune_mode = prune_mode

        self.too_few_prune = too_few_prune

        
        if rescale_radius:
            self.VN.rescale_fixed_radius()
            self.VN.radius_check()
            self.VN.save_vtk(os.path.join(work_dir, 'rescaled.vtk'))



        self.initialize()




    def split_pruned_from_fixed(self, iter=0):
        for node in range(len(self.VN.tree), 0, -1):
            if node in self.stable_nodes:
                neighbor_leaves = [i for i in self.VN.tree.neighbors(node) if not self.VN.tree.nodes[i]['fixed'] and
                                   len(list(self.VN.tree.neighbors(i))) == 1]
                neighbor_leaves = np.array(neighbor_leaves)
                if len(neighbor_leaves) <= 1:
                    continue

                chosen_locs = [self.VN.tree.nodes[n]['loc'] for n in neighbor_leaves]
                opt_point_loc = self.get_centroid(chosen_locs + [self.VN.tree.nodes[node]['loc']])

                

                opt_point = self.VN.add_branching_point(opt_point_loc)

                
                for n in neighbor_leaves:
                    self.VN.add_vessel(opt_point, n, self.VN.tree[node][n]['radius'], self.VN.tree[node][n]['flow'])

                r_sum = np.sum([self.VN.tree[node][n]['radius'] ** self.VN.c for n in neighbor_leaves])
                f_sum = np.sum([self.VN.tree[node][n]['flow'] for n in neighbor_leaves])

                if r_sum > 0:
                    r_sum = r_sum ** (1 / self.VN.c)
                else:
                    print('something wrong')
                    r_sum = r_sum ** (1 / self.VN.c)

                for n in neighbor_leaves:
                    self.VN.tree.remove_edge(node, n)

                self.VN.add_vessel(node, opt_point, r_sum, f_sum)

    def initialize(self, iter=0):
        """
        After connecting each leaf node with the closest fixed node, we can regard each fixed node as the root
        node and the nodes incident to it as the leaf nodes in a single GCO problem. Then we have created multiple
        GCO problems and the result will be a set of GCO trees.

        Initializes each GCO tree.
        """
        if iter == 0:
            print("Initial connected component: %d" % len(list(nx.weakly_connected_components(self.VN.tree))))
        else:
            print(f'doing split after pruning for iter_{iter}')

        for node in self.stable_nodes:
            
            neighbors = np.array(list(self.VN.tree.neighbors(node)))
            leaf_mask = neighbors >= len(self.stable_nodes)
            if np.sum(leaf_mask) == 0:
                continue

            
            
            
            
            
            
            
            
            
            

            neighbor_leaves = neighbors[leaf_mask]

            neighbor_leaf_locs = [self.VN.tree.nodes[n]['loc'] for n in neighbor_leaves]
            locs = np.concatenate((neighbor_leaf_locs, np.array([self.VN.tree.nodes[node]['loc']])))
            opt_point_loc = self.get_centroid(locs)

            neighbor_leaf_locs = np.array(neighbor_leaf_locs)
            avg_dist = np.mean(np.linalg.norm(neighbor_leaf_locs - opt_point_loc, axis=1))

            
            
            
            
            

            opt_point = self.VN.add_branching_point(opt_point_loc, pseudo_fixed=True)

            

            all_radius = np.array([self.VN.tree[node][i]['radius'] for i in neighbor_leaves])
            all_flow = np.array([self.VN.tree[node][i]['flow'] for i in neighbor_leaves])

            for i in neighbor_leaves:
                self.VN.add_vessel(opt_point, i, self.VN.tree[node][i]['radius'], self.VN.tree[node][i]['flow'])
                self.VN.tree.remove_edge(node, i)

            self.VN.add_vessel(node, opt_point, np.sum(np.power(all_radius, 3)) ** (1 / 3), np.sum(all_flow))

        print('finish initialize')
        self.VN.save_vtk(os.path.join(self.work_dir, 'whole_structure_initialize.vtk'))

        

    def relax(self, node):
        """
        A local operator. Finds the optimal location of a branching point.

        Parameters
        --------------------
        node -- branching point
        """

        neighbors = list(self.VN.tree.neighbors(node))
        neighbor_num = len(neighbors)
        if neighbor_num <= 1:
            return

        root = list(self.VN.tree.predecessors(node))

        neighbors = root + neighbors

        
        neighbor_radii = np.array([self.VN.tree[node][n]['radius'] if n not in root
                                   else self.VN.tree[n][node]['radius'] for n in neighbors])

        neighbor_locs = np.array([self.VN.tree.nodes[n]['loc'] for n in neighbors], dtype=float)

        neighbor_flows = np.array([self.VN.tree[node][n]['flow'] if n not in root
                                   else self.VN.tree[n][node]['flow'] for n in neighbors])

        local_optimizer = self.optimizer(neighbor_locs, neighbor_radii, self.VN.tree.nodes[node]['loc'],
                                         self.cost_mode, optimize_r=self.optimize_r,
                                         min_r=self.min_r, max_r=self.max_r,
                                         w1=self.w1, w2=self.w2, w3=self.w3,
                                         include_surface_area=self.include_surface_area,
                                         vsize=self.vsize,
                                         flow=neighbor_flows, viscosity=self.mu,
                                         )

        new_loc, new_radii, cost = local_optimizer.optimize()

        self.VN.move_node(node, new_loc)

        i = 0
        for n in neighbors:
            if n in root:
                edge = (n, node)
            else:
                edge = (node, n)
            self.VN.update_radius_and_flow(edge, new_radii[i], flow=neighbor_flows[i])

            i += 1

        if self.remove_negative and np.any(new_radii <= 0):
            print('something wrong, need to remove negative radius')
            

            self.VN.remove_negative_raidus()

    def final_merge_prune(self):
        self.VN.reorder_nodes()

        merge_count = 0

        for node in self.VN.tree:
            all_neighbors = list(self.VN.tree.neighbors(node)) + list(self.VN.tree.predecessors(node))

            neighbor_edge_lengths = [self.VN.tree[node][n]['length'] for n in all_neighbors]
            shortest_edge_idx = np.argmin(neighbor_edge_lengths)

            if len(neighbor_edge_lengths) == 1:
                return

            second_shortest_edge_idx = np.argsort(neighbor_edge_lengths)[1]

            if neighbor_edge_lengths[shortest_edge_idx] / \
                    neighbor_edge_lengths[second_shortest_edge_idx] <= self.merge_threshold:

                if len(list(self.VN.tree.neighbors(all_neighbors[shortest_edge_idx]))):
                    self.VN.tree.remove_node(all_neighbors[shortest_edge_idx])
                    self.VN.leaves = range(self.leaves[0], self.leaves[-1])
                    self.leaves = self.VN.leaves
                    print('removing merge leaf')
                    self.VN.reorder_nodes()
                    merge_count += 1

        print(f'final merged {merge_count} nodes')

    def merge_fixed(self, node):

        all_neighbors = list(self.VN.tree.neighbors(node)) + list(self.VN.tree.predecessors(node))

        num_fix_neighbors = [n for n in all_neighbors if self.VN.tree.nodes[n]['fixed']]

        if len(num_fix_neighbors) != 2:
            return

        neighbor_edge_lengths = [self.VN.tree[node][n]['length'] for n in all_neighbors]

        shortest_edge_idx = np.argmin(neighbor_edge_lengths)

        if len(neighbor_edge_lengths) == 0 or not self.VN.tree.nodes[all_neighbors[shortest_edge_idx]]['fixed']:
            return


        second_shortest_edge_idx = [self.VN.tree[node][n]['length'] for n in all_neighbors
                                    if not self.VN.tree.nodes[all_neighbors[shortest_edge_idx]]['fixed']]

        if len(second_shortest_edge_idx) == 0:
            return
        second_shortest_edge_idx = np.argmin(second_shortest_edge_idx)

        shortest = all_neighbors[shortest_edge_idx]

        if len(neighbor_edge_lengths) == 1:
            return

        if neighbor_edge_lengths[shortest_edge_idx] / \
                neighbor_edge_lengths[second_shortest_edge_idx] <= self.merge_threshold:

            if shortest in self.VN.leaves:

                return

            else:

                self.VN.fixed = range(len(self.VN.fixed) - 1)
                self.VN.leaves = range(len(self.VN.fixed), self.VN.leaves[-1])

                self.stable_nodes = self.VN.fixed
                self.leaves = self.VN.leaves

                if self.VN.tree.nodes[shortest]['level'] < self.VN.tree.nodes[node]['level']:
                    self.VN.merge(node, shortest)
                    self.VN.reorder_removed_leaf(shortest)

                else:
                    self.VN.merge(shortest, node)
                    self.VN.reorder_removed_leaf(node)

                print('removing merge fixed')

                self.VN.reorder_nodes()

    def merge(self, node):
        """
        A local operator. Determines whether merging of a branching point is needed.

        Parameters
        --------------------
        node -- branching point
        """
        root = list(self.VN.tree.predecessors(node))
        all_neighbors = list(self.VN.tree.neighbors(node)) + root
        neighbor_edge_lengths = [self.VN.tree[node][n]['length'] if n not in root
                                 else self.VN.tree[n][node]['length'] for n in all_neighbors]
        shortest_edge_idx = np.argmin(neighbor_edge_lengths)

        if len(neighbor_edge_lengths) <= 1:
            return

        second_shortest_edge_idx = np.argsort(neighbor_edge_lengths)[1]

        if neighbor_edge_lengths[shortest_edge_idx] / \
                neighbor_edge_lengths[second_shortest_edge_idx] <= self.merge_threshold:

            if self.VN.tree.nodes[all_neighbors[shortest_edge_idx]]['leaf']:
                return

            elif self.VN.tree.nodes[all_neighbors[shortest_edge_idx]]['fixed']:
                self.VN.merge(all_neighbors[shortest_edge_idx], node)
            else:
                self.VN.merge(node, all_neighbors[shortest_edge_idx])

    def split(self, node, fixed=False):
        """
        A local operator. Determines whether spliting of a branching point is needed.
        """

        neighbors = list(self.VN.tree.neighbors(node))

        if fixed:
            neighbors = [i for i in neighbors if not self.VN.tree.nodes[i]['fixed'] or self.VN.tree.nodes[i]['level'] == -1]

        neighbor_num = len(neighbors)

        if neighbor_num <= 2:
            return

        neighbors = np.array(neighbors)
        try:
            root = list(self.VN.tree.predecessors(node))[0]
        except:
            print('ok')
        edges_to_split, max_rs = self.split_two_edges(node, fixed=fixed)


        all_local_derivative = np.array(self.local_derivative(node, neighbors, no_sum=True))
        all_r_s = np.array([self.VN.tree[node][n]['radius'] ** self.VN.c for n in neighbors])

        all_f_s = np.array([self.VN.tree[node][n]['flow'] for n in neighbors])

        while len(edges_to_split) < neighbor_num:
            ranges = np.array(range(neighbor_num))
            ranges = np.setdiff1d(ranges, edges_to_split)
            if len(ranges) == 0:
                break
            neighbo_cur = np.array(edges_to_split)
            neighbo_cur = np.expand_dims(neighbo_cur, 0)
            neighbo_cur = np.repeat(neighbo_cur, len(ranges), 0)
            neighbo_cur = np.column_stack([neighbo_cur, np.array(ranges)])

            loc_grad = all_local_derivative[neighbo_cur]

            loc_grad = np.sum(loc_grad, axis=1)
            loc_grad_norm = np.linalg.norm(loc_grad, axis=1)

            remaining_nodes = neighbo_cur

            radiuses = all_r_s[remaining_nodes]
            radiuses = np.sum(radiuses, axis=1)
            
            radiuses = np.power(radiuses, 1 / self.VN.c)

            flows = all_f_s[remaining_nodes]
            flows = np.sum(flows, axis=1)

            rs_es = self.rupture_strength(loc_grad_norm, radiuses, flows)

            if len(rs_es) == 0:
                break

            cur_max_rs = np.max(rs_es)

            target_idx = ranges[np.argmax(rs_es)] if cur_max_rs > max_rs else None
            max_rs = np.max(rs_es) if cur_max_rs > max_rs else max_rs

            if target_idx != None:
                edges_to_split.append(target_idx)
            else:
                break

        if max_rs <= 0 or len(edges_to_split) < 2 or (neighbor_num - len(edges_to_split)) == 0:

            if len(edges_to_split) < 2:
                print('this should never happen, always have 2 to split')

            
            

            if neighbor_num == len(edges_to_split):
                print('all wants to split, this will move to middle, but would not change neighbor number')

            return

        chosen_nodes = np.array(neighbors)[edges_to_split]
        chosen_locs = [self.VN.tree.nodes[n]['loc'] for n in chosen_nodes]

        node2_loc = self.get_centroid(chosen_locs + [self.VN.tree.nodes[node]['loc']])

        if np.all(node2_loc == self.VN.tree.nodes[node]['loc']):
            return

        self.VN.split(node, node2_loc, chosen_nodes)


    def local_opt(self, i):
        """
        Optimizes each branching point with local operators.

        Parameters
        --------------------
        node -- branching point
        i -- current iteration
        """

        
        all_nodes = list(self.VN.tree.nodes)[::-1]

        for n in all_nodes:
            if n not in self.VN.tree.nodes or self.VN.tree.nodes[n]['fixed'] or len(
                    list(self.VN.tree.neighbors(n))) == 0:
                continue
            self.relax(n)

        print('finished relax ... doing merge... ', end=',')
        print("cost middle point after relax: " + '{:.5e}'.format(self.global_cost()), end=',')

        if i == self.max_iter_2:
            return
        for n in all_nodes:
            if n not in self.VN.tree.nodes or self.VN.tree.nodes[n]['fixed'] or len(
                    list(self.VN.tree.neighbors(n))) <= 1:
                continue
            self.merge(n)
        
        
        
        
        
        print("after merge: " + '{:.5e}'.format(self.global_cost()), end=',')
        
        print('finished merge ... doing split... ', end=',')

        all_nodes = list(self.VN.tree.nodes)[::-1]

        for n in all_nodes:
            if n not in self.VN.tree.nodes:
                continue
            if not self.split_fixed:
                if self.VN.tree.nodes[n]['fixed'] or len(
                        list(self.VN.tree.neighbors(n))) == 0:
                    continue
                else:
                    self.split(n)
            else:
                if len(list(self.VN.tree.neighbors(n))) == 0:
                    continue
                elif self.VN.tree.nodes[n]['fixed']:
                    self.split(n, fixed=True)
                else:
                    self.split(n)

        print('done')

    def GCO_opt(self):
        """
        Optimizes the whole network configuration.
        """
        self.VN.reorder_nodes()

        cur_l = self.max_l
        count_l = 0
        cur_iter = 0

        self.total_loss.append(self.global_cost())

        while cur_iter <= self.max_iter_1:
            print("\nItearation %d" % cur_iter)
            diff_flag = True
            i = 0

            
            while diff_flag and i <= self.max_iter_2:
                cost_before = self.global_cost()
                print(f"Itearation {cur_iter} [{i}] num_node={len(self.VN.tree.nodes)}")
                print("cost before: " + '{:.5e}'.format(cost_before))
                self.local_opt(i)
                cost_after = self.global_cost()
                print("cost after: " + '{:.5e}'.format(cost_after))

                self.total_loss.append(cost_after)

                
                
                
                
                
                
                

                if not self.only_save_last:
                    self.VN.save_vtk(os.path.join(self.work_dir, f'Itearation_{cur_iter}_{i}.vtk'))

                if self.save_interm_res:
                    path_name = os.path.join(self.work_dir, f'json_data_{cur_iter}_{i}.pkl')
                    with open(path_name, 'wb') as outp:
                        
                        pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

                diff_flag = (cost_before != cost_after)
                i += 1

            
            self.VN.update_order()
            
            if self.prune_mode == 'HS':
                self.VN.update_order(self.prune_mode)
            cur_level = self.VN.get_max_level(self.prune_mode)

            self.VN.save_vtk(os.path.join(self.work_dir, f'Itearation_{cur_iter}_finished_before_prune.vtk'))


            if cur_level >= self.prune_threshold and cur_iter < self.max_iter_1:

                analytical_level = cur_level - self.prune_threshold

                print(f'analytical_level needs to be pruned = {analytical_level}')

                total_pruned = self.VN.prune(cur_l, self.prune_mode)
                count_l += 1
                self.VN.reconnect(self.avoid_root_recon)

                self.VN.rescale_fixed_radius()

                if total_pruned > 0:
                    print(f'get {total_pruned} pruned of order < {cur_l}')

                    self.split_pruned_from_fixed(iter=cur_iter + 1)


            else:
                print(f'current largest order = {cur_level}, no pruning')

            if count_l % 4 == 0:
                cur_l = 0 if cur_l == 1 else cur_l - 1

            self.VN.save_vtk(os.path.join(self.work_dir, f'Itearation_{cur_iter}_finished.vtk'))

            cur_iter += 1

            self.total_loss.append(self.global_cost())
            self.save_loss()

        self.VN.final_merge()
        self.VN.save_vtk(os.path.join(self.work_dir, f'Final merged.vtk'))

        self.save_loss(final=True)

        
        print("Final connected component: %d" % len(list(nx.weakly_connected_components(self.VN.tree))))
        self.save_results()



    def save_results(self, lab=''):
        """
        Stores the resulting network structure.
        """

        file_id = 13

        coord_file = os.path.join(self.work_dir, f'{lab}_test_1_result_coords.npy')
        connection_file = os.path.join(self.work_dir, f'{lab}_test_1_result_connections.npy')
        radius_file = os.path.join(self.work_dir, f'{lab}_test_1_result_radii.npy')
        order_file = os.path.join(self.work_dir, f'{lab}_test_1_result_HS_order.npy')
        level_file = os.path.join(self.work_dir, f'{lab}_test_1_result_level_order.npy')

        nodes = dict()
        coords = list()
        connections = list()
        radii = list()
        order = list()
        l_order = list()
        self.VN.update_final_order('HS')
        self.VN.update_final_order('level')

        
        
        

        for edge in list(self.VN.tree.edges):
            node1, node2 = edge
            for node in edge:
                if not node in nodes:
                    nodes[node] = len(coords)
                    coords.append(self.VN.tree.nodes[node]['loc'])
                    order.append(self.VN.tree.nodes[node]['HS'])
                    l_order.append(self.VN.tree.nodes[node]['level'])
            connections.append([nodes[node1], nodes[node2]])
            radii.append(abs(self.VN.tree[node1][node2]['radius']))

        np.save(coord_file, coords)
        np.save(connection_file, connections)
        np.save(radius_file, radii)
        print("Save coords, edges and radius.")
        np.save(order_file, order)
        np.save(level_file, l_order)
        print("Save orders.")

    def visualize(self, with_label=False, mode='level'):
        """
        Visualizes the whole network configuration.
        """
        from mayavi import mlab

        dim = len(self.VN.tree.nodes[0]['loc'])
        if dim == 2:
            locs = nx.get_node_attributes(self.VN.tree, 'loc')
            nx.draw(self.VN.tree, locs, with_labels=False, node_size=20)
            
            
            
            
            plt.show()
        else:
            nodes = dict()
            coords = list()
            connections = list()
            labels = list()
            text_scale = list()
            radii = list()
            for edge in list(self.VN.tree.edges):
                node1, node2 = edge
                if not node1 in nodes:
                    nodes[node1] = len(coords)
                    coords.append(self.VN.tree.nodes[node1]['loc'])
                    if node1 in self.stable_nodes:
                        text_scale.append((3, 3, 3))
                    else:
                        text_scale.append((1, 1, 1))
                    if mode == 'HS' or mode == 'level':
                        labels.append(str(self.VN.tree.nodes[node1][mode]))
                    else:
                        labels.append(str(node1))
                if not node2 in nodes:
                    nodes[node2] = len(coords)
                    coords.append(self.VN.tree.nodes[node2]['loc'])
                    if node2 in self.stable_nodes:
                        text_scale.append((3, 3, 3))
                    else:
                        text_scale.append((1, 1, 1))
                    if mode == 'HS' or mode == 'level':
                        labels.append(str(self.VN.tree.nodes[node2][mode]))
                    else:
                        labels.append(str(node2))
                connections.append([nodes[node1], nodes[node2]])
                radii.append(self.VN.tree[node1][node2]['radius'])

            coords = np.array(coords)
            mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            mlab.clf()
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.5, scale_mode='none',
                                colormap='Blues', resolution=60)

            
            i = 0
            for n in connections:
                n1, n2 = n[0], n[1]
                r = radii[i]
                mlab.plot3d([coords[n1][0], coords[n2][0]], [coords[n1][1], coords[n2][1]],
                            [coords[n1][2], coords[n2][2]], tube_radius=r / self.vsize)
                i += 1
            if not with_label:
                mlab.show()
                return

            
            for i in range(len(coords)):
                mlab.text3d(coords[i][0], coords[i][1], coords[i][2], labels[i], scale=text_scale[i])
            mlab.show()

    
    
    

    def split_two_edges(self, node, fixed=False):
        neighbors = list(self.VN.tree.neighbors(node))

        if fixed:
            neighbors = [i for i in neighbors if not self.VN.tree.nodes[i]['fixed'] or self.VN.tree.nodes[i]['level'] == -1]

        neighbor_num = len(neighbors)
        neighbors = np.array(neighbors)

        if neighbor_num <= 2:
            return [], 0
        
        max_rs = 0
        target_edges = []

        all_edges = [[i, j] for i in range(neighbor_num - 1) for j in range(i + 1, neighbor_num)]

        if len(all_edges) == 0:
            return target_edges, max_rs

        all_local_derivative = np.array(self.local_derivative(node, neighbors, no_sum=True))
        all_edges = np.array(all_edges)

        loc_grad = all_local_derivative[all_edges]
        loc_grad = np.sum(loc_grad, axis=1)
        

        loc_grad_norm = np.linalg.norm(loc_grad, axis=1)

        all_r_s = [self.VN.tree[node][n]['radius'] ** self.VN.c for n in neighbors]
        all_r_s = np.array(all_r_s)

        all_f_s = [self.VN.tree[node][n]['flow'] for n in neighbors]
        all_f_s = np.array(all_f_s)

        remaining_nodes = all_edges

        radiuses = all_r_s[remaining_nodes]
        radiuses = np.sum(radiuses, axis=1)
        
        radiuses = np.power(radiuses, 1 / self.VN.c)

        flows = all_f_s[remaining_nodes]
        flows = np.sum(flows, axis=1)

        all_rs_es = self.rupture_strength(loc_grad_norm, radiuses, flows)

        if len(all_rs_es) == 0:
            return target_edges, max_rs

        max_rs = np.max(all_rs_es)
        target_edges = all_edges[np.argmax(all_rs_es)]

        return list(target_edges), max_rs

    def get_centroid(self, node_locs):
        return tuple([sum(x) / len(x) for x in zip(*node_locs)])

    def local_cost(self, edge):
        node1, node2 = edge
        self.VN.tree[node1][node2]['length'] = self.vsize * np.linalg.norm(self.VN.tree.nodes[node1]['loc'] - self.VN.tree.nodes[node2]['loc'])

        mc = np.pi * self.VN.tree[node1][node2]['radius'] ** 2 * self.VN.tree[node1][node2]['length']

        if self.include_surface_area:
            mc += 2 * np.pi * self.VN.tree[node1][node2]['radius'] * self.VN.tree[node1][node2]['length']

        if self.cost_mode == 'MC':
            w1 = self.w1 if self.global_weighting else 1
            return w1 * mc

        else:
            pc = self.VN.tree[node1][node2]['flow']**2 * self.VN.tree[node1][node2]['radius'] ** (-4) * self.VN.tree[node1][node2]['length'] * 8 * self.mu/np.pi
            
            w1 = self.w1 if self.global_weighting else 1
            w2 = self.w2 if self.global_weighting else 1

            

            return w1 * mc + w2 * pc
            

    def local_cost_mc(self, edge):
        node1, node2 = edge
        self.VN.tree[node1][node2]['length'] = self.vsize * np.linalg.norm(self.VN.tree.nodes[node1]['loc'] - self.VN.tree.nodes[node2]['loc'])

        mc = np.pi * self.VN.tree[node1][node2]['radius'] ** 2 * self.VN.tree[node1][node2]['length']

        if self.include_surface_area:
            mc += 2 * np.pi * self.VN.tree[node1][node2]['radius'] * self.VN.tree[node1][node2]['length']
        return mc * self.w1

    def local_cost_pc(self, edge):
        node1, node2 = edge
        self.VN.tree[node1][node2]['length'] = self.vsize * np.linalg.norm(self.VN.tree.nodes[node1]['loc'] - self.VN.tree.nodes[node2]['loc'])

        pc = self.VN.tree[node1][node2]['flow']**2 * self.VN.tree[node1][node2]['radius'] ** (-4) * self.VN.tree[node1][node2]['length'] * 8 * self.mu/np.pi

        return pc * self.w2

    def global_cost(self):
        cost_list = [self.local_cost(edge) for edge in list(self.VN.tree.edges)]

        print(f'mc={np.sum([self.local_cost_mc(edge) for edge in list(self.VN.tree.edges)])}')
        print(f'pc={np.sum([self.local_cost_pc(edge) for edge in list(self.VN.tree.edges)])}')

        return np.sum(cost_list)

    def local_cost_neighbors(self, node, neighbors, no_sum=False):

        neighbor_leaf_locs = [self.VN.tree.nodes[n]['loc'] for n in neighbors]
        testRadius = [self.VN.tree[node][n]['radius'] for n in neighbors]

        locs = np.concatenate((neighbor_leaf_locs, np.array([self.VN.tree.nodes[node]['loc']])))
        opt_point_loc = np.array(self.get_centroid(locs))

        r0 = np.power(np.sum([n ** self.VN.c for n in testRadius]), 1 / self.VN.c)
        l0 = np.linalg.norm(opt_point_loc - self.VN.tree.nodes[node]['loc'])

        if self.cost_mode == 'MC':
            temp = 0.0
            for i in range(len(neighbor_leaf_locs)):
                surface_term = self.vsize * 2 * np.linalg.norm(opt_point_loc - neighbor_leaf_locs[i]) * testRadius[i] if \
                    self.include_surface_area else 0
                temp += self.vsize * np.linalg.norm(opt_point_loc - neighbor_leaf_locs[i]) * (testRadius[i] ** 2) + \
                        surface_term
            temp *= self.w1
        else:
            if self.include_surface_area:
                temp1 = np.array(
                    [self.vsize * np.linalg.norm(opt_point_loc - neighbor_leaf_locs[i]) * testRadius[i] ** 2 +
                     2 * self.vsize * np.linalg.norm(opt_point_loc - neighbor_leaf_locs[i]) * testRadius[i]
                     for i in range(len(neighbor_leaf_locs))])
            else:
                temp1 = np.array(
                    [self.vsize * np.linalg.norm(opt_point_loc - neighbor_leaf_locs[i]) * testRadius[i] ** 2
                     for i in range(len(neighbor_leaf_locs))])

            temp1 = self.w1 * np.sum(temp1)

            temp2 = np.array(
                [testRadius[i] ** 4 / (self.vsize * np.linalg.norm(opt_point_loc - neighbor_leaf_locs[i]))
                 for i in range(len(opt_point_loc))])
            temp2 = 1 / np.sum(temp2) + self.vsize * l0 / r0 ** 4

            temp2 *= self.w2
            return temp1 + temp2

        return temp

    def local_derivative_power(self, node, neighbors, no_sum=False):

        vecs = [1 / self.VN.tree[node][n]['length'] ** 2 for n in neighbors]
        if no_sum:
            return np.array(vecs)
            
        return np.sum(vecs, axis=0)

    def local_derivative(self, node, neighbors, no_sum=False):

        

        if self.cost_mode == 'MC':
            if self.include_surface_area:
                vecs = [np.pi * (self.VN.tree[node][n]['radius'] ** 2 + 2 * self.VN.tree[node][n]['radius']) *
                        ((self.VN.tree.nodes[n]['loc'] - self.VN.tree.nodes[node]['loc']) /
                         np.linalg.norm(self.VN.tree.nodes[node]['loc'] - self.VN.tree.nodes[n]['loc'])) for n in neighbors]
            else:
                vecs = [np.pi * self.VN.tree[node][n]['radius'] ** 2 *
                        ((self.VN.tree.nodes[n]['loc'] - self.VN.tree.nodes[node]['loc']) /
                         np.linalg.norm(self.VN.tree.nodes[node]['loc'] - self.VN.tree.nodes[n]['loc'])) for n in neighbors]

        else:
            if self.include_surface_area:
                vecs = [
                    (self.w1 * np.pi * self.VN.tree[node][n]['radius'] ** 2 + self.w1 * np.pi * 2 * self.VN.tree[node][n]['radius'] +
                     self.w2 * self.VN.tree[node][n]['flow']**2 * (self.VN.tree[node][n]['radius'] ** (-4) * 8 * self.mu/np.pi)
                     ) *
                    ((self.VN.tree.nodes[n]['loc'] - self.VN.tree.nodes[node]['loc']) /
                     np.linalg.norm(self.VN.tree.nodes[node]['loc'] - self.VN.tree.nodes[n]['loc'])) for n in neighbors]
            else:

                vecs = [(self.w1 * self.VN.tree[node][n]['radius'] ** 2 * np.pi +
                         self.w2 * self.VN.tree[node][n]['flow']**2 * (self.VN.tree[node][n]['radius'] ** (-4) * 8 * self.mu/np.pi)
                         ) *
                        ((self.VN.tree.nodes[n]['loc'] - self.VN.tree.nodes[node]['loc']) /
                         np.linalg.norm(self.VN.tree.nodes[node]['loc'] - self.VN.tree.nodes[n]['loc'])) for n in neighbors]

        if no_sum:
            return vecs
            

        return np.sum(vecs, axis=0)

    def rupture_strength(self, pull_force, new_edge_r, new_edge_f):
        surface_term = 2 * new_edge_r * np.pi if self.include_surface_area else 0
        
        if self.cost_mode == 'MC':
            return pull_force - np.pi * new_edge_r ** 2 - surface_term
        else:
            return pull_force - self.w1 * np.pi * new_edge_r ** 2 - self.w2 * new_edge_f**2 * (
                    new_edge_r ** (-4)) * 8 * self.mu/np.pi - self.w1 * surface_term

    def save_loss(self, final=False):
        
        plt.figure()
        plt.plot(self.total_loss)
        save_name = os.path.join(self.work_dir, 'loss_iter.png')
        if final:
            plt.title('final_loss = ' + '{:.5e}'.format(self.total_loss[-1]))

        plt.savefig(save_name)

        plt.figure()
        plt.plot(self.total_loss)
        plt.yscale('log')
        save_name = os.path.join(self.work_dir, 'log_loss_iter.png')
        if final:
            plt.title('final_loss = ' + '{:.5e}'.format(self.total_loss[-1]))

        plt.savefig(save_name)

        plt.clf()

        
        


    def final_save_vtk(self, file='vascularTree.vtk', include_line=True, lab=''):

        coord_file = os.path.join(self.work_dir, f'{lab}_test_1_result_coords.npy')
        connection_file = os.path.join(self.work_dir, f'{lab}_test_1_result_connections.npy')
        radius_file = os.path.join(self.work_dir, f'{lab}_test_1_result_radii.npy')
        order_file = os.path.join(self.work_dir, f'{lab}_test_1_result_HS_order.npy')
        level_file = os.path.join(self.work_dir, f'{lab}_test_1_result_level_order.npy')

        self.connections = np.load(connection_file)
        self.radii = np.load(radius_file) + 1e-10
        self.coords = np.load(coord_file)
        self.node_HS = np.array(np.load(order_file), dtype=int)
        self.node_level = np.array(np.load(level_file), dtype=int)

        
        
        


        points_dict = OrderedDict()
        radius_list = []
        points_list = []
        edges_inds = []

        i = 0
        for n in self.connections:
            n1, n2 = n[0], n[1]
            r = self.radii[i]
            radius_list.append(r)

            if n1 not in points_dict:
                points_dict[n1] = len(points_dict)
                points_list.append(self.coords[points_dict[n1]])

            if n2 not in points_dict:
                points_dict[n2] = len(points_dict)
                points_list.append(self.coords[points_dict[n2]])

            edges_inds.append([points_dict[n1], points_dict[n2]])

            i += 1

        radius_list = np.array(radius_list)
        points_list = np.array(points_list)
        edges_inds = np.array(edges_inds)

        lines = np.hstack([np.ones((len(edges_inds), 1), dtype=np.uint8) * 2, edges_inds])

        mesh = pyvista.PolyData(points_list,
                                lines=lines if include_line else None
                                )

        if len(radius_list) > 0 and len(radius_list) == mesh.n_lines:
            mesh.cell_data['radius'] = radius_list

        self.to_node_radii()
        mesh.point_data['node_radius'] = self.node_radii

        mesh.save(file)

    def to_node_radii(self):

        node_radii = np.zeros(len(self.coords))

        for i, n in enumerate(self.connections):
            n1, n2 = n[0], n[1]

            a, b = (n1, n2) if self.node_level[n1] > self.node_level[n2] else (n2, n1)

            r = self.radii[i]

            node_radii[b] = r

        print(np.sum(node_radii == 0))

        assert np.sum(node_radii == 0) == 1

        node_radii[node_radii == 0] = np.max(node_radii)

        self.node_radii = node_radii

