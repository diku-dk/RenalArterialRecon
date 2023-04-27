
from GCODi import *

class GCOForest(GCO):
    """
    Given a set of fixed nodes (root nodes) and leaf nodes, generates an optimized vascular network that
    connects all leaf nodes with the root nodes using Global Constructive Optimization (GCO).
    """

    def __init__(self, *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def GCO_opt(self):
        """
        Optimizes the network configuration.
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
            self.VN.update_forest_order('level')

            if 'HS' in self.prune_mode.upper():
                self.VN.update_order(self.prune_mode)
                self.VN.update_forest_order(self.prune_mode)

            cur_level = self.VN.get_max_forest_level(self.prune_mode)

            self.VN.save_vtk(os.path.join(self.work_dir, f'Itearation_{cur_iter}_finished_before_prune.vtk'))

            print(f'cur_level={cur_level}')

            assert nx.is_tree(self.VN.tree)

            if cur_level >= self.prune_threshold and cur_iter < self.max_iter_1:

                analytical_level = cur_level - self.prune_threshold

                print(f'analytical_level needs to be pruned = {analytical_level}')

                total_pruned = self.VN.forest_prune(cur_l, self.prune_mode, self.prune_threshold, self.too_few_prune)

                count_l += 1

                self.VN.forest_reconnect(self.avoid_root_recon)

                if total_pruned > 0:
                    print(f'get {total_pruned} pruned of order < {cur_l}')


            else:
                print(f'current largest order = {cur_level}, no pruning')


            if count_l % 2 == 0:
                cur_l += 1

            self.VN.save_vtk(os.path.join(self.work_dir, f'Itearation_{cur_iter}_finished.vtk'))

            cur_iter += 1

            self.total_loss.append(self.global_cost())
            self.save_loss()

        self.VN.final_merge()
        self.VN.save_vtk(os.path.join(self.work_dir, f'Final merged.vtk'))

        self.save_loss(final=True)

        print("Final connected component: %d" % len(list(nx.weakly_connected_components(self.VN.tree))))

        self.save_results()

        self.final_merge_prune()
        self.VN.save_vtk(os.path.join(self.work_dir, f'Final merged twice.vtk'))

        self.VN.remove_intermediate()
        self.save_results(lab='inter_removed')
        self.VN.save_vtk(os.path.join(self.work_dir, f'Final merged inter_removed.vtk'))

        self.final_save_vtk(file='final.vtk', lab='inter_removed')


if __name__ == '__main__':


    f_coords, new_edge_list, no_branch = [], [], [] # these are only needed if no vtk files exist

    n_outer_loop = 35
    merge_threshold = 0.15

    vsize = 22.6
    f_init = 10
    w1, w2, w3 = 5.5e-8, 1, 0,

    r_leaf = np.array([10.08, 0.14])

    prune_threshold, max_l = 8, 2
    prune_mode = 'level'
    optimizer = GD_Optimizer
    use_C = False
    cost_mode = 'PC'
    optimize_r = False
    avoid_root_recon = False
    rescale_radius = True
    only_save_last = True
    split_fixed = True
    global_weighting = True
    include_surface_area = False
    r_init = 30 * vsize  # these are only needed if no vtk files exist


    root_loc = [588, 217, 650]

    pt_file = 'final_main_artery.vtk'

    random_points_file = 'sampled_terminal_nodes.vtk'

    r_coords = pyvista.read(random_points_file).points
    r_coords = np.array(r_coords)

    split_fixed_str = 'split_fixed' if split_fixed else ''

    n_inner_loop, too_few_prune = 0, 0

    t_flow, mu = 1.167e11/len(r_coords), 3.6e-15

    optimizer = GD_OptimizerUnit

    n_outer_loop = 25


    g = GCOForest(f_coords, r_coords, r_init=r_init, f_init=f_init, p_init=2, edge_list=new_edge_list,
            work_dir=f'{prune_mode}_{split_fixed_str}_{pt_file}_{cost_mode.lower()}_pruned_{prune_threshold}_{max_l}_w1={w1}_w2={w2}'
                     f'vsize_{vsize}',
            r_leaf=r_leaf, max_l=max_l, n_outer_loop=n_outer_loop,
            optimize_r=optimize_r, cost_mode=cost_mode, use_C=use_C, remove_negative=True,
            prune_threshold=prune_threshold,  w1=w1, w2=w2, w3=w3,
            global_weighting=global_weighting, vsize=vsize, merge_threshold=merge_threshold, optimizer=optimizer,
            no_branch=no_branch, prune_mode=prune_mode, avoid_root_recon=avoid_root_recon,
            pt_file=pt_file, root_loc=root_loc, rescale_radius=rescale_radius,
            only_save_last=only_save_last, split_fixed=split_fixed,
            n_inner_loop=n_inner_loop, too_few_prune=too_few_prune, only_connect_non_branching=True, t_flow=t_flow, mu=mu
            )

    g.GCO_opt()

