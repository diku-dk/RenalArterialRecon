
from scipy.stats import norm

from VesselGen.result_analysis.simulate_fow_pressure import *
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import scipy

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class VtkMorphAnalyer(VtkNetworkAnalysis):
    def __init__(self, pt_file=None, root_loc=None, vsize=22.6):
        super().__init__(pt_file, root_loc, vsize)

    def build_morphologies(self):

        self.update_order('HS')

        nodes = dict()
        coords = list()
        connections = list()
        radii = list()
        flow = list()
        pressure = list()
        pressure_end = list()

        order = list()
        l_order = list()
        for edge in list(self.tree.edges):
            node1, node2 = edge
            for node in edge:
                if node not in nodes:
                    nodes[node] = len(coords)
                    coords.append(self.tree.nodes[node]['loc'])
                    order.append(self.tree.nodes[node]['HS'])
                    l_order.append(self.tree.nodes[node]['level'])

                    if 'pressure_mmhg' in self.tree.nodes[node].keys():
                        pressure.append(abs(self.tree.nodes[node]['pressure_mmhg']))

            connections.append([nodes[node1], nodes[node2]])
            radii.append(abs(self.tree[node1][node2]['radius']))
            flow.append(abs(self.tree[node1][node2]['flow']))

            if 'pressure_mmhg' in self.tree.nodes[node2].keys():
                end_node = node2 if self.tree.nodes[node2]['level'] < self.tree.nodes[node1]['level'] else node1
                pressure_end.append(self.tree.nodes[end_node]['pressure_mmhg'])

            self.tree[node1][node2]['edge_HS'] = min(self.tree.nodes[node1]['HS'], self.tree.nodes[node2]['HS'])
            self.tree[node1][node2]['root_order'] = -1

        self.coords = np.array(coords)
        self.radii = np.array(radii)
        self.pressure = np.array(pressure)
        self.node_HS = np.array(order)
        self.node_level = np.array(l_order)
        self.connections = np.array(connections)
        self.flow = np.array(flow)
        self.pressure_end = np.array(pressure_end)
        self.HS = []
        self.levels = []
        self.lengths = []

        self.edge_mapping = {}

        for i, n in enumerate(self.connections):
            n1, n2 = n[0], n[1]
            self.HS.append(min(self.node_HS[n1], self.node_HS[n2]))
            self.levels.append(min(self.node_level[n1], self.node_level[n2]))
            a, b = (n1, n2) if self.node_level[n1] > self.node_level[n2] else (n2, n1)
            self.edge_mapping[f'{a}_{b}'] = i
            self.lengths.append(np.linalg.norm(self.coords[n1] - self.coords[n2]) * self.vsize)


        self.HS = np.array(self.HS, dtype=int)
        self.levels = np.array(self.levels, dtype=int)
        self.lengths = np.array(self.lengths)
        self.lengths = self.lengths + 1e-5

        if -1 in self.HS:
            assert np.sum(self.HS == -1) == 1
            self.HS[np.where(self.HS == -1)[0][0]] = np.max(self.HS) + 1

        self.HS = self.HS - 1

        self.max_HS = np.max(self.HS)

        for edge in list(self.tree.edges):
            node1, node2 = edge
            self.tree[node1][node2]['edge_HS'] = self.max_HS if self.tree[node1][node2]['edge_HS'] == -1 \
                else self.tree[node1][node2]['edge_HS'] - 1


        if -1 in self.levels:
            assert np.sum(self.levels == -1) == 1
            self.levels[np.where(self.levels == -1)[0][0]] = np.max(self.levels) + 1


    def find_root(self, node):
        neighbors = list(self.tree.neighbors(node))
        neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
        root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
        root = neighbors[np.where(neighbor_orders == root)[0][0]]
        return root

    def label_branching_pattern(self):

        self.root_hs = []

        for node in self.tree.nodes:
            neighbors = list(self.tree.neighbors(node))
            neighbors = np.array(neighbors)

            if len(neighbors) == 1 and not self.tree.nodes[node]['root']:
                root = self.find_root(node)
                root_root = self.find_root(root)

                self.root_hs.append(self.tree[root_root][root]['edge_HS'])

                self.tree[root][node]['root_order'] = self.tree[root_root][root]['edge_HS']

        for edge in self.tree.edges:
            a, b = edge
            if self.tree.nodes[a]['root'] or self.tree.nodes[b]['root']:
                self.tree[a][b]['root_order'] = -1
                continue

            if self.tree.nodes[a]['level'] < self.tree.nodes[b]['level']:
                a, b = b, a


            root = self.find_root(a)

            self.tree[a][b]['root_order'] = self.tree[root][a]['edge_HS']

    def radius_real(self, save=False, save_path=None):
        """
        Generates a boxplot for radius v.s. Strahler order.
        """

        mean = [10.08, 13.9, 20.06, 29.87, 39.29, 44.23, 53.87, 86.15, 139.83, 191.42, 216.10]
        std = [0.14,     3.8,   6.9,   0.35,   1.08, 9.81, 12.51, 24.06,   20.11, 17.79, 4.74]
        mean = np.array(mean)
        std = np.array(std)

        self.radius_real_avg = np.copy(mean)


        # mean *= 2
        # std *= 2

        title = 'Radius v.s. Strahler Order'
        ylabel = 'Mean Radius'

        plt.errorbar(np.arange(11), mean, yerr=std, fmt='o', color='black', capsize=5)
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)

        plt.savefig("radius_real.pdf", format="pdf", bbox_inches="tight")

        plt.show()


    def plot_everything(self):
        raw_data = []
        for i in range(len(self.connections)):
            hs = self.HS[i]
            r = self.radii[i]
            raw_data.append([hs, r])

        avg = []
        std = []
        raw_data = np.array(raw_data)
        for i in sorted(np.unique(self.HS)):
            cur = raw_data[raw_data[:, 0] == i][:, 1]
            avg.append(np.mean(cur))
            std.append(np.std(cur))

        plt.figure()
        title = 'Radius v.s. Strahler Order'
        plt.errorbar(np.arange(0, self.max_HS + 1), avg, yerr=std, fmt='o',
                     # color='blue',
                     capsize=5, label='simulation')

        mean = [10.08, 13.9, 20.06, 29.87, 39.29, 44.23, 53.87, 86.15, 139.83, 191.42, 216.10]
        std = [0.14,     3.8,   6.9,   0.35,   1.08, 9.81, 12.51, 24.06,   20.11, 17.79, 4.74]
        mean = np.array(mean)
        std = np.array(std)

        plt.errorbar(np.arange(11), mean, yerr=std, fmt='D',
                     # color='black',
                     capsize=5, label='measurements')
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel('Mean Radius')
        plt.legend()

        plt.savefig("radius_everything.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        res = scipy.stats.pearsonr(mean, avg)
        print(f' radius pearson = {res}')

        p = np.polyfit(np.arange(11), np.log(mean), 1)
        y = np.exp(np.polyval(p, np.arange(11)))
        plt.plot(np.arange(11), y,color='#ff7f0e',)
        plt.errorbar(np.arange(11), mean, yerr=std,
                     fmt='d',
                     color='#ff7f0e',
                     label='Nordsletten',
                     capsize=5)

        p = np.polyfit(np.arange(11), np.log(avg), 1)
        y = np.exp(np.polyval(p, np.arange(11)))
        plt.plot(np.arange(11), y, color='#1f77b4',)
        plt.errorbar(np.arange(11), avg, yerr=std,
                     fmt='o',
                     color='#1f77b4',
                     label='simulation',
                     capsize=5)

        plt.yscale('log')
        plt.autoscale()
        plt.legend()
        plt.savefig("log_radius.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        raw_data = []
        all_length = self.all_branch_length()
        for i in range(0, self.max_HS + 1):
            for l in all_length[i - 1]:
                raw_data.append([i, l])

        avg = [np.mean(i) for i in all_length]
        std = [np.std(i) for i in all_length]

        plt.figure()
        plt.errorbar(np.arange(0, self.max_HS + 1), avg, yerr=std, fmt='o',
                     label='simulation',
                     capsize=5
                     )

        mean = [0.312, 0.423, 0.404, 0.656, 1.001, 0.511, 1.031, 2.516, 8.975, 1.440, 0.185]
        std = [0.285, 0.283, 0.390, 0.286, 0.216, 0.00, 0.674, 2.053, 1.331, 0.647, 0]
        mean = np.array(mean)
        std = np.array(std)

        mean *= 1e3
        std *= 1e3

        title = 'Length v.s. Strahler Order'
        ylabel = 'Mean Length'

        plt.errorbar(np.arange(11), mean, yerr=std, fmt='D',
                     capsize=5, label='measurements')
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)
        plt.legend()

        plt.savefig("length_everything.pdf", format="pdf", bbox_inches="tight")
        plt.show()


        hs, count = np.unique(self.HS, return_counts=True)
        raw_data = {'x': hs, 'y': np.log(count)}
        df = pd.DataFrame(raw_data, index=hs)

        sns.regplot('x', 'y', data=df, fit_reg=True, label='simulation', scatter_kws={"marker": "o",
                                                                                # "color": "blue"
                                                                 # "s": 100
                                                                 })
        mean = [29566, 13070, 4373, 1245, 578, 247, 90, 21, 6, 3, 1]
        mean = np.array(mean)

        raw_data = {'x': np.arange(11), 'y': np.log(mean)}
        df = pd.DataFrame(raw_data, index=np.arange(11))
        title = 'Log Strahler Order Frequency'
        ylabel = 'Log Frequency'

        sns.regplot('x', 'y', data=df, fit_reg=True, label='measurements', marker='D')

        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig("freq_everything.pdf", format="pdf", bbox_inches="tight")

        plt.show()


        res = scipy.stats.pearsonr(mean, count, alternative='two-sided')
        # res = scipy.stats.pearsonr(mean, count, alternative='greater')
        print(f' freq pearson = {res}')



        hs, count = np.unique(self.HS, return_counts=True)

        cross_sections = np.array([np.sum(np.pi * self.radii[self.HS == hs[i]] ** 2) for i in range(len(hs))])

        plt.plot(np.sort(np.unique(hs)), cross_sections/1e6, marker='o', color='k')
        plt.title('cross section v strahlers order')
        plt.xlabel('Strahler Order')
        plt.ylabel('Total Cross Section Area')
        plt.savefig("cross_section.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        cs_norsletten = np.array([9.4286, 7.9248, 5.5214, 3.4824, 2.8, 1.51, 0.80952, 0.54866, 0.3577, 0.3333, 0.13673])

        plt.plot(np.sort(np.unique(hs)), cross_sections/1e6, marker='o',  label='simulation')
        plt.plot(np.sort(np.unique(hs)), cs_norsletten, marker='D', label='measurements')
        plt.legend()
        plt.title('cross section v strahlers order')
        plt.xlabel('Strahler Order')
        plt.ylabel('Total Cross Section Area')
        plt.savefig("cross_section_both.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        res = scipy.stats.pearsonr(cs_norsletten, cross_sections)
        print(f' cross section pearson = {res}')

        raw_data = {'x': hs, 'y': count}
        df = pd.DataFrame(raw_data, index=hs)
        title = 'Strahler Order Frequency'
        ylabel = 'Frequency'

        sns.regplot('x', 'y', data=df, fit_reg=False, scatter_kws={"marker": "o", "color": "k"
                                                                  # "s": 100
                                                                  })
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)
        plt.autoscale()
        plt.savefig("freq.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        raw_data = {'x': hs, 'y': np.log(count)}
        df = pd.DataFrame(raw_data, index=hs)
        title = 'Log Strahler Order Frequency'
        ylabel = 'Log Frequency'

        sns.regplot('x', 'y', data=df, fit_reg=True, scatter_kws={"marker": "o", "color": "k"
                                                                 # "s": 100
                                                                 })
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)
        plt.autoscale()


        plt.savefig("log_freq.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        raw_data = {'x': hs, 'y': count}
        df = pd.DataFrame(raw_data, index=hs)
        p = np.polyfit(raw_data['x'], np.log(raw_data['y']), 1)
        y = np.exp(np.polyval(p, raw_data['x']))
        plt.plot(raw_data['x'], y)
        plt.scatter(raw_data['x'], raw_data['y'], marker='o', color='k')
        plt.yscale('log')
        plt.autoscale()
        plt.savefig("log_freq_scale.pdf", format="pdf", bbox_inches="tight")
        plt.show()


        raw_data = {'x': hs, 'y': count}
        p = np.polyfit(raw_data['x'], np.log(raw_data['y']), 1)
        y = np.exp(np.polyval(p, raw_data['x']))
        plt.plot(raw_data['x'], y)
        plt.scatter(raw_data['x'], raw_data['y'], marker='o', label='simulation')

        mean = [29566, 13070, 4373, 1245, 578, 247, 90, 21, 6, 3, 1]
        std = [5965, 2293, 664, 198, 71, 23, 6, 1, 0, 0, 0]
        raw_data = {'x': np.arange(11), 'y': mean}
        df = pd.DataFrame(raw_data)
        p = np.polyfit(raw_data['x'], np.log(raw_data['y']), 1)
        y = np.exp(np.polyval(p, raw_data['x']))
        plt.plot(raw_data['x'], y)
        plt.errorbar(np.arange(11), mean, yerr=std,  fmt='D',
                     marker='D',
                     label='measurements', color='#ff7f0e',
                     capsize=5
                     )

        plt.yscale('log')
        plt.autoscale()
        plt.legend()
        plt.savefig("log_freq_scale_both.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        self.label_branching_pattern()
        hs, count = np.unique(self.root_hs, return_counts=True)

        plt.plot(hs, count, marker='o', color='k')
        plt.title('Root Strahlers order frequency')
        plt.xlabel('Root Strahlers order')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.savefig("root_strahler.pdf", format="pdf", bbox_inches="tight")
        print(f'root order frequency = {count[np.argsort(hs)]}')
        plt.show()

        plt.bar(hs, count,
                width=0.9
                )
        plt.title('Root Strahlers order frequency')
        plt.xlabel('Root Strahlers order')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.savefig("root_strahler_hist1.pdf", format="pdf", bbox_inches="tight")
        print(f'root order frequency = {count[np.argsort(hs)]}')
        plt.show()

        connectivity = {}

        for i in range(self.max_HS + 1):
            connectivity[i] = []

        for edge in self.tree.edges:
            a, b = edge
            cur_HS = self.tree[a][b]['edge_HS']
            connectivity[cur_HS] =  connectivity[cur_HS] + [self.tree[a][b]['root_order']]

        for cur_HS, root_hs in connectivity.items():
            hs, count = np.unique(root_hs, return_counts=True)

            plt.ticklabel_format(style='plain', axis='x', useOffset=False)

            plt.plot(hs, count, marker='o', color='k')
            plt.title(f'Root Strahlers order frequency for order {cur_HS}')
            plt.xlabel('Root Strahlers order')
            plt.ylabel('Frequency')
            plt.yscale('log')
            plt.xticks(np.arange(np.min(hs), np.max(hs) + 0.1))
            plt.autoscale()
            plt.savefig(f"root_strahler_{cur_HS}.pdf", format="pdf", bbox_inches="tight")
            print(f'root order frequency = {count[np.argsort(hs)]}')
            plt.show()


    def length_real(self, save=False, save_path=None):
        """
        Generates a boxplot for radius v.s. Strahler order.
        """

        mean = [0.312, 0.423, 0.404, 0.656, 1.001, 0.511, 1.031, 2.516, 8.975, 1.440, 0.185]
        std = [0.285, 0.283, 0.390, 0.286, 0.216, 0.00, 0.674, 2.053, 1.331, 0.647, 0]
        mean = np.array(mean)
        std = np.array(std)

        mean *= 1e3
        std *= 1e3

        self.length_real_avg = np.copy(mean)


        title = 'Length v.s. Strahler Order'
        ylabel = 'Mean Length'

        plt.errorbar(np.arange(11), mean, yerr=std, fmt='o', color='black', capsize=5)
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)

        plt.ylim((0, 1.05e4))

        plt.savefig("length_real.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def strahler_real(self):
        """
        Generates a boxplot for radius v.s. Strahler order.
        """

        mean = [29566, 13070, 4373, 1245, 578, 247, 90, 21, 6, 3, 1]
        std = [5965,    2293, 664,  198,    71, 23, 6,  1, 0, 0, 0]
        mean = np.array(mean)
        std = np.array(std) + 1e-10

        title = 'Strahler Order'
        ylabel = 'Estimated No.'

        plt.errorbar(np.arange(11), mean, yerr=std, fmt='o', color='black', capsize=5)
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)

        plt.savefig("freq_real.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        self.strahler_real_avg = np.copy(mean)

        plt.figure()

        raw_data = {'x': np.arange(11), 'y': np.log(mean)}
        df = pd.DataFrame(raw_data, index=np.arange(11))
        title = 'Log Strahler Order Frequency'
        ylabel = 'Log Frequency'

        sns.regplot('x', 'y', data=df, fit_reg=True, scatter_kws={"marker": "D",
                                                                 # "s": 100
                                                                 })

        plt.errorbar(np.arange(11), np.log(mean) - np.maximum(np.log(std), 0) ** 2/(2*mean**2), yerr=std/mean,
                     fmt='o', color='black',
                     capsize=5)
        plt.title(title)
        plt.xlabel('Strahler Order')
        plt.ylabel(ylabel)
        plt.savefig("log_freq_real.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        raw_data = {'x': np.arange(11), 'y': mean}
        df = pd.DataFrame(raw_data)
        p = np.polyfit(raw_data['x'], np.log(raw_data['y']), 1)
        y = np.exp(np.polyval(p, raw_data['x']))
        plt.plot(raw_data['x'], y)
        plt.errorbar(np.arange(11), mean, yerr=std,
                     fmt='o', color='black',
                     capsize=5)

        plt.yscale('log')
        plt.autoscale()
        plt.savefig("log_freq_real_log.pdf", format="pdf", bbox_inches="tight")
        plt.show()


    def all_branch_length(self):
        """
        Computes average branch length for each Strahler order.
        """

        mean_l = []
        for i in range(0, self.max_HS + 1):
            idices = np.nonzero(self.HS == i)[0]
            HS_graph = nx.Graph()
            ls = []
            for idx in idices:
                HS_graph.add_edge(self.connections[idx][0], self.connections[idx][1], length=self.lengths[idx])
            for c in nx.connected_components(HS_graph):
                sg = HS_graph.subgraph(c)
                els = [HS_graph[e[0]][e[1]]['length'] for e in sg.edges]
                ls.append(np.sum(els))
            mean_l.append(ls)
        return mean_l


    def sub_tree_volume(self, mode='level'):

        for n in self.tree.nodes:
            if self.tree.degree[n] == 1 and not self.tree.nodes[n]['root']:
                root = list(self.tree.neighbors(n))[0]
                self.tree.nodes[n][mode] = 1
                self.tree[root][n]['sub_volume'] = 0

            else:
                self.tree.nodes[n][mode] = 0

            if self.tree.nodes[n]['root']:
                self.tree.nodes[n][mode] = -1

        count_no_label = len(self.tree.nodes)

        while count_no_label != 0:
            for node in self.tree.nodes:
                if self.tree.nodes[node][mode] != 0 or len(list(self.tree.neighbors(node))) == 0:
                    continue

                neighbors = list(self.tree.neighbors(node))

                neighbor_orders = np.array([self.tree.nodes[n][mode] for n in neighbors])

                if -1 in neighbor_orders and np.count_nonzero(neighbor_orders == 0) >= 1:
                    continue
                if -1 not in neighbor_orders and np.count_nonzero(neighbor_orders == 0) > 1:
                    continue

                max_order = np.max(neighbor_orders)
                self.tree.nodes[node][mode] = max_order + 1

                if -1 in neighbor_orders:
                    root_idx = np.where(neighbor_orders == -1)[0][0]
                else:
                    root_idx = np.where(neighbor_orders == 0)[0][0]

                root_idx = neighbors[root_idx]

                r_child = np.array([self.tree[node][n]['sub_volume'] for n in neighbors
                                    if n != root_idx])

                vol_child = np.array([np.pi * self.tree[node][n]['radius']**2 *
                                    np.linalg.norm(self.tree.nodes[node]['loc'] -
                                                   self.tree.nodes[n]['loc']) * self.vsize for n in neighbors
                                    if n != root_idx])

                self.tree[root_idx][node]['sub_volume'] = np.sum(r_child + vol_child)

            count_no_label = np.count_nonzero(np.array([self.tree.nodes[n][mode] for n in self.tree.nodes]) == 0)

        node = [n for n in self.tree.nodes if self.tree.nodes[n][mode] == -1][0]
        neighbors = list(self.tree.neighbors(node))
        neighbor_orders = np.array([self.tree.nodes[n][mode] for n in neighbors])
        max_order = np.max(neighbor_orders)
        max_count = np.count_nonzero(neighbor_orders == max_order)
        self.tree.nodes[node][mode] = max_order if max_count == 1 and mode == 'HS' else max_order + 1

        for n in neighbors:
            r_child = self.tree[node][n]['sub_volume']
            vol_child = np.pi * self.tree[node][n]['radius'] ** 2 * np.linalg.norm(self.tree.nodes[node]['loc'] -
                       self.tree.nodes[n]['loc']) * self.vsize
            self.tree[node][n]['sub_volume'] = r_child + vol_child


if __name__ == '__main__':


    root_loc = [588, 217, 650]


    pt_file = 'result.vtk'
    

    save_file = pt_file[:-4] + '_w_morph.vtk'

    vspace = 22.6

    vt = VtkMorphAnalyer(pt_file, root_loc, vsize=vspace)
    vt.build()
    vt.build_morphologies()

    vt.plot_everything()
    # plt.savefig("sub_tree.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    vt.save(save_file)

