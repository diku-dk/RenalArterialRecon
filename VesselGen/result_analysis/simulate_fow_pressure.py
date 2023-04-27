import os.path

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import networkx as nx
import pyvista
from VesselGen.utils.graph_modelling import *
from collections import OrderedDict
from VesselGen.preprocessing.read_to_networkx import VtkNetwork
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import scipy

import matplotlib

font = {'family' : "Times New Roman", 'size': 14}


matplotlib.rc('font', **font)


class VtkNetworkAnalysis(VtkNetwork):

    def __init__(self, pt_file=None, root_loc=None, vsize=22.6, mu=0, t_flow=0):
        super().__init__(pt_file, root_loc, vsize)

        self.mu = mu

    def build(self):

        mesh = pyvista.read(self.pt_file)

        points = mesh.points
        line = mesh.lines

        self.tree = nx.Graph()

        point_data = mesh.point_data
        cell_data = mesh.cell_data

        point_feature_all = {key: value for (key, value) in point_data.items()}

        for i, p in enumerate(points):
            
            
            
            point_feature = {key: value[i] for (key, value) in point_feature_all.items()}

            self.tree.add_node(i, loc=np.array(p),
                               root=False, fixed=False,
                               **point_feature
                               )

        i = 1
        while i < len(line):
            node1, node2 = line[i], line[i + 1]

            edge_feature = {}
            for key in cell_data.keys():
                edge_feature[key] = cell_data[key][i // 3]
            

            self.tree.add_edge(node1, node2,
                               **edge_feature,
                               )

            i += 3

        for edge in self.tree.edges:
            a, b = edge
            self.tree[a][b]['radius_neg'] = - self.tree[a][b]['radius']

        all_nodes = [i for i in self.tree.nodes]

        if 'level' in self.tree[all_nodes[0]]:
            all_levels = np.array([self.tree.nodes[i]['level'] for i in all_nodes])
            if -1 in all_levels:
                root_idx = np.where(all_levels == -1)[0][0]
            else:
                rooot_idx = np.argmax(all_levels)

            self.root = all_nodes[root_idx]

        elif self.root_loc is not None:
            self.root = self.find_n_with_coord(self.root_loc)

        self.tree.nodes[self.root]['root'] = True
        self.tree.nodes[self.root]['fixed'] = True

        Gcc = sorted(nx.connected_components(self.tree), key=len, reverse=True)
        self.tree = nx.Graph(self.tree.subgraph(Gcc[0]))

        self.remove_intermediate()

        self.update_order()


    def add_vessel(self, node, neighbor_node, radius=None, flow=None, sub=-1):
        """
        Adds a vessel between two nodes.

        Parameters
        --------------------
        node -- one endpoint
        neighbor_node -- the other endpoint
        """

        r = radius
        f = flow
        dis = np.linalg.norm(self.tree.nodes[node]['loc'] - self.tree.nodes[neighbor_node]['loc']
                             ) * self.vsize
        self.tree.add_edge(node, neighbor_node, radius=r, flow=f, length=dis, sub=sub)


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

                edge_feature = {}
                for key in self.tree[left][n].keys():
                    edge_feature[key] = self.tree[left][n][key]

                self.tree.add_edge(left, right, **edge_feature)

                self.tree.remove_node(n)

    def label_pressure_drop(self):
        for n in self.tree.edges:
            a, b = n[0], n[1]
            r = self.tree[a][b]['radius']

            l = np.linalg.norm(self.tree.nodes[a]['loc'] - self.tree.nodes[b]['loc']) * self.vsize
            self.tree[a][b]['length'] = l


            mu = self.mu

            self.tree[a][b]['resistence'] = 8 * mu * l / (np.pi * r ** 4)
            self.tree[a][b]['pressure_drop'] = self.tree[a][b]['resistence'] * self.tree[a][b]['flow']
            self.tree[a][b]['viscosity'] = mu

    def label_pressure_from_root(self, node=None, root_pressure=0.0):

        if node is None:
            node = [n for n in self.tree.nodes if self.tree.nodes[n]['root']][0]
            self.tree.nodes[node]['pressure'] = root_pressure
            self.tree.nodes[node]['pressure_mmhg'] = self.tree.nodes[node]['pressure'] * 1e12 / 133.322

            children = np.array(list(self.tree.neighbors(node)))

        else:
            neighbors = np.array(list(self.tree.neighbors(node)))
            neighbor_orders = np.array([self.tree.nodes[n]['level'] for n in neighbors])
            root = -1 if -1 in neighbor_orders else np.max(neighbor_orders)
            children = np.array([neighbors[i] for i in range(len(neighbors)) if neighbor_orders[i] != root])
            root = neighbors[np.where(neighbor_orders == root)[0][0]]
            self.tree.nodes[node]['pressure'] = \
                self.tree.nodes[root]['pressure'] - self.tree[root][node]['pressure_drop']

            self.tree.nodes[node]['pressure_mmhg'] = self.tree.nodes[node]['pressure'] * 1e12 / 133.322

        for i in children:
            self.label_pressure_from_root(i)


if __name__ == '__main__':


    root_loc = [588, 217, 650]

    pt_file = 'result.vtk'

    save_file = pt_file[:-4] + '_w_pressure.vtk'

    t_flow, mu = 1.167e11/3e4, 3.6e-15
    vspace = 22.6

    vt = VtkNetworkAnalysis(pt_file, root_loc, vsize=vspace, mu=mu, t_flow=t_flow)
    vt.build()
    vt.label_pressure_drop()

    vt.label_pressure_from_root(root_pressure=100 * 133.322e-12)

    all_leaf_pressure = [vt.tree.nodes[i]['pressure'] for i in vt.tree.nodes if vt.tree.nodes[i]['level'] == 1]
    plt.hist(all_leaf_pressure, 20)
    plt.xlabel('Terminal pressure (1e12Pa)')
    plt.ylabel('Frequency')
    plt.title('Terminal pressure frequency')
    plt.savefig("pressure_hist_pa.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    all_leaf_pressure = [vt.tree.nodes[i]['pressure_mmhg'] for i in vt.tree.nodes if vt.tree.nodes[i]['level'] == 1]
    plt.hist(all_leaf_pressure, 20)
    plt.xlabel('Terminal pressure (in mmHg)')
    plt.ylabel('Frequency')

    plt.title('Terminal pressure frequency')
    plt.legend()
    plt.savefig("pressure_hist_mmhg.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    vt.save(save_file)