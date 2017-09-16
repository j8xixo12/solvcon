#!/usr/bin/env python3

from modulegraph.modulegraph import ModuleGraph
mg = ModuleGraph('./')
mg.run_script('./solvcon/parcel/gas/physics.py')


remove_nodes_prefix = ['ctypes', 'numpy', '__future__']

remove_nodes = []
for node in mg.nodes():
    if node.identifier.split('.')[0] in remove_nodes_prefix:
        remove_nodes.append(node.identifier)

for node in remove_nodes:
    mg.removeNode(node)

f = open('physics.dot','w')
mg.graphreport(f)
f.close()
