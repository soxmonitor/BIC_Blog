from lava.magma.core.process.process import AbstractProcess as Proc
from lava.proc.lif.process import LIF


class CustomNeuron(Proc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 在这里定义神经元属性和行为


class self_define_neuron:
    def __init__(self):
        self.id = 0  # 神经元地址/编号
        self.I_O_type = 0  # 初始化为0， 0代表O型， 1代表I型
        self.NUM_Neuron = 0
        self.Pop = 0
        self.W = []
        self.C = []
        self.P = []
        self.constrains = []


class OG_GNC:
    def __init__(self):
        self.Epop = []  # 数据结构暂定为List，内部放什么我不清楚, 这是种群记录项
        self.NUM_Neuron = 0  # 神经元总数
        self.Pop = 0
        self.W = []
        self.C = []
        self.P = []
        self.constrains = []


class GNC:
    def __init__(self, N, Pop, W, C, P):
        self.N = N  # 神经元数量
        self.Pop = Pop  # Population个数
        self.W = W  # 权重矩阵
        self.C = C  # 连接矩阵
        self.P = []  # （神经元状态|参数）作为tuples


def convert_oggnc_to_gnc(og_gnc):
    # 从 OG_GNC 实例中读取相应的属性
    N = og_gnc.NUM_Neuron
    Pop = og_gnc.Pop
    W = og_gnc.W
    C = og_gnc.C
    P = og_gnc.P

    # 创建并返回一个新的 GNC 实例
    return GNC(N, Pop, W, C, P)


class GNCList:  # <----------传入数据结构
    def __init__(self):
        self.nodes = []  # 初始化一个空列表来存储节点
        self.relationships = {}  # 初始化一个字典来存储节点间的关系

    def add_node(self, node):
        self.nodes.append(node)  # 添加一个GNC节点到列表
        self.relationships[node] = []  # 为新节点初始化一个空的关系列表

    def add_relationship(self, node1, node2):
        if node1 in self.nodes and node2 in self.nodes:
            self.relationships[node1].append(node2)
            self.relationships[node2].append(node1)  # 双向连接

    def get_node(self, index):
        if index < len(self.nodes):
            return self.nodes[index]  # 返回指定索引处的节点
        return None  # 如果索引超出范围则返回None

    def get_relationships(self, node):
        if node in self.nodes:
            return self.relationships[node]  # 返回节点的所有关联节点
        return None  # 如果节点不在列表中，则返回None


class GNCGraph:  #
    def __init__(self, size):
        self.size = size
        self.matrix = [[0] * size for _ in range(size)]  # 创建一个size x size的矩阵，初始化为0
        self.nodes = [None] * size  # 存储 GNC 实例的列表

    def add_node(self, node, index):
        if 0 <= index < self.size:
            self.nodes[index] = node

    def add_edge(self, from_index, to_index):
        if 0 <= from_index < self.size and 0 <= to_index < self.size:
            self.matrix[from_index][to_index] = 1  # 设置矩阵相应位置为1，表示存在边

    def get_node(self, index):
        if 0 <= index < self.size:
            return self.nodes[index]


def convert_GNClist_to_GNCgraph(gnc_list):
    # 创建一个GNCGraph实例，大小与GNCList中的节点数量相同
    graph = GNCGraph(size=len(gnc_list.nodes))

    # 将GNCList中的所有节点添加到GNCGraph中
    for index, node in enumerate(gnc_list.nodes):
        graph.add_node(node, index)

    # 根据GNCList中的关系来设置GNCGraph的边
    for node, related_nodes in gnc_list.relationships.items():
        from_index = gnc_list.nodes.index(node)  # 获取当前节点的索引
        for related_node in related_nodes:
            to_index = gnc_list.nodes.index(related_node)  # 获取关联节点的索引
            graph.add_edge(from_index, to_index)  # 添加边到图中

    return graph  # 返回构建好的GNCGraph实例

#######################################################################################################################

#                             convert_GNClist_to_GNCgraph  Example Usage

#######################################################################################################################


# # 创建几个GNC节点
# gnc1 = GNC(N=10, Pop=100, W=[[0.1]*10]*10, C=[[1]*10]*10, P=[(0,0.1)]*10)
# gnc2 = GNC(N=15, Pop=150, W=[[0.2]*15]*15, C=[[1]*15]*15, P=[(1,0.2)]*15)
# gnc3 = GNC(N=20, Pop=200, W=[[0.3]*20]*20, C=[[1]*20]*20, P=[(2,0.3)]*20)
#
# # 创建一个GNCList并添加这些节点
# gnc_list = GNCList()
# gnc_list.add_node(gnc1)
# gnc_list.add_node(gnc2)
# gnc_list.add_node(gnc3)
#
# # 添加一些节点间的关系
# gnc_list.add_relationship(gnc1, gnc2)
# gnc_list.add_relationship(gnc2, gnc3)
#
# # 使用函数转换GNCList到GNCGraph
# gnc_graph = convert_GNClist_to_GNCgraph(gnc_list)
#
# # 打印生成的GNCGraph的节点信息和邻接矩阵以验证正确性
# for index, node in enumerate(gnc_graph.nodes):
#     print(f"Node {index}:", "N =", node.N, "Pop =", node.Pop)
# print("Adjacency Matrix:")
# for row in gnc_graph.matrix:
#     print(row)



#######################################################################################################################

#                                           EOF

#######################################################################################################################




#######################################################################################################################

#                             GNCGraph class  Example Usage

#######################################################################################################################
#
# # 使用样例：
# # 创建图实例
# g = GNCGraph(3)  # 假设有三个节点
#
# # 创建 GNC 实例
# gnc1 = GNC(N=100, Pop=10, W=[[1, 0], [0, 1]], C=[[1, 0], [0, 1]], P=[])
# gnc2 = GNC(N=150, Pop=15, W=[[0, 1], [1, 0]], C=[[0, 1], [1, 0]], P=[])
# gnc3 = GNC(N=200, Pop=20, W=[[1, 1], [1, 1]], C=[[1, 1], [1, 1]], P=[])
#
# # 添加节点到图中
# g.add_node(gnc1, 0)
# g.add_node(gnc2, 1)
# g.add_node(gnc3, 2)
#
# # 添加边
# g.add_edge(0, 1)
# g.add_edge(1, 2)
# g.add_edge(2, 0)  # 创建一个环
#
# # 访问并打印一个节点
# node = g.get_node(1)
# if node:
#     print("神经元数量:", node.N)
#     print("Population个数:", node.Pop)
#######################################################################################################################

#                                           EOF

#######################################################################################################################

# 废案，已用lava调用的scipy中的sparse matrix类以及csr方法替代

# class SparseMatrixCSR:
#     def __init__(self):
#         self.row = []
#         self.col = []
#         self.value = []
#
#     def add_entries(self, cols, values, base_index):
#         self.row.append(base_index)
#         self.col.extend(cols)
#         self.value.extend(values)
#
#     def finalize(self):
#         self.row.append(len(self.value))  # 最后一个元素指向value数组的长度
