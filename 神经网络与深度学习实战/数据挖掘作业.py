import  pandas as pd
#读取.csv文件，获取数据
data_path = "data.csv"
data = pd.read_csv(data_path)
#转换数据格式
data = data.values
#得到数据的总数和位数
count,column= data.shape
# print(count,column)
#节点类
class Node(object):
    def __init__(self):
        self.elem = []  #初始设置为空
        self.next = []  # 初始设置下一节点为空
#分类方法a
a = [[1,4,7],[2,5,8],[3,6,9]]
print(data)
# 建立所有节点
# nodes = []
# for dt in data:
#     node = Node()
#     node.elem.append(dt)
#     nodes.append(node)
# for i in range(count):
#     node = Node()
#     for j in range(column):
#         node.elem.append(data[i][j])
#     nodes.append(node)
#print(len(nodes))
# for node in nodes:
#     print(node.elem)
#建立哈希树,q为叶子节点最大项数
q = 3
def hashTree(nodes,root,index):
    if(index>2):
        for nd in nodes:
            root.next.append(nd)
        return
    for ai in a:
        nds = []
        for node in nodes:
            if (node.elem[index] in ai):
                nds.append(node)
        node = Node()
        root.next.append((node))
        if(len(nds)<=q):
            for nd in nds:
                node.elem.append(nd.elem)
        else:
            hashTree(nds,node,index+1)


#从线索0开始
index = 0
#建立根节点
root = Node()
hashTree(nodes,root,index)

#
def outTree(root):
  print(root.elem)
  for nx  in root.next:
      outTree(nx)
outTree(root)




