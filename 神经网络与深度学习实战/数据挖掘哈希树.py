import  pandas as pd

#读取.csv文件获取数据，转换数据格式
data_path = "data.csv"
datas = pd.read_csv(data_path).values.tolist()

#节点类
class Node(object):
    def __init__(self):
        self.elem = []  #初始设置为空
        self.next = []  # 初始设置下一节点为空

#分类方法a
a = [[1,4,7],[2,5,8],[3,6,9]]

#建立哈希树,q为叶子节点最大项数
q = 3
def hashTree(datas,root,index):
    if(index>2):
        node = Node()
        root.next.append((node))
        node.elem=datas
        return
    for ai in a:
        ds = []
        node = Node()
        root.next.append((node))
        for data in datas:
            if (data[index] in ai):
                ds.append(data)
        if(len(ds)<=q):
            node.elem=ds
        else:
            hashTree(ds,node,index+1)
#从线索0开始
index = 0
#建立根节点
root = Node()
hashTree(datas,root,index)

#深度优先输出哈希树

def outTree(root,i=0):
    print(i+1,"层",root.elem)
    for nx in root.next:
        if(nx!=None):
            outTree(nx,i+1)
#队列
class queue(object):
    def __init__(self):
        self.elem = None  #初始设置为空
        self.next = None # 初始设置下一节点为空

#宽度优先输出哈希树
def outTree2(root):
    qu = []
    front = real = -1
    que = queue()
    que.elem=root
    real += 1
    qu.append(que)
    while(front!=real):
        front+=1
        p = qu[front]
        print(p.elem.elem)
        for nx in p.elem.next:
            if(nx!=None):
                que = queue()
                que.elem = nx
                real+=1
                qu.append(que)
#outTree(root)
#outTree2(root)

t = [1,2,3,5,6]

all = []
#target每组要输出的元素个数，step游标
def combine(data, step, select_data, target_num):
    # 如果已经凑齐一组完成的组合，输出当前组
    if(len(select_data) == target_num):
        return select_data
    # 如果游标超过数组长，说明后续元素都被遍历完成，跳转到上一个元素的循环
    if(step >= len(data)):
        return
    # 把选中的元素加入临时输出列表中，n个一组作为输出
    select_data.append(data[step])
    # 递归调整最后一位
    hh = combine(data, step + 1, select_data, target_num )
    if(hh!=None):
        all.append(hh)
        print(hh)
    # 构建一个新组合，首先要删掉上次输出组中药排除的元素
    select_data.pop()
    # 递归向上调整一位元素
    combine(data, step + 1, select_data, target_num )

combine(t, 0, [], q)
print("all",all)

a = [[1,2,3]]
def aaa():
    a.append([4])
