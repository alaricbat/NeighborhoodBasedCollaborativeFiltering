import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

class CF(object):

    def __init__(self, Y_data, k, dist_fun = cosine_similarity, uuCF = 1):
        """
        初始化推荐系统对象。

        Args:
            Y_data (np.ndarray): 用户-物品交互数据矩阵。
                默认结构: [用户ID, 物品ID, 评分]
                若 uuCF = 0: 结构将被转换为 [物品ID, 用户ID, 评分]
            k (int): 模型中考虑的最近邻数量
            dist_fun (function): 用于计算数据点之间距离/相似度的函数
            uuCF (int, optional): 
                - 1 (默认): 使用用户-用户协同过滤 (User-User Collaborative Filtering)
                - 0: 使用物品-物品协同过滤 (Item-Item Collaborative Filtering)
        
        Attributes:
            uuCF (int): 存储协同过滤模式状态
            Y_data (np.ndarray): 原始数据(已根据uuCF参数调整)
            k (int): 最近邻数量
            dist_fun (function): 距离计算函数
            Ybar_data (None): 初始化用于后续存储标准化后的数据
            n_users (int): 从数据中计算得到的用户总数
            n_items (int): 从数据中计算得到的物品总数
        """
        self.uuCF = uuCF
        self.Y_data = Y_data if uuCF else Y_data[:, [1 ,0 ,2]]
        self.k = k
        self.dist_fun = dist_fun
        self.Ybar_data = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    
    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis=0)

    def __normalize_Y(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            item_ids = self.Y_data[ids, 1]
            ratings = self.Y_data[ids, 2]
            mean = np.mean(ratings)
            if np.isnan(mean):
                mean = 0 # to avoid empty array and nan value
            self.Ybar_data[ids, 2] = ratings - mean
        
        """
        根据规范化的用户-物品交互数据构建COO(坐标格式)稀疏矩阵。

        此函数创建的稀疏矩阵中：
        - 行代表物品
        - 列代表用户
        - 值代表规范化的交互分数（如规范化评分）

        参数：
        ----------
        从对象属性派生的隐式参数：
        - self.Ybar_data[:, 2]: 规范化值数组
        - self.Ybar_data[:, 1]: 物品索引数组（行坐标）
        - self.Y_data[:, 0]: 用户索引数组（列坐标）
        - self.n_items: 物品总数（矩阵行数）
        - self.n_users: 用户总数（矩阵列数）

        返回值：
        --------
        形状为 (n_items, n_users) 的 scipy.sparse.coo_matrix

        示例：
        --------
        给定输入数据：
        Y_data = [  [0, 0, 5],   # 用户0对物品0评分5
                    [0, 1, 3],   # 用户0对物品1评分3
                    [1, 0, 4],   # 用户1对物品0评分4
                    [2, 2, 2]]   # 用户2对物品2评分2

        Ybar_data = [   [0, 0, 0.8],  # 规范化值: 0.8
                        [0, 1, 0.5],  # 规范化值: 0.5
                        [1, 0, 0.7],  # 规范化值: 0.7
                        [2, 2, 0.3]]  # 规范化值: 0.3

        生成的COO稀疏矩阵将包含:
        - 位置 (0, 0): 值 0.8 (物品0, 用户0)
        - 位置 (1, 0): 值 0.5 (物品1, 用户0)
        - 位置 (0, 1): 值 0.7 (物品0, 用户1)
        - 位置 (2, 2): 值 0.3 (物品2, 用户2)

        对应的密集矩阵为：
        [   [0.8 0.7 0. ]
            [0.5 0.  0. ]
            [0.  0.  0.3]]

        注意：
        -----
        这种稀疏表示只高效存储非零元素，
        对于大多数用户-物品对没有交互的大型用户-物品交互数据集
        具有内存效率高的优点。
        """
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2], 
                                       (self.Ybar_data[:, 1], self.Y_data[:, 0])), 
                                       (self.n_items, self.n_users))
        
    def __similarity(self):
        self.S = self.dist_fun(self.Ybar.T, self.Ybar.T)

    def __refresh(self):
        self.__normalize_Y()
        self.__similarity()

    def fit(self):
        self.__refresh()

    def __pred(self, u, i, normalized = 1):
        
        #第一步：物品 i 的项目
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)

        #第二步：找到所有对物品 i 评分的用户
        users_rated_i = np.where(self.Y_data[ids, 0]).astype(np.int32)

        #第三步：计算当前用户与其他已对物品 i 评分的用户之间的相似度
        similarity = self.S[u, users_rated_i]

        #第四步：找出 k 个最相似的用户
        a = np.argsort(similarity)[-self.k:]

        #第五步：以及相应的相似度水平 (And the corresponding similarity level)
        nearest_s = similarity[a]

        #每个最相似的用户如何对物品 i 评分 - How did each of 'near' users rated them i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            #Add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)
        
        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalize = 1):
        if self.uuCF: return self.__pred(u, i, normalize)
        return self.__pred(i, u, normalize)
    
    def recommendation(self, u, normalized = 1):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i, normalized)
                if rating > 0:
                    recommended_items.append(i)
        return recommended_items
    
    def print_recommendation(self):
        for u in range(self.n_users):
            recommended_items = self.recommendation(u)
            if self.uuCF:
                print('Recommend item(s): ', recommended_items, 'to user', u)
            else: 
                print('Recommend item', u, 'to user(s) : ', recommended_items)