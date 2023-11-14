import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
from lib.model.abstract_state_model import AbstractStateModel
from lib.model import loss
import numpy as np
import torch.nn.init as init
from geopy.distance import geodesic


def calculate_scaled_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    L' = 2L/lambda - I

    Args:
        adj: adj_matrix

    Returns:
        np.ndarray: L'
    """
    n = adj.shape[0]
    d = np.sum(adj, axis=1)  # D
    lap = np.diag(d) - adj     # L=D-A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                lap[i, j] /= np.sqrt(d[i] * d[j])
    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0
    lam = np.linalg.eigvals(lap).max().real
    # L_ = 2 * lap / lam - np.eye(n)
    return lap


def calculate_cheb_poly(lap, ks):
    """
    k-order Chebyshev polynomials : T0(L)~Tk(L)
    T0(L)=I/1 T1(L)=L Tk(L)=2LTk-1(L)-Tk-2(L)

    Args:
        lap: scaled laplacian matrix
        ks: k-order

    Returns:
        np.ndarray: T0(L)~Tk(L)
    """
    n = lap.shape[0]
    lap_list = [np.eye(n), lap[:]]
    for i in range(2, ks):
        lap_list.append(np.matmul(2 * lap, lap_list[-1]) - lap_list[-2])
    if ks == 0:
        raise ValueError('Ks must bigger than 0!')
    if ks == 1:
        return np.asarray(lap_list[0:1])  # 1*n*n
    else:
        return np.asarray(lap_list)       # Ks*n*n


def calculate_first_approx(weight):
    '''
    1st-order approximation function.
    :param W: weighted adjacency matrix of G. Not laplacian matrix.
    :return: np.ndarray
    '''
    # TODO: 如果W对角线本来就是全1？
    n = weight.shape[0]
    adj = weight + np.identity(n)
    d = np.sum(adj, axis=1)
    # sinvd = np.sqrt(np.mat(np.diag(d)).I)
    # return np.array(sinvd * A * sinvd)
    sinvd = np.sqrt(np.linalg.inv(np.diag(d)))
    lap = np.matmul(np.matmul(sinvd, adj), sinvd)  # n*n
    lap = np.expand_dims(lap, axis=0)              # 1*n*n
    return lap


def calculate_adjacency_matrix(adj_mx):
    """
        使用带有阈值的高斯核计算邻接矩阵的权重，如果有其他的计算方法，可以覆盖这个函数,
        公式为：$ w_{ij} = \exp \left(- \\frac{d_{ij}^{2}}{\sigma^{2}} \\right) $, $\sigma$ 是方差,
        小于阈值`weight_adj_epsilon`的值设为0：$  w_{ij}[w_{ij}<\epsilon]=0 $

        Returns:
            np.ndarray: adj_mx, N*N的邻接矩阵
    """
    weight_adj_epsilon = 0.1
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))
    adj_mx[adj_mx < weight_adj_epsilon] = 0
    # print(self.adj_mx)
    # np.save('adj_mx.npy', self.adj_mx)
    # print(done)
    return calculate_scaled_laplacian(adj_mx.numpy())


class GcnOperation(nn.Module):
    def __init__(self, num_of_features, num_of_filter, activation="GLU"):
        super(GcnOperation,self).__init__()
        assert activation in {'GLU', 'relu'}
        self.num_of_filter = num_of_filter
        self.num_of_features = num_of_features
        self.activation = activation
        if activation == "GLU":
            self.layer = nn.Linear(num_of_features, 2 * num_of_filter)
        elif activation == "relu":
            self.layer = nn.Linear(num_of_features, num_of_filter)

    def forward(self, data, adj):
        data = torch.matmul(adj, data)

        if self.activation == "GLU":
            data = self.layer(data)
            lhs, rhs = data.split(self.num_of_filter, -1)
            data = lhs * torch.sigmoid(rhs)

        elif self.activation == "relu":
            data = torch.relu(self.layer(data))

        return data

class GraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat,  use_bn=True, mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj, mask=None):
        if self.add_self:
            adj = adj + torch.eye(adj.size(0)).to(x.device)

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(h_k.size(1)).to(x.device)
            h_k = self.bn(h_k)
        # if mask is not None:
        #     h_k = h_k * mask.unsqueeze(2).expand_as(h_k)
        return h_k


class DiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, is_final=False):
        super(DiffPool, self).__init__()
        self.is_final = is_final
        # self.embed = GraphSAGE(nfeat, nhid, use_bn=True)
        # self.assign_mat = GraphSAGE(nfeat, nnext, use_bn=True)
        # self.calculate = GraphSAGE(nhid, nhid, use_bn=True)
        self.embed = GcnOperation(nfeat, nhid)
        self.assign_mat = GcnOperation(nfeat, nnext)
        self.calculate = GcnOperation(nhid, nhid)
        self.entropy_loss = 0


    def forward(self, x, adj):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        x_next = self.calculate(xnext, torch.mean(anext,dim=0))
        x_out = torch.matmul(s_l, x_next)
        self.entropy_loss = torch.distributions.Categorical(probs=torch.mean(s_l,dim=0)).entropy()
        self.entropy_loss = self.entropy_loss.sum()
        return x_out, self.entropy_loss


class AGCN(nn.Module):
    # Cluster Embedding GCN
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.cluster_embedding = 10
        self.cluster_weights_pool = nn.Parameter(torch.FloatTensor(420, cheb_k, dim_in, dim_out, self.cluster_embedding))
        self.bias_pool = nn.Parameter(torch.FloatTensor(420, dim_out, self.cluster_embedding))
        self.maxpool = nn.MaxPool3d(kernel_size=[1,1,1],dilation=[1,1,1],stride=[1,1,10],padding=[0,0,0])

    def forward(self, x, node_embeddings, laplacian_mx):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        # 直接当作是拉普拉斯矩阵 Todo
        # supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # supports = F.softmax(F.relu(torch.mm(node_embeddings_1, node_embeddings_2)), dim=1)
        # supports = torch.from_numpy(laplacian_mx).to(node_embeddings.device)
        supports = torch.from_numpy(laplacian_mx[0]).to(node_embeddings.device)

        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # cluster_weights =  torch.einsum('nd,dkioc->nkioc', node_embeddings, self.weights_pool)
        # cluster_weights_1 = self.maxpool(cluster_weights)[:,:,:,:,0]
        # cluster_weights = torch.max(cluster_weights, dim=-1)[0]
        # weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(node_embeddings, self.bias_pool)                       # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkioc->bnoc', x_g, self.cluster_weights_pool)  + self.bias_pool    # b, N, dim_out, cluster_dim
        x_gconv = torch.max(x_gconv, dim=-1)[0]
        # Maxpooling
        # x_gconv = self.maxpool(x_gconv)
        return x_gconv

class Embedded_GCN(nn.Module):
    # Todo
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(Embedded_GCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings, laplacian_mx):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        # 直接当作是拉普拉斯矩阵 Todo
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # supports = F.softmax(F.relu(torch.mm(node_embeddings_1, node_embeddings_2)), dim=1)
        # supports = torch.from_numpy(laplacian_mx).to(node_embeddings.device)
        # supports = torch.from_numpy(laplacian_mx[0]).to(node_embeddings.device)

        support_set = [torch.eye(node_num).to(node_embeddings.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # b, N, dim_out
        # Maxpooling
        # x_gconv = self.maxpool(x_gconv)
        return x_gconv


class HCGCN(nn.Module):
    # Todo
    # Hierarchical Cluster Graph Convolution
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(HCGCN, self).__init__()
        self.cheb_k = cheb_k
        self.pool_ratio = 0.1
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.hc = DiffPool(dim_out, int(420*self.pool_ratio), dim_out)

    def forward(self, x, node_embeddings, laplacian_mx):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        # 直接当作是拉普拉斯矩阵 Todo
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # supports = F.softmax(F.relu(torch.mm(node_embeddings_1, node_embeddings_2)), dim=1)
        # supports = torch.from_numpy(laplacian_mx[0]).to(node_embeddings.device)
        # laplacian_mx_2 = torch.from_numpy(laplacian_mx[0]).to(node_embeddings.device)
        laplacian_mx_2 = supports
        # supports = laplacian_mx[0]

        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # Todo
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # b, N, dim_out
        # Maxpooling
        # x_gconv = self.maxpool(x_gconv)
        x_next, loss = self.hc(x_gconv, laplacian_mx_2)
        return x_gconv, x_next, loss


class MutiClusterGCNCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, layer):
        super(MutiClusterGCNCell, self).__init__()
        self.layer = layer
        self.perceptron = nn.ModuleList()
        self.graphconv = nn.ModuleList()
        self.attlinear = nn.Linear(281* dim_out, 1)
        self.graphconv.append(HCGCN(dim_in, dim_out, cheb_k, embed_dim))
        for i in range(1, layer):
            self.graphconv.append(AGCN(dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, inputs, node_embeddings, supports):
        outputs = []
        for i in range(self.layer):
            inputs, inputs_2, loss = self.graphconv[i](inputs,node_embeddings,[supports[i]])
            outputs.append(inputs)
            outputs.append(inputs_2)
        out = self.attention(torch.stack(outputs, dim=1))
        # out = outputs[-1]
        return out, loss

    def attention(self, inputs):
        b, g, n, f = inputs.size()
        x = inputs.reshape(b, g, -1)
        out = self.attlinear(x)  # (batch, graph, 1)
        weight = F.softmax(out, dim=1)

        outputs = (x * weight).sum(dim=1).reshape(b, n, f)
        return outputs


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, layer):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = MutiClusterGCNCell(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim, layer)
        self.update = MutiClusterGCNCell(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim, layer)

    def forward(self, x, state, node_embeddings,laplacian_mx):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        # 这边是GRU的实现
        state = state.to(x.device)
        # x_location = x[:,:,1:]
        # x_input = x[:,:,:1]
        input_and_state = torch.cat((x, state), dim=-1)
        tmp, loss_0 = self.gate(input_and_state, node_embeddings, laplacian_mx)
        z_r = torch.sigmoid(tmp)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        cc, loss_1 = self.update(candidate, node_embeddings, laplacian_mx)
        hc = torch.tanh(cc)
        h = r*state + (1-r)*hc
        return h, loss_0 + loss_1

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class RGRUEncoder(nn.Module):
    def __init__(self, config):
        super(RGRUEncoder, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_dim = config['feature_dim']
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10) # embed_dim 看成cluster number 进行聚类
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        self.layers = config.get('graph_layer', 2)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.feature_dim-2,
                                          self.hidden_dim, self.cheb_k, self.embed_dim, self.layers))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim, self.layers))

    def forward(self, x, init_state, node_embeddings, laplacian_mx):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.feature_dim
        seq_length = x.shape[1]
        current_inputs = x[:,:,:,:1]
        output_hidden = []
        current_location = x[:,:,:,1:]
        # current_laplacian_mx = self.calculate_current_scaled_laplacian(current_location)
        a_loss = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state, loss = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, laplacian_mx)
                inner_states.append(state)
                a_loss.append(loss)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        a_loss = torch.Tensor(a_loss).mean()
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden, a_loss


    def calculate_current_scaled_laplacian(self, x_location):
        x_sum_location = torch.mean(x_location, dim=1)
        # b, n = x_sum_location.shape[0], x_sum_location.shape[1]
        # batch_zero = torch.zeros(b, n, n)
        # for i in range(b):
        #     batch_location = x_sum_location[i,:,:]
        #     for k in range(n):
        #         for v in range(n):
        #             batch_zero[i][k][v] = geodesic((batch_location[k][0],batch_location[k][1]), (batch_location[v][0],batch_location[v][1])).km
        #     batch_zero[i] = torch.from_numpy(calculate_adjacency_matrix(batch_zero[i]))
        x_sum_location = torch.mean(x_sum_location, dim=0)
        n = x_sum_location.shape[0]
        batch_zero = torch.zeros(n, n)
        for k in range(n):
            for v in range(n):
                    batch_zero[k][v] = geodesic((x_sum_location[k][0],x_sum_location[k][1]), (x_sum_location[v][0],x_sum_location[v][1])).km
        batch_zero = torch.from_numpy(calculate_adjacency_matrix(batch_zero))
        return batch_zero


    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      # (num_layers, B, N, hidden_dim)


class RGRUDecoder(nn.Module):
    def __init__(self, config):
        super(RGRUDecoder, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_dim = config['feature_dim']
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        self.output_window = config.get('output_window', 1)
        self.output_dim = config.get('output_dim', 1)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.output_dim,
                                          self.hidden_dim, self.cheb_k, self.embed_dim))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim))
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, hidden_states, node_embeddings, seq_length):

        b, n, _ = hidden_states[0].shape

        current_inputs = torch.zeros(b, n, self.output_dim, device=hidden_states[0].device, dtype=hidden_states[0].dtype)
        # output_hidden = []

        new_outputs = list()
        for t in range(seq_length):
            for i in range(self.num_layers):
                current_inputs = self.dcrnn_cells[i](current_inputs, hidden_states[i], node_embeddings)
            current_inputs = self.out(current_inputs)
            new_outputs.append(current_inputs)
        output = torch.stack(new_outputs, 1)

        # for i in range(self.num_layers):
        #     state = init_state[i]
        #     inner_states = []
        #     for t in range(seq_length):
        #         state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
        #         inner_states.append(state)
        #     output_hidden.append(state)
        #     current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return output


class HiGRU(AbstractStateModel):
    def __init__(self, config, data_feature):
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim

        super().__init__(config, data_feature)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)


        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.encoder = RGRUEncoder(config)
        # self.decoder = RGRUDecoder(config)
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        # self.decoder_out = nn.Linear(self.output_dim, self.output_window, bias=True)
        self.attlinear = nn.Linear(self.output_dim * self.num_nodes, 1)

        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._init_parameters()
        self.thea = 0.001

        self.graph_conv_type = config.get('graph_conv_type', 'chebconv')
        adj_mx = data_feature['adj_mx']  # ndarray
        # 计算GCN邻接矩阵的归一化拉普拉斯矩阵和对应的切比雪夫多项式或一阶近似
        if self.graph_conv_type.lower() == 'chebconv':
            self.laplacian_mx = calculate_scaled_laplacian(adj_mx)
            # self.Lk = calculate_cheb_poly(laplacian_mx, self.Ks)
            # self._logger.info('Chebyshev_polynomial_Lk shape: ' + str(self.Lk.shape))
            # self.Lk = torch.FloatTensor(self.Lk).to(self.device)
        elif self.graph_conv_type.lower() == 'gcnconv':
            self.Lk = calculate_first_approx(adj_mx)
            # self._logger.info('First_approximation_Lk shape: ' + str(self.Lk.shape))
            # self.Lk = torch.FloatTensor(self.Lk).to(self.device)
            # self.Ks = 1  # 一阶近似保留到K0和K1，但是不是数组形式，只有一个n*n矩阵，所以是1（本质上是2）
        else:
            raise ValueError('Error graph_conv_type, must be chebconv or gcnconv.')

        m, p, n = torch.svd(torch.from_numpy(self.laplacian_mx))
        initemb1 = torch.mm(m[:, :self.embed_dim], torch.diag(p[:self.embed_dim] ** 0.5))
        initemb2 = torch.mm(torch.diag(p[:self.embed_dim] ** 0.5), n[:, :self.embed_dim].t())
        self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
        self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
        self.w1 = nn.Parameter(torch.eye(self.embed_dim), requires_grad=True)
        self.w2 = nn.Parameter(torch.eye(self.embed_dim), requires_grad=True)
        self.b1= nn.Parameter(torch.zeros(self.embed_dim), requires_grad=True)
        self.b2=nn.Parameter(torch.zeros(self.embed_dim), requires_grad=True)
        self.graph0 = self.laplacian_mx
        self.graph1 = None
        self.graph2 = None

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        source = batch['X']
        seq_length = source.shape[1]

        # 生成聚类的邻接矩阵
        graph = list()
        graph.append(self.graph0)
        graph.append(self.graph0)
        #
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _, a_loss = self.encoder(source, init_state, self.node_embeddings, graph)  # B, T, N, hidden
        output_1 = output[:, -1:, :, :]                                       # B, 1, N, hidden

        # Decoder based predictor
        # init_state_2 = self.decoder.init_hidden(source.shape[0])qia
        # output_2 = self.decoder(hidden_states, self.node_embeddings, seq_length)
        # output_2 = self.decoder_out(output_2)

        # CNN based predictor
        output_cnn = self.end_conv(output_1)                           # B, T*C, N, 1
        output_cnn = output_cnn.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output_cnn = output_cnn.permute(0, 1, 3, 2)                      # B, T, N, C


        return output_cnn, a_loss

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted, a_loss = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true, 0) + self.thea * a_loss

    def predict(self, batch):
        return self.forward(batch) 

    # def attention(self, o):
    #     b, g, n, f = inputs.size()
    #     x = inputs.reshape(b, g, -1)
    #     out = self.attlinear(x)  # (batch, graph, 1)
    #     weight = F.softmax(out, dim=1)
    #     outputs = (x * weight).sum(dim=1).reshape(b, n, f)
    #     return outputs
