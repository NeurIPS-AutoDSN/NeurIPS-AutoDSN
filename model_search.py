import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class BinaryConnectNetwork(torch.nn.Module):
    def __init__(self, args):
        super(BinaryConnectNetwork, self).__init__()
        self.args = args
        field_dims = args.field_dims
        cross_num = args.cross_num
        embed_dim = args.embed_dim
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.linear = FeaturesLinear(field_dims)
        self.cross_num = cross_num
        self.network_weight_decay = args.weight_decay
        self.coeff = args.coeff

        self._arch_parameters = dict()
        self._arch_parameters['second_order'] = Variable(torch.ones(
            [self.cross_num, 2, self.num_fields], dtype=torch.float, device='cuda:0') / 2, requires_grad=True)
        self._arch_parameters['second_order'].data.add_(
            torch.randn_like(self._arch_parameters['second_order'])*1e-3)

        self.optimizer = torch.optim.Adam(self.arch_parameters(
        ), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

        final_embedding_size = self.cross_num * self.embed_dim
        self.mlp_prediction = MultiLayerPerceptron(final_embedding_size, [int(
            final_embedding_size/2)], dropout=0.5, output_layer=True).to('cuda:0')
        self.linear = Variable(torch.zeros(
            [final_embedding_size, 1], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.linear.data.add_(torch.randn_like(self.linear)*1e-2)
        self.bias = Variable(torch.zeros(
            [final_embedding_size, 1], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.criterion = torch.nn.BCELoss().to('cuda:0')

    def new(self):
        model_new = BinaryConnectNetwork(self.args).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data = y.data.clone()
        return model_new

    def arch_parameters(self):
        return [self._arch_parameters['second_order']]

    def binarize(self):
        self._cache = self._arch_parameters['second_order'].clone()
        pass

    def recover(self):
        self._arch_parameters['second_order'].data = self._cache
        del self._cache

    def genotype(self):
        genotypes = []
        genotype_ps = []
        try:
            a = self._cache
            # print(a)
        except:
            a = self._arch_parameters['second_order']
            # print(a)
        for k in range(a.shape[0]):
            genotype = [a[k, 0].argmax().cpu().numpy(),
                        a[k, 1].argmax().cpu().numpy()]
            genotypes.append(genotype)
            genotype_p = [F.softmax(
                a[k, 0], dim=-1).cpu().detach(), F.softmax(a[k, 1], dim=-1).cpu().detach()]
            genotype_ps.append(genotype_p)
        return genotypes, genotype_ps

    def forward_full(self, x):
        # forward with the full supernet
        x_embedding = self.embedding.embedding(x)
        arch_1 = self._arch_parameters['second_order'][:, 0, :].reshape(
            1, -1, self.num_fields)
        arch_2 = self._arch_parameters['second_order'][:, 1, :].reshape(
            1, -1, self.num_fields)
        fea_1 = torch.matmul(arch_1, x_embedding).reshape(-1,
                                                          self.cross_num * self.embed_dim)
        fea_2 = torch.matmul(arch_2, x_embedding).reshape(-1,
                                                          self.cross_num * self.embed_dim)
        cross = fea_1 * fea_2
        pred = torch.sum(cross, dim=1)
        output = torch.sigmoid(pred)
        return output

    def forward(self, x):
        assert self._arch_parameters['second_order'].sum(
        ) == 2. * self.cross_num
        x_embedding = self.embedding.embedding(x)
        a = torch.matmul(self._arch_parameters['second_order'].reshape(
            1, -1, self.num_fields), self.embedding.embedding(x))
        a = a.reshape(-1, self.cross_num, 2, self.embed_dim)
        b = torch.prod(a, 2)
        c = b.reshape(-1, self.cross_num * self.embed_dim)
        d = torch.sum(c, 1) / (self.cross_num)
        output = torch.sigmoid(d)
        regs = 0.0 * (torch.norm(x_embedding))
        return output  # ,regs

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        theta = _concat(self.parameters()).detach()
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            loss, self.parameters())).detach() + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment+dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.zero_grad()
        self.optimizer.zero_grad()
        self.binarize()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer, self.coeff)
        else:
            self._backward_step(input_valid, target_valid, self.coeff)
        self.recover()
        self.optimizer.step()
        # print('')

    def _backward_step(self, input_valid, target_valid, coeff):
        logits = self.forward_full(input_valid)
        loss = self.compute_loss(logits, target_valid, coeff)
        loss.backward()

    def compute_loss(self, logits, target, coeff):
        a = torch.pow(
            torch.sum(self._arch_parameters['second_order'], dim=0), 2)
        b = torch.sum(
            torch.pow(self._arch_parameters['second_order'], 2), dim=0)
        constraint = 0.5 * coeff * torch.sum(a - b)
        return self.criterion(logits, target) + constraint

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, coeff):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)
        unrolled_logits = unrolled_model.forward_full(input_valid)
        unrolled_loss = self.compute_loss(unrolled_logits, target_valid, coeff)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.detach() for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(
            vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.new()
        model_dict = self.state_dict()

        params, offset = {}, 0
        for k, v in self.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.parameters(), vector):
            p.data.add_(R, v)
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        grads_p = torch.autograd.grad(loss, self.arch_parameters())

        for p, v in zip(self.parameters(), vector):
            p.data.sub_(2*R, v)
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        grads_n = torch.autograd.grad(loss, self.arch_parameters())

        for p, v in zip(self.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


class StackNetwork(torch.nn.Module):
    def __init__(self, args):
        super(StackNetwork, self).__init__()
        self.args = args
        field_dims = args.field_dims
        cross_num = args.cross_num
        embed_dim = args.embed_dim
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.order = args.order
        self.ensemble = args.ensemble

        self.num_1, self.num_2, self.num_3 = [1, 1, 1]
        self.network_weight_decay = args.weight_decay
        self.coeff = args.coeff

        self._arch_parameters = dict()
        if self.order == '12':
            self._arch_parameters['first_order'] = Variable(torch.ones(
                [self.num_1, 1, self.num_fields], dtype=torch.float, device='cuda:0') / 2, requires_grad=False)
        else:
            self._arch_parameters['first_order'] = Variable(torch.ones(
                [self.num_1, 1, self.num_fields], dtype=torch.float, device='cuda:0') / 2, requires_grad=True)
        self._arch_parameters['first_order'].data.add_(
            torch.randn_like(self._arch_parameters['first_order'])*1e-3)

        self._arch_parameters['second_order'] = Variable(torch.ones(
            [self.num_2, 1, self.num_fields], dtype=torch.float, device='cuda:0') / 2, requires_grad=True)
        self._arch_parameters['second_order'].data.add_(
            torch.randn_like(self._arch_parameters['second_order'])*1e-3)

        self.optimizer = torch.optim.Adam(self.arch_parameters(
        ), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        if self.order == '2':
            final_embedding_size = self.num_2 * self.embed_dim
        elif self.order == '1':
            final_embedding_size = self.num_1 * self.embed_dim
        elif self.order == '12':
            if self.ensemble:
                final_embedding_size = (
                    self.num_1 + self.num_2) * self.embed_dim
            else:
                final_embedding_size = self.num_2 * self.embed_dim
        elif self.order == '3':
            final_embedding_size = self.num_3 * self.embed_dim
        else:
            raise NotImplementedError
        self.mlp_prediction = MultiLayerPerceptron(final_embedding_size, [int(
            final_embedding_size/2)], dropout=0.5, output_layer=True).to('cuda:0')
        self.linear = Variable(torch.zeros(
            [final_embedding_size, 1], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.linear.data.add_(torch.randn_like(self.linear)*1e-2)
        self.bias = Variable(torch.zeros(
            [final_embedding_size, 1], dtype=torch.float, device='cuda:0'), requires_grad=True)

        self.criterion = torch.nn.BCELoss().to('cuda:0')

    def set_weight(self, weight):
        print(torch.from_numpy(weight).to('cuda:0'))
        self._arch_parameters['first_order'].data = torch.from_numpy(
            weight).to('cuda:0').reshape(self.num_1, 1, self.num_fields)

    def set_embedding(self, embedding):
        self.embedding.embedding.weight.data = torch.from_numpy(
            embedding).to('cuda:0')

    def new(self):
        model_new = StackNetwork(self.args).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data = y.data.clone()
        return model_new

    def arch_parameters(self):
        if self.order == '1':
            return [self._arch_parameters['first_order']]
        elif self.order == '12' or self.order == '2':
            return [self._arch_parameters['first_order'], self._arch_parameters['second_order']]
        else:
            pass

    def binarize(self):
        if self.order == '1':
            self._cache = self._arch_parameters['first_order'].clone()
        elif self.order == '12' or self.order == '2':
            self._cache = [self._arch_parameters['first_order'].clone(
            ), self._arch_parameters['second_order'].clone()]
        else:
            pass

    def recover(self):
        if self.order == '1':
            self._arch_parameters['first_order'].data = self._cache
        elif self.order == '12' or self.order == '2':
            self._arch_parameters['first_order'].data = self._cache[0]
            self._arch_parameters['second_order'].data = self._cache[1]
        else:
            pass
        del self._cache

    def get_embedding(self):
        return self.embedding.embedding.weight.cpu().detach()

    def get_weight(self):
        return self._arch_parameters['first_order'].data.cpu().detach()

    def get_rank(self):
        a = self._arch_parameters['first_order'].data.cpu().detach()[0][0]
        # print(a)
        b = self._arch_parameters['second_order'].data.cpu().detach()[0][0]
        new_rank = []
        for i in range(len(a)):
            for j in range(i+1, len(b)):
                new_rank.append(a[i] * b[j] + a[j] * b[i])
        print('rank: %s' % list(np.argsort(new_rank)[::-1]))

    def genotype(self):
        genotypes = []
        genotype_ps = []
        if self.order == '12' or self.order == '2':
            try:
                a = self._cache
            except:
                a = [self._arch_parameters['first_order'],
                     self._arch_parameters['second_order']]
            print(a)
            for k in range(a[0].shape[0]):
                print(a[0], k)
                genotype = [a[0][k, 0].argmax().cpu().numpy()]
                genotypes.append(genotype)
                genotype_p = [F.softmax(a[0][k, 0], dim=-1).cpu().detach()]
                genotype_ps.append(genotype_p)
            for k in range(a[1].shape[0]):
                genotype = [a[1][k, 0].argmax().cpu().numpy()]
                genotypes.append(genotype)
                genotype_p = [F.softmax(a[1][k, 0], dim=-1).cpu().detach()]
                genotype_ps.append(genotype_p)
            return genotypes, genotype_ps
        elif self.order == '1':
            try:
                a = self._cache
            except:
                a = self._arch_parameters['first_order']
            for k in range(a.shape[0]):
                genotype = [a[k, 0].argmax().cpu().numpy()]
                genotypes.append(genotype)
                genotype_p = [F.softmax(a[k, 0], dim=-1).cpu().detach()]
                genotype_ps.append(genotype_p)
            return genotypes, genotype_ps

    def forward_full(self, x):
        if self.order == '1':
            x_embedding = self.embedding.embedding(x)
            arch_order1 = self._arch_parameters['first_order'][:, 0, :].reshape(
                1, -1, self.num_fields)
            mixed_embedding = torch.matmul(
                arch_order1, x_embedding).reshape(-1, self.num_1 * self.embed_dim)
            pred = torch.sum(mixed_embedding, dim=1)
            output = torch.sigmoid(pred)
        elif self.order == '12' or self.order == '2':
            x_embedding = self.embedding.embedding(x)
            arch_1 = self._arch_parameters['first_order'][:, 0, :].reshape(
                1, -1, self.num_fields)
            arch_2 = self._arch_parameters['second_order'][:, 0, :].reshape(
                1, -1, self.num_fields)
            fea_1 = torch.matmul(arch_1, x_embedding).reshape(-1,
                                                              self.num_2 * self.embed_dim)
            fea_2 = torch.matmul(arch_2, x_embedding).reshape(-1,
                                                              self.num_2 * self.embed_dim)
                                                            
            fea_3 = torch.matmul(arch_2+arch_1*2, x_embedding).reshape(-1,
                                                              self.num_2 * self.embed_dim)
            cross = fea_1 * fea_2 * fea_3
            if self.ensemble:
                final_feature = torch.cat([fea_1, cross], dim=1)
            else:
                final_feature = cross
            pred = torch.sum(final_feature, dim=1)
            output = torch.sigmoid(pred)
        return output

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        theta = _concat(self.parameters()).detach()
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            loss, self.parameters())).detach() + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment+dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.zero_grad()
        self.optimizer.zero_grad()
        self.binarize()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer, self.coeff)
        else:
            self._backward_step(input_valid, target_valid, self.coeff)
        self.recover()
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, coeff):
        logits = self.forward_full(input_valid)
        loss = self.compute_loss(logits, target_valid, coeff)
        loss.backward()

    def compute_loss(self, logits, target, coeff):
        a = torch.pow(
            torch.sum(self._arch_parameters['second_order'], dim=0), 2)
        b = torch.sum(
            torch.pow(self._arch_parameters['second_order'], 2), dim=0)
        constraint = 0.5 * coeff * torch.sum(a - b)
        return self.criterion(logits, target) + constraint

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, coeff):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)
        unrolled_logits = unrolled_model.forward_full(input_valid)
        unrolled_loss = self.compute_loss(unrolled_logits, target_valid, coeff)
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.detach() for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(
            vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
        for v, g in zip(self.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.new()
        model_dict = self.state_dict()

        params, offset = {}, 0
        for k, v in self.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.parameters(), vector):
            p.data.add_(R, v)
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        grads_p = torch.autograd.grad(loss, self.arch_parameters())

        for p, v in zip(self.parameters(), vector):
            p.data.sub_(2*R, v)
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        grads_n = torch.autograd.grad(loss, self.arch_parameters())

        for p, v in zip(self.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


class GTN(torch.nn.Module):
    def __init__(self, args):
        super(GTN, self).__init__()
        self.args = args
        field_dims = args.field_dims
        cross_num = args.cross_num
        embed_dim = args.embed_dim
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.order = args.order

        # self.cross_num = cross_num
        self.num_1, self.num_2 = [1, 1]
        self.network_weight_decay = args.weight_decay
        self.coeff = args.coeff

        self._arch_parameters = dict()
        self._arch_parameters['arbitrary'] = Variable(torch.ones(
            [self.num_1, 1, self.num_fields], dtype=torch.float, device='cuda:0') / 2, requires_grad=self.order != '12')
        self._arch_parameters['first_order'].data.add_(
            torch.randn_like(self._arch_parameters['first_order'])*1e-3)

        self._arch_parameters['second_order'] = Variable(torch.ones(
            [self.num_2, 1, self.num_fields], dtype=torch.float, device='cuda:0') / 2, requires_grad=True)
        self._arch_parameters['second_order'].data.add_(
            torch.randn_like(self._arch_parameters['second_order'])*1e-3)

        self.optimizer = torch.optim.Adam(self.arch_parameters(
        ), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        if self.order == '2':
            final_embedding_size = self.num_2 * self.embed_dim
        elif self.order == '1':
            final_embedding_size = self.num_1 * self.embed_dim
        elif self.order == '12':
            final_embedding_size = (self.num_1 + self.num_2) * self.embed_dim
        else:
            raise NotImplementedError
        self.mlp_prediction = MultiLayerPerceptron(final_embedding_size, [int(
            final_embedding_size/2)], dropout=0.5, output_layer=True).to('cuda:0')
        self.linear = Variable(torch.zeros(
            [final_embedding_size, 1], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.linear.data.add_(torch.randn_like(self.linear)*1e-2)
        self.bias = Variable(torch.zeros(
            [final_embedding_size, 1], dtype=torch.float, device='cuda:0'), requires_grad=True)

        self.criterion = torch.nn.BCELoss().to('cuda:0')

    def set_weight(self, weight):
        print(torch.from_numpy(weight).to('cuda:0'))
        self._arch_parameters['first_order'].data = torch.from_numpy(
            weight).to('cuda:0').reshape(self.num_1, 1, self.num_fields)

    def set_embedding(self, embedding):
        self.embedding.embedding.weight.data = torch.from_numpy(
            embedding).to('cuda:0')

    def new(self):
        model_new = StackNetwork(self.args).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data = y.data.clone()
        return model_new

    def arch_parameters(self):
        if self.order == '1':
            return [self._arch_parameters['first_order']]
        elif self.order == '12':
            return [self._arch_parameters['first_order'], self._arch_parameters['second_order']]
        else:
            return [self._arch_parameters['second_order']]

    def binarize(self):
        if self.order == '1':
            self._cache = self._arch_parameters['first_order'].clone()
        elif self.order == '12':
            self._cache = [self._arch_parameters['first_order'].clone(
            ), self._arch_parameters['second_order'].clone()]
        else:
            self._cache = self._arch_parameters['second_order'].clone()
        pass

    def recover(self):
        if self.order == '1':
            self._arch_parameters['first_order'].data = self._cache
        elif self.order == '12':
            self._arch_parameters['first_order'].data = self._cache[0]
            self._arch_parameters['second_order'].data = self._cache[1]
        else:
            self._arch_parameters['second_order'].data = self._cache
        del self._cache

    def get_embedding(self):
        return self.embedding.embedding.weight.cpu().detach()

    def get_weight(self):
        return self._arch_parameters['first_order'].data.cpu().detach()

    def genotype(self):
        genotypes = []
        genotype_ps = []
        if self.order == '2':
            try:
                a = self._cache
            except:
                a = self._arch_parameters['second_order']
            for k in range(a.shape[0]):
                genotype = [a[k, 0].argmax().cpu().numpy(),
                            a[k, 1].argmax().cpu().numpy()]
                genotypes.append(genotype)
                genotype_p = [F.softmax(
                    a[k, 0], dim=-1).cpu().detach(), F.softmax(a[k, 1], dim=-1).cpu().detach()]
                genotype_ps.append(genotype_p)
            return genotypes, genotype_ps
        elif self.order == '12':
            try:
                a = self._cache
            except:
                a = [self._arch_parameters['first_order'],
                     self._arch_parameters['second_order']]
            print(a)
            for k in range(a[0].shape[0]):
                print(a[0], k)
                genotype = [a[0][k, 0].argmax().cpu().numpy()]
                genotypes.append(genotype)
                genotype_p = [F.softmax(a[0][k, 0], dim=-1).cpu().detach()]
                genotype_ps.append(genotype_p)
            for k in range(a[1].shape[0]):
                genotype = [a[1][k, 0].argmax().cpu().numpy()]
                genotypes.append(genotype)
                genotype_p = [F.softmax(a[1][k, 0], dim=-1).cpu().detach()]
                genotype_ps.append(genotype_p)
            return genotypes, genotype_ps
        elif self.order == '1':
            try:
                a = self._cache
            except:
                a = self._arch_parameters['first_order']
            for k in range(a.shape[0]):
                genotype = [a[k, 0].argmax().cpu().numpy()]
                genotypes.append(genotype)
                genotype_p = [F.softmax(a[k, 0], dim=-1).cpu().detach()]
                genotype_ps.append(genotype_p)
            return genotypes, genotype_ps

    def forward_full(self, x):
        if self.order == '1':
            x_embedding = self.embedding.embedding(x)
            arch_order1 = self._arch_parameters['first_order'][:, 0, :].reshape(
                1, -1, self.num_fields)
            mixed_embedding = torch.matmul(
                arch_order1, x_embedding).reshape(-1, self.num_1 * self.embed_dim)
            pred = torch.sum(mixed_embedding, dim=1)
            output = torch.sigmoid(pred)
        elif self.order == '2':
            x_embedding = self.embedding.embedding(x)
            arch_1 = self._arch_parameters['first_order'][:, 0, :].reshape(
                1, -1, self.num_fields)
            arch_2 = self._arch_parameters['second_order'][:, 0, :].reshape(
                1, -1, self.num_fields)
            fea_1 = torch.matmul(arch_1, x_embedding).reshape(-1,
                                                              self.cross_num * self.embed_dim)
            fea_2 = torch.matmul(arch_2, x_embedding).reshape(-1,
                                                              self.cross_num * self.embed_dim)
            cross = fea_1 * fea_2
            pred = torch.sum(cross, dim=1)
            output = torch.sigmoid(pred)
        elif self.order == '12':
            x_embedding = self.embedding.embedding(x)
            arch_1 = self._arch_parameters['first_order'][:, 0, :].reshape(
                1, -1, self.num_fields)
            arch_2 = self._arch_parameters['second_order'][:, 0, :].reshape(
                1, -1, self.num_fields)
            fea_1 = torch.matmul(arch_1, x_embedding).reshape(-1,
                                                              self.num_2 * self.embed_dim)
            fea_2 = torch.matmul(arch_2, x_embedding).reshape(-1,
                                                              self.num_2 * self.embed_dim)
            cross = fea_1 * fea_2
            final_feature = torch.cat([fea_1, cross], dim=1)
            pred = torch.sum(final_feature, dim=1)
            output = torch.sigmoid(pred)
        return output

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        theta = _concat(self.parameters()).detach()
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            loss, self.parameters())).detach() + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment+dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.zero_grad()
        self.optimizer.zero_grad()
        self.binarize()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer, self.coeff)
        else:
            self._backward_step(input_valid, target_valid, self.coeff)
        self.recover()
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, coeff):
        logits = self.forward_full(input_valid)
        loss = self.compute_loss(logits, target_valid, coeff)
        loss.backward()

    def compute_loss(self, logits, target, coeff):
        a = torch.pow(
            torch.sum(self._arch_parameters['second_order'], dim=0), 2)
        b = torch.sum(
            torch.pow(self._arch_parameters['second_order'], 2), dim=0)
        constraint = 0.5 * coeff * torch.sum(a - b)
        return self.criterion(logits, target) + constraint

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, coeff):
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer)
        unrolled_logits = unrolled_model.forward_full(input_valid)
        unrolled_loss = self.compute_loss(unrolled_logits, target_valid, coeff)
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.detach() for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(
            vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
        for v, g in zip(self.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.new()
        model_dict = self.state_dict()

        params, offset = {}, 0
        for k, v in self.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.parameters(), vector):
            p.data.add_(R, v)
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        grads_p = torch.autograd.grad(loss, self.arch_parameters())

        for p, v in zip(self.parameters(), vector):
            p.data.sub_(2*R, v)
        logits = self.forward_full(input)
        loss = self.criterion(logits, target)
        grads_n = torch.autograd.grad(loss, self.arch_parameters())

        for p, v in zip(self.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
