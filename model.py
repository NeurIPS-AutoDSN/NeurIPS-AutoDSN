import torch
import time
import torch.nn.functional as F

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class NetworkCTR(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, genotype, high_order):
        """
        :param genotype: the gate state of feature interactions.
        # serve as the attention units in the retrain stage.
        (feature_num, 1)
        """
        super().__init__()
        self.num_fields = len(field_dims)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.high_order = high_order
        if self.high_order:
            self.genotype_2nd = torch.nn.Parameter(
                torch.from_numpy(genotype['2nd']).float())
            self.genotype_3rd = torch.nn.Parameter(
                torch.from_numpy(genotype['3rd']).float())
            # self.genotype_4th = torch.nn.Parameter(
                # torch.from_numpy(genotype['4th']).float())
        else:
            self.genotype = torch.nn.Parameter(
                torch.from_numpy(genotype).float())
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        if self.high_order:
            self.embed_output_dim = (len(genotype['2nd']) + len(genotype['3rd'])) * self.embed_dim 
        else:
            self.embed_output_dim = len(genotype['2nd']) * self.embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, [16, 16], dropout=0.2)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embedding_x = self.embedding(
            x)     # shape: (batch_size, num_fields, embed_dim)
        if self.high_order:
            field1, field2, field3 = list(), list(), list()
            for i in range(self.num_fields - 2):
                for j in range(i + 1, self.num_fields-1):
                    for k in range(j + 1, self.num_fields):
                        field1.append(i), field2.append(j), field3.append(k)
            p1, q1, r1 = embedding_x[:, field1], embedding_x[:,
                                                             field2], embedding_x[:, field3]
            inner_product_3rd = p1 * q1 * r1

            # field1, field2, field3, field4 = list(), list(), list(), list()
            # for i in range(self.num_fields - 3):
            #     for j in range(i + 1, self.num_fields-2):
            #         for k in range(j + 1, self.num_fields-1):
            #             for t in range(k+1, self.num_fields):
            #                 field1.append(i), field2.append(j), field3.append(k), field4.append(t)
            # p1, q1, r1, s1 = embedding_x[:, field1], embedding_x[:, field2], embedding_x[:, field3], embedding_x[:, field4]
            # inner_product_4th = p1 * q1 * r1 * s1

            # print(field1)
            row, col = list(), list()
            for i in range(self.num_fields - 1):
                for j in range(i + 1, self.num_fields):
                    row.append(i), col.append(j)
            p, q = embedding_x[:, row], embedding_x[:, col]
            inner_product_2nd = p * q

            nasfm_2nd_prod = torch.mul(inner_product_2nd, self.genotype_2nd.detach())
            nasfm_3rd_prod = torch.mul(inner_product_3rd, self.genotype_3rd.detach())
            # masfm_4th_prod = torch.mul(inner_product_4th, self.genotype_4th.detach())

            nasfm_2nd = torch.sum(torch.sum(nasfm_2nd_prod, dim=1), dim=1, keepdim=True)
            nasfm_3rd = torch.sum(torch.sum(nasfm_3rd_prod, dim=1), dim=1, keepdim=True)
            # nasfm_4th = torch.sum(torch.sum(masfm_4th_prod, dim=1), dim=1, keepdim=True)
            nasfm = nasfm_2nd  + nasfm_3rd# + nasfm_4th
            # print(nasfm_2nd_prod.shape, nasfm_3rd_prod.shape, nasfm_2nd.shape, nasfm_3rd.shape)
            # nasfm_cat = nasfm_2nd_prod
            # nasfm_cat = torch.cat([nasfm_2nd_prod, nasfm_3rd_prod])
            # batch_size = nasfm_cat.shape()[0]
            # nasfm_cat = nasfm_cat.reshape(batch_size, -1)
        else:
            row, col = list(), list()
            for i in range(self.num_fields - 1):
                for j in range(i + 1, self.num_fields):
                    row.append(i), col.append(j)
            p, q = embedding_x[:, row], embedding_x[:, col]
            inner_product = p * q
            nasfm = torch.sum(torch.sum(
                torch.mul(inner_product, self.genotype.detach()), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + nasfm
        # print(nasfm_cat.shape)
        # x = self.linear(x) + nasfm + self.mlp(nasfm_cat.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class NetworkCTR_Sparse(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, genotype, high_order, fields_list):
        """
        :param genotype: the gate state of feature interactions.
        # serve as the attention units in the retrain stage.
        (feature_num, 1)
        """
        super().__init__()
        self.num_fields = len(field_dims)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.high_order = high_order
        self.fields_list = fields_list
        if self.high_order:
            self.genotype_2nd = torch.nn.Parameter(
                torch.from_numpy(genotype['2nd']).float())
            self.genotype_3rd = torch.nn.Parameter(
                torch.from_numpy(genotype['3rd']).float())
        else:
            self.genotype = torch.nn.Parameter(
                torch.from_numpy(genotype).float())
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        if self.high_order:
            self.embed_output_dim = (len(genotype['2nd']) + len(genotype['3rd'])) * self.embed_dim 
        else:
            self.embed_output_dim = len(genotype['2nd']) * self.embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, [16, 16], dropout=0.2)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embedding_x = self.embedding(
            x)     # shape: (batch_size, num_fields, embed_dim)
        count2, count3 = 0, 0
        if self.high_order:
            field1, field2, field3 = self.fields_list[1]
            p1, q1, r1 = embedding_x[:, field1], embedding_x[:,
                                                             field2], embedding_x[:, field3]
            inner_product_3rd = p1 * q1 * r1

            row, col = self.fields_list[0]
            p, q = embedding_x[:, row], embedding_x[:, col]
            inner_product_2nd = p * q

            nasfm_2nd_prod = torch.mul(inner_product_2nd, self.genotype_2nd.detach())
            nasfm_3rd_prod = inner_product_3rd
            nasfm_2nd = torch.sum(torch.sum(nasfm_2nd_prod, dim=1), dim=1, keepdim=True)
            nasfm_3rd = torch.sum(torch.sum(nasfm_3rd_prod, dim=1), dim=1, keepdim=True)
            nasfm = nasfm_2nd  + nasfm_3rd
        else:
            row, col = list(), list()
            for i in range(self.num_fields - 1):
                for j in range(i + 1, self.num_fields):
                    row.append(i), col.append(j)
            p, q = embedding_x[:, row], embedding_x[:, col]
            inner_product = p * q
            nasfm = torch.sum(torch.sum(
                torch.mul(inner_product, self.genotype.detach()), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + nasfm
        return torch.sigmoid(x.squeeze(1))

class NetworkCTR_Old(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, genotype):
        """
        :param genotype: the gate state of feature interactions.
        # serve as the attention units in the retrain stage.
        (feature_num, 1)
        """
        super().__init__()
        self.num_fields = len(field_dims)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.genotype = torch.nn.Parameter(torch.from_numpy(genotype).float())
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embedding_x = self.embedding(
            x)     # shape: (batch_size, num_fields, embed_dim)
        row, col = list(), list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row.append(i), col.append(j)
        p, q = embedding_x[:, row], embedding_x[:, col]
        inner_product = p * q
        nasfm = torch.sum(torch.sum(
            torch.mul(inner_product, self.genotype.detach()), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + nasfm
        return torch.sigmoid(x.squeeze(1))
