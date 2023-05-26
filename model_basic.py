import torch
import torch.nn as nn

class MF_BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num, model):
		super(MF_BPR, self).__init__()

		self.model = model
		self.embed_user_MLP = nn.Embedding(user_num, factor_num)
		self.embed_item_MLP = nn.Embedding(item_num, factor_num)
		self._init_weight_()

	def _init_weight_(self):
		nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
		nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

	def forward_one_item(self, user, item):

		embed_user_MLP = self.embed_user_MLP(user)
		embed_item_MLP = self.embed_item_MLP(item)
		pred = torch.mul(embed_user_MLP, embed_item_MLP)
		pred = torch.sum(pred, 1)

		return pred.view(-1)

	def forward(self, user, item_i, item_j):
		prediction_i = self.forward_one_item(user, item_i)
		prediction_j = self.forward_one_item(user, item_j)

		return prediction_i, prediction_j