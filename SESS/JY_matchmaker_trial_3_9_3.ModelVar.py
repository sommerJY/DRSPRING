



self.G_convs_1_chem = torch.nn.ModuleList([pyg_nn.SAGEConv(self.G_indim_chem, self.G_hiddim_chem)])
self.G_convs_1_chem.extend([pyg_nn.SAGEConv(self.G_hiddim_chem, self.G_hiddim_chem) for i in range(self.G_layer_chem-2)])
self.G_convs_1_chem.extend([pyg_nn.SAGEConv(self.G_hiddim_chem, self.G_hiddim_chem)])
self.G_bns_1_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])







# 기본 세번 
class GNN(torch.nn.Module):
	def __init__(self, num_layer, in_dim, hid_dim, out_dim, normalize=False, lin=True): 
		super(GNN, self).__init__()
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.lin = lin
        #
        self.Convs = torch.nn.ModuleList([pyg_nn.DenseGCNConv(self.in_dim, self.hid_dim, self.normalize)])
        self.Convs.extend([pyg_nn.DenseGCNConv(self.hid_dim, self.hid_dim, self.normalize) for i in range(self.num_layer-2)])
        self.Convs.extend([pyg_nn.DenseGCNConv(self.hid_dim, self.hid_dim, self.normalize)])
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.hid_dim) for i in range(self.num_layer-1)])
        #
		if lin is True : 
			self.lin = torch.nn.Linear(2*self.hid_dim+self.out_dim, self.out_dim )
		else : 
			self.lin = None
	#
	def bn (self, step , x ):
		batch_size , num_nodes , num_features = x.size()
		x = x.view(-1, num_features)
		x = getattr(self, 'bn{}'.format(step))(x)
		x = x.view(batch_size, num_nodes, num_features)
		return x 
	#
	def forward(self, x, adj, mask=None):
		batch_size, num_nodes, in_channels = x.size()
        layer_dict = {}
        x0 = x
        key_list = ["x_"+str(i) for i range()]
        for i in range(self.num_layer):
            layer_dict[]
            
		
		x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
		x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
		x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))
		x = torch.cat([x1, x2, x3], dim=-1)
		if self.lin is not None:
			x = F.relu(self.lin(x))
		return x


class DIFFPOOL_logP(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, max_nodes, num_pool, ratio , dropout):
		super(DIFFPOOL_logP, self).__init__()
		#
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.conv_num = 3
		self.num_pool = num_pool
		self.max_nodes = max_nodes
		self.ratio = ratio
		self.dropout = dropout
		#
		self.num_nodes = [ceil(self.ratio * self.max_nodes)]
		self.check_num_nodes()
		#
		self.POOLS = nn.ModuleList([GNN(self.input_dim, self.hidden_dim, self.num_nodes[1] )])
		self.POOLS.extend([GNN(self.conv_num * self.hidden_dim, self.hidden_dim, self.num_nodes[i+2]) for i in range(self.num_pool-1)])
		#
		self.EMBED = nn.ModuleList([GNN(self.input_dim, self.hidden_dim, self.hidden_dim, lin=False)])
		self.EMBED.extend([GNN(self.conv_num * self.hidden_dim, self.hidden_dim, self.hidden_dim, lin=False) for i in range(self.num_pool)])
		#
		self.lin1 = torch.nn.Linear(self.conv_num * self.hidden_dim, self.hidden_dim)
		self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
		self.lin3 = torch.nn.Linear(self.hidden_dim, self.output_dim)  
		#
		self.reset_parameters()
	#
	def reset_parameters(self):
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()
		self.lin3.reset_parameters()
	#
	def check_num_nodes(self):
		for L in range(self.num_pool):
			this_node = ceil( self.ratio * self.num_nodes[-1])
			self.num_nodes.append(this_node)
	#	
	def forward(self, x, adj, mask = None):
		x = F.normalize(x)
		s = self.POOLS[0](x, adj, mask)
		x = self.EMBED[0](x, adj, mask)
		x, adj, l, e = pyg_nn.dense_diff_pool(x, adj, s, mask)
		#
		for i in range(self.num_pool-1):
			s = self.POOLS[i+1](x, adj)
			x = self.EMBED[i+1](x, adj)
			x, adj, l, e = pyg_nn.dense_diff_pool(x, adj, s)
		x = self.EMBED[-1](x, adj)
		#
		x = x.mean(dim=1)
		x = F.relu(self.lin1(x))
		x = torch.tanh(self.lin2(x))
		x = self.lin3(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		return x





def train(epoch):
	model.train()
	loss_all = 0
	for step, data in enumerate(train_loader):
		X = data[0]
		ADJ = data[1].float()
		ANS = data[2].flatten().long()
		#
		optimizer.zero_grad()
		output, _, _ = model(X, ADJ)
		loss = F.nll_loss(output, ANS.view(-1))
		loss.backward()
		loss_all += ANS.size(0) * loss.item()
		optimizer.step()
	return loss_all / len(train_loader.dataset.indices)

def test(loader):
	model.eval()
	correct = 0
	for step, data in enumerate(test_loader):
		with torch.no_grad():
			X = data[0]
			ADJ = data[1].float()
			ANS = data[2].flatten().long()
			#
			pred = model(X, ADJ)[0].max(dim=1)[1]
			correct += pred.eq(ANS.view(-1)).sum().item()
	RESULT = correct / len(test_loader.dataset.indices)
	print(pred)
	return RESULT

