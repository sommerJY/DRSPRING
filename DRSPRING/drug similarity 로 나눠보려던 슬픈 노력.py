drug similarity 로 나눠보려던 슬픈 노력



# 야매 tanimoto sim 비교 

cid_sm_list1 = A_B_C_S_SET[['CID_A','ROW_CAN_SMILES']]
cid_sm_list1.columns = ['CID','SM']
cid_sm_list2 = A_B_C_S_SET[['CID_B','COL_CAN_SMILES']]
cid_sm_list2.columns = ['CID','SM']
cid_sm_list = pd.concat([cid_sm_list1, cid_sm_list2]).drop_duplicates()

cid_sm_list_filt = cid_sm_list[cid_sm_list.CID.isin(CID_set)]

from rdkit import DataStructs

def calculate_internal_pairwise_similarities(smiles_list) :
	"""
	Computes the pairwise similarities of the provided list of smiles against itself.
		Symmetric matrix of pairwise similarities. Diagonal is set to zero.
	"""
	mols = [Chem.MolFromSmiles(x.strip()) for x in smiles_list]
	fps = [Chem.RDKFingerprint(x) for x in mols]
	nfps = len(fps)
	#
	similarities = np.zeros((nfps, nfps))
	#
	for i in range(1, nfps):
		sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
		similarities[i, :i] = sims
		similarities[:i, i] = sims
	return similarities 


# similarity_matrix = 1 / (1 + distance_matrix)

sim_matrix_order = list(cid_sm_list_filt.SM)
sim_matrix = calculate_internal_pairwise_similarities(sim_matrix_order)


from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation

dist_matrix = 1-sim_matrix
#clustering = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='complete' ).fit(dist_matrix) # 이미 거리 계산 했으므로
#
#kmeans = KMeans(n_clusters=5, random_state=0)
#cluster_labels = kmeans.fit_predict(dist_matrix)
#
#clustering = DBSCAN(eps= 0.3, min_samples=2).fit(dist_matrix)
#
#clustering = AffinityPropagation(random_state=5, affinity='precomputed').fit(dist_matrix)




model = manifold.TSNE(n_components = 2, metric = 'precomputed')
X_embedded = model.fit_transform(np.array(sim_matrix))

TSNE_DF = pd.DataFrame(X_embedded)
TSNE_DF.columns = ['comp1','comp2']
TSNE_DF['drug'] = list(cid_sm_list_filt.CID)
TSNE_DF['cluster'] = cluster_labels


color_dict = {a : ['#d95e92','#FF7514','#025669','#308446','#84C3BE'][a] for a in range(5)}

fig = plt.figure(figsize=(10,10))
sns.scatterplot(data = TSNE_DF, x = 'comp1', y = 'comp2', hue = 'cluster', legend='full', palette=color_dict, sizes =[40]*pca_df.shape[0])  # hue = 'tissue', , palette=color_dict, size = 'tissue', sizes=[40]*10
for i in range(TSNE_DF.shape[0]):
    plt.text(TSNE_DF.comp1[i], TSNE_DF.comp2[i], TSNE_DF.drug[i]) # -0.2+0.2, size = 8

plt.tight_layout()
path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W403_349_MIS2'
plotname = 'tsne_oneil_chem1'
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')




# 3D
import plotly.express as px

model = manifold.TSNE(n_components = 3, metric = 'precomputed')
X_embedded = model.fit_transform(np.array(sim_matrix))

TSNE_DF = pd.DataFrame(X_embedded)
TSNE_DF.columns = ['comp1','comp2','comp3']
TSNE_DF['drug'] = list(cid_sm_list_filt.CID)
TSNE_DF['cluster'] = cluster_labels

fig = px.scatter_3d(TSNE_DF, x='comp1', y='comp2', z='comp3', hover_name="drug",
              color='cluster', color_discrete_map = color_dict)

path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W403_349_MIS2'
plotname = 'tsne_oneil_chem1'
fig.write_html('{}/{}.html'.format(path, plotname))







# PCA 로 갈아타보자 
# Center the data
mean_similarity = np.mean(sim_matrix, axis=0)
centered_similarity_matrix = sim_matrix - mean_similarity

# Calculate the covariance matrix
covariance_matrix = np.cov(centered_similarity_matrix, rowvar=False)

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Sort eigenvectors by decreasing eigenvalues and select the top-k components
k = 2  # Number of principal components to keep
top_eigenvectors = eigenvectors[:, -k:][::-1]

# Project data onto the selected principal components
pca_scores = np.dot(centered_similarity_matrix, top_eigenvectors)
pca_df = pd.DataFrame(pca_scores)
pca_df.columns = ['pc1','pc2']
pca_df['drug'] = list(cid_sm_list_filt.CID)
pca_df['cluster'] = cluster_labels

color_dict = {a : ['#d95e92','#FF7514','#025669','#308446','#84C3BE'][a] for a in range(5)}

fig = plt.figure(figsize=(10,10))
sns.scatterplot(data = pca_df, x = 'pc1', y = 'pc2', hue = 'cluster', legend='full', palette=color_dict, sizes =[40]*pca_df.shape[0])  # hue = 'tissue', , palette=color_dict, size = 'tissue', sizes=[40]*10
for i in range(pca_df.shape[0]):
    plt.text(pca_df.pc1[i], pca_df.pc2[i], pca_df.drug[i]) # -0.2+0.2, size = 8

plt.tight_layout()
path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_W403_349_MIS2'
plotname = 'tsne_oneil_chem2'
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')





row_means = []
for i in range(sim_matrix.shape[0]):
	indexes = [a for a in range(sim_matrix.shape[0])]
	indexes.pop(i)
	row_tmp = sim_matrix[i][indexes]
	row_mean = np.mean(row_tmp)
	row_means = row_means + [row_mean]







