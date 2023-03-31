

# 민지꺼 중간 값 확인 


from sklearn.decomposition import PCA
import numpy as np



 /st06/jiyeonH/13.DD_SESS/01.PRJ2/chem_mid_fugcn_a3t4.csv &  cell_mid_fugcn_a3t4.csv

/st06/jiyeonH/13.DD_SESS/01.PRJ2/chem_mid_fugcn_hst1.csv & cell_mid_fugcn_hst1.csv



chem_mid_978 = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/chem_mid_fugcn_a3t4.csv')
chem_mid_978.columns = ['Unnamed: 0','pert_id','cell_id']+["col_{}".format(C) for C in range(8)]

cell_mid_978 = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/cell_mid_fugcn_a3t4.csv')
cell_mid_978.columns = ['Unnamed: 0','pert_id','cell_id']+["col_{}".format(C) for C in range(8)]

chem_mid_349 = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/chem_mid_fugcn_hst1.csv')
chem_mid_349.columns = ['Unnamed: 0','pert_id','cell_id']+["col_{}".format(C) for C in range(8)]

cell_mid_349 = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/cell_mid_fugcn_hst1.csv')
cell_mid_349.columns = ['Unnamed: 0','pert_id','cell_id']+["col_{}".format(C) for C in range(8)]



tmp = cell_mid_978[['cell_id','col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']].drop_duplicates()
tmp.describe()

tmp = chem_mid_978[['pert_id','col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']].drop_duplicates()
tmp.describe()


tmp = cell_mid_349[['cell_id','col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']].drop_duplicates()
tmp.describe()

tmp = chem_mid_349[['pert_id','col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']].drop_duplicates()
tmp.describe()





# violin plot  보면? 

import matplotlib.pyplot as plt
import numpy as np

# 2. 데이터 준비
tmp = cell_mid_978[['cell_id','col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']].drop_duplicates()
df_name = 'cell_mid_978' 

tmp = chem_mid_978[['pert_id','col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']].drop_duplicates()
df_name = 'chem_mid_978' 

tmp = cell_mid_349[['cell_id','col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']].drop_duplicates()
df_name = 'cell_mid_349' 

tmp = chem_mid_349[['pert_id','col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']].drop_duplicates()
df_name = 'chem_mid_349' 



tmp2 = tmp[['col_0','col_1','col_2','col_3', 'col_4', 'col_5', 'col_6', 'col_7']]

# 1. 기본 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

# 3. 그래프 그리기
fig, ax = plt.subplots()

violin = ax.violinplot(tmp2)
ax.set_xlabel('dim')
ax.set_ylabel('Value')

plt.savefig('/home/jiyeonH/'+"{}.png".format(df_name), format = 'png', dpi=300)
plt.close()









