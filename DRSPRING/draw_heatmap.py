heatmap 진짜 그려봐야하나 고민 


# JY_6


# 일단 예측값 가져오긴 해야함 
# 시각화 예시를 들어보기 위한거니까 구린애여도 노상관

WORK_NAME = 'WORK_322' # 349
W_NAME = 'W322'
PRJ_NAME = 'M3V5'
MJ_NAME = 'M3V5'
WORK_DATE = '23.05.26' # 349
MISS_NAME = 'MIS2'
PPI_NAME = '349'

# 지금 평균낸거로 하는데, 사실 이렇게 하믄 안되고 retrain 시킨애로 봐야할거임 
# 아 바본가 
# 지금 뭔가 깨달았다
# 사실 처음부터 5CV 를 하면 애초부터 pred 값을 얻을 수가 있구나
# test 고정하는게 의미가 물론 parameter setting 할때는 의미가 있는데
# CV 가 그렇게 되면 딱이구나 
# 존나 바보다 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 어휴 

# test 데이터 가져오게 하기 
# 지금은 그냥 테스트용 


ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_train'])]
ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]


                    PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

                    ALL_TEST_RESULTS = pd.read_csv(PRJ_PATH + 'TEST_RESULTS.5CV.csv', sep = '\t')

                    FILT_TEST_RES = ALL_TEST_RESULTS[['DrugCombCCLE','type', 'cell_onehot','ANS','pred_5_CV0', 'pred_5_CV1', 'pred_5_CV2', 'pred_5_CV3', 'pred_5_CV4']]

                    FILT_TEST_RES['tissue'] = [a.split('_')[1] for a in ALL_TEST_RESULTS.DrugCombCCLE]
                    FILT_TEST_RES['PRED'] = ALL_TEST_RESULTS[['pred_5_CV0', 'pred_5_CV1', 'pred_5_CV2', 'pred_5_CV3', 'pred_5_CV4']].mean(axis =1)
                    FILT_TEST_RES2 = FILT_TEST_RES[['DrugCombCCLE','tissue','type', 'cell_onehot','ANS','PRED']]



ABCS_test_result = ABCS_test[['DrugCombCCLE','type','CID_CID', 'CID_CID_CCLE']]
ABCS_test_result['tissue'] = [a.split('_')[1] for a in ABCS_test_result.DrugCombCCLE]
ABCS_test_result['ANS'] = np.random.uniform(-10,10, 35931)
ABCS_test_result['PRED'] = np.random.uniform(-10,10, 35931)
ABCS_test_result['DIFF'] = abs(ABCS_test_result.ANS - ABCS_test_result.PRED)
ABCS_test_result['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


# make heatmap table


cells_info = pd.DataFrame(
    data = np.inf,
    columns=['tissue'],
    index=list(set(ABCS_test_result.DrugCombCCLE))
)

cells_info['tissue'] = ['_'.join(a.split('_')[1:]) for a in list(set(ABCS_test_result.DrugCombCCLE))]



my_heatmap = pd.DataFrame(
    data = 0, #np.inf,
    columns=list(set(ABCS_test_result.DrugCombCCLE)),
    index=list(set(ABCS_test_result.CID_CID))
)

c_c_c_list = list(ABCS_test_result.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : 
    if c_ind%100 == 0 : 
        print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
        datetime.now() 
    c_c_c = c_c_c_list_set[c_ind]
    tmp_res = ABCS_test_result[ABCS_test_result.CID_CID_CCLE==c_c_c]
    c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
    c = c_c_c.split('___')[2]
    if tmp_res.shape[0] ==1 :
        my_heatmap.at[c_c, c] = tmp_res['DIFF'].item()
    else  :
        my_heatmap.at[c_c, c] = np.mean(tmp_res['DIFF'])
    


tissue_order = ['SKIN', 'BREAST', 'LUNG', 'OVARY', 'LARGE_INTESTINE', 'CENTRAL_NERVOUS_SYSTEM', 'KIDNEY', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','PROSTATE', 'BONE', 'PLEURA']
tiss_cell_dict ={a : list(cells_info[cells_info.tissue==a].index) for a in tissue_order }


my_map_skin = my_heatmap[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap[tiss_cell_dict['PLEURA']]



cell_bar = pd.DataFrame(
    data = np.inf,
    columns = ['DIFF_M'],
    index = list(set(ABCS_test_result.DrugCombCCLE))
)

for a in cell_bar.index :
    meanval = np.mean(ABCS_test_result[ABCS_test_result.DrugCombCCLE==a]['DIFF'])
    cell_bar.at[a,'DIFF_M'] = meanval





col_ha = HeatmapAnnotation( TISSUE=anno_simple(cells_info.tissue, add_text=False,legend=True), axis=1,
                           MEAN=anno_barplot(cell_bar, legend=True, colors = [color_dict[tt] for tt in cells_info.tissue] ),
                           legend=True, legend_gap=5, hgap=0.5) # plot=True, 


my_heatmap2 = pd.concat([
    my_map_skin, my_map_breast, my_map_lung, my_map_ovary, my_map_Lins,
    my_map_nerv, my_map_kidn, my_map_hema, my_map_prot, my_map_bone, my_map_pleu], axis = 1)

my_heatmap3 = my_heatmap2.sort_values(list(my_heatmap2.columns))


plt.figure(figsize=(5.5, 6.5))
my_cm = ClusterMapPlotter(data=my_heatmap3, top_annotation=col_ha,
                       col_cluster=False, row_cluster=False,
                       # col_split=cells_info.tissue,
                       # col_split_gap=0.5,
                       # row_dendrogram=True, # label='values',
                       show_rownames=False, show_colnames=True,
                       verbose=0, legend_gap=5,  # tree_kws={'row_cmap': 'Set1'}, 
                       cmap='YlGn', xticklabels_kws={'labelrotation':-90,'labelcolor':'black'})
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')











# 아예 dot plot 으로 시도 

my_heatmap_dot = pd.DataFrame(
    data = "NA", #np.inf,
    columns=list(set(ABCS_test_result.DrugCombCCLE)),
    index=list(set(ABCS_test_result.CID_CID))
)

c_c_c_list = list(ABCS_test_result.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : 
    if c_ind%100 == 0 : 
        print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
        datetime.now() 
    c_c_c = c_c_c_list_set[c_ind]
    tmp_res = ABCS_test_result[ABCS_test_result.CID_CID_CCLE==c_c_c]
    c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
    c = c_c_c.split('___')[2]
    if tmp_res.shape[0] ==1 :
        tmp_result = tmp_res['DIFF'].item()
    else  :
        tmp_result = np.mean(tmp_res['DIFF'])
    # 
    if tmp_result < 5 : 
        my_heatmap_dot.at[c_c, c] = "under5"
    elif  tmp_result < 10 : 
        my_heatmap_dot.at[c_c, c] = "under10"
    else :
        my_heatmap_dot.at[c_c, c] = "over10"
    

my_map_skin = my_heatmap_dot[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap_dot[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap_dot[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap_dot[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap_dot[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap_dot[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap_dot[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap_dot[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap_dot[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap_dot[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap_dot[tiss_cell_dict['PLEURA']]



cell_bar = pd.DataFrame(
    data = np.inf,
    columns = ['DIFF_M'],
    index = list(set(ABCS_test_result.DrugCombCCLE))
)

for a in cell_bar.index :
    meanval = np.mean(ABCS_test_result[ABCS_test_result.DrugCombCCLE==a]['DIFF'])
    cell_bar.at[a,'DIFF_M'] = meanval




colors = [color_dict[tt] for tt in cells_info.tissue]
col_ha = HeatmapAnnotation( TISSUE=anno_simple(cells_info.tissue, add_text=False,legend=True), axis=1,
                           MEAN=anno_barplot(cell_bar, legend=True ),
                           legend=True, legend_gap=5, hgap=0.5) # plot=True, 


my_heatmap2 = pd.concat([
    my_map_skin, my_map_breast, my_map_lung, my_map_ovary, my_map_Lins,
    my_map_nerv, my_map_kidn, my_map_hema, my_map_prot, my_map_bone, my_map_pleu], axis = 1)

my_heatmap3 = my_heatmap2.sort_values(list(my_heatmap2.columns), ascending = False)




my_dc = DotClustermapPlotter(
    data=my_heatmap3, 
    top_annotation=col_ha,
    cmap={'under5':'Reds','under10':'Purples','over10':'Blues'},
    colors={'under5':'red','under10':'purple','over10':'green'},
    #col_split_gap=0.5, #row_dendrogram=True col_split=11, 
)












ABCS_test_result2 = copy.deepcopy(ABCS_test_result)




c_c_c_list = list(ABCS_test_result2.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result2.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : 
    if c_ind%1000 == 0 : 
        print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
        datetime.now() 
    #
    c_c_c = c_c_c_list_set[c_ind]
    tmp_res = ABCS_test_result2[ABCS_test_result2.CID_CID_CCLE==c_c_c]
    if tmp_res.shape[0] > 0 :
        new_diff = np.mean(tmp_res['DIFF'])
        indices = list(tmp_res.index)
        for ind in indices :
            ABCS_test_result2.at[ind, 'DIFF'] = new_diff


ABCS_test_result3 = ABCS_test_result2[['DrugCombCCLE', 'CID_CID', 'CID_CID_CCLE', 'tissue','DIFF']].drop_duplicates()
ABCS_test_result3['D_level'] = ABCS_test_result3.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )


my_dc = DotClustermapPlotter(
    data=ABCS_test_result3, x = 'CID_CID', y = 'DrugCombCCLE', value='DIFF',
    #top_annotation=col_ha,
    hue='D_level', #cmap={'under5':'Reds','under10':'Purples','over10':'Blues'},
    #colors={'under5':'red','under10':'purple','over10':'green'},
    #col_split_gap=0.5, #row_dendrogram=True col_split=11, 
)





# type 때문에 더 잘나왔을수도 있음... 아닌가 그래도 어차피 순서 고려까지 아니더라도 적당히 smiles 로 나눴으니까 괜찮나? 
# 고오민 










# 

onco로 마지막으로한번만 더 

cols=['under5', 'under10', 'over10']
colors=["red","blue","#008000"]

ABCS_test_result3['under5'] = ABCS_test_result3.D_level.apply(lambda x: 1 if x == 'under5' else 0)
ABCS_test_result3['under10'] = ABCS_test_result3.D_level.apply(lambda x:1 if x == 'under10' else 0)
ABCS_test_result3['over10'] = ABCS_test_result3.D_level.apply(lambda x:1 if x == 'over10' else 0)

row_vc=ABCS_test_result3.groupby('CID_CID').apply(lambda x:x.loc[:,cols].sum())
col_vc=ABCS_test_result3.groupby('DrugCombCCLE').apply(lambda x:x.loc[:,cols].sum())


plt.figure(figsize=(12,8))

my_cp = oncoPrintPlotter(
    data=ABCS_test_result3, x = 'CID_CID', y = 'DrugCombCCLE', 
    values =cols, 
    colors = colors,
   
)


plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')


df=pd.DataFrame(ABCS_test_result3[cols].iloc[1,:].values.tolist()).apply(lambda x:x/x.sum(),axis=1).fillna(0)


 #value='DIFF',
    #top_annotation=col_ha,
    #cmap={'under5':'Reds','under10':'Purples','over10':'Blues'},
    #colors={'under5':'red','under10':'purple','over10':'green'},
    #col_split_gap=0.5, #row_dendrogram=True col_split=11, 





아 진자 진심으로 모든 조합에 대한 내용이 없어서 그러니 
생각해보면 지금 이 조합이 test 에 대한거만 하고 있는데, 전체가 들어오면 그럴 필요가 없을지도?
아닌가 그래도 채워야하나 


all_ccle = list(set(ABCS_test_result3.DrugCombCCLE))
all_cidcid= list(set(ABCS_test_result3.CID_CID))

items = [all_ccle, all_cidcid]
from itertools import product
item_list = list(product(*items))

CCLE = [a for a,b in item_list]
CID_CID = [b for a,b in item_list]

ABCS_test_result4 = pd.DataFrame({'CID' : CID_CID, 'CCLE' : CCLE})
ABCS_test_result4['under5'] = 0
ABCS_test_result4['under10'] = 0
ABCS_test_result4['over10'] = 0
ABCS_test_result4['result'] = 0




for ind in range(len(item_list)) :
    if ind%1000 == 0 : 
        print(str(ind)+'/'+str(len(item_list)) )
        datetime.now() 
    a = item_list[ind][0]
    b = item_list[ind][1]
    tmp = ABCS_test_result3[ (ABCS_test_result3.DrugCombCCLE == a) & (ABCS_test_result3.CID_CID == b) ]
    if tmp.shape[0] == 1 :
        ABCS_test_result4.at[ind, 'under5'] = tmp['under5'].item()
        ABCS_test_result4.at[ind, 'under10'] = tmp['under10'].item()
        ABCS_test_result4.at[ind, 'over10'] = tmp['over10'].item()
        #print(tmp.shape)
    elif tmp.shape[0] > 1 :
        print(ind)


                                            a = 'A375_SKIN'
                                            b = '3385___46926350'

cols=['under5', 'under10', 'over10', 'result']
colors=["#ff647e","#ffc1cb","#ffe28a", '#f5f8fb']


row_vc=ABCS_test_result4.groupby('CID').apply(lambda x:x.loc[:,cols].sum())
col_vc=ABCS_test_result4.groupby('CCLE').apply(lambda x:x.loc[:,cols].sum())


# 문제는 전체를 다 그리려고 하면 뭔가 python 으로 하려는 과정에서 문제가 생긴다는것...
# 아 이거 하나 그리려고 R 을 써야하나 진짜 
# 왜자꾸 inf 문제가 생기는지 알수가 없네
# 뭐든 예시를 가져가야할것 같은디요



tmp = ABCS_test_result4[ABCS_test_result4.CCLE.isin(tiss_cell_dict['SKIN'])]

plt.figure(figsize=(12,24)) 

my_cp = oncoPrintPlotter(
    data=tmp, x = 'CCLE', y = 'CID', 
    values =cols, 
    colors = colors  
)

plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')












# complexheatmap 포기하고 그냥 matplotlib 써보기 
from matplotlib import colors as mcolors
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib




fig, ax = plt.subplots(1, 1, figsize=(8, 6))

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


im, _ = heatmap(harvest, vegetables, farmers, ax=ax,
                cmap="Wistia", cbarlabel="harvest [t/year]")
annotate_heatmap(im, valfmt="{x:.1f}", size=7)


annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
                 textcolors=("red", "black"))


corr_matrix = np.corrcoef(harvest)
im, _ = heatmap(corr_matrix, vegetables, vegetables, ax=ax4,
                cmap="PuOr", vmin=-1, vmax=1,
                cbarlabel="correlation coeff.")


def func(x, pos):
    return f"{x:.2f}".replace("0.", ".").replace("1.00", "")

annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)


plt.tight_layout()
plt.show()









def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    #
    if not ax:
        ax = plt.gca()
    #
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    #
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    #
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    #
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    #
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    #
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    #
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    #
    return im, cbar



def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    #
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    #
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.
    #
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    #
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    #
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    #
    return texts    


def func(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")




my_heatmap3



fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        data = np.random.randn(6, 6)
        y = ["Prod. {}".format(i) for i in range(10, 70, 10)]
        x = ["Cycle {}".format(i) for i in range(1, 7)]

data = my_heatmap3
y = list(my_heatmap3.index)
x = list(my_heatmap3.columns)



qrates = list("ABCDEFG")
norm = matplotlib.colors.BoundaryNorm(np.linspace(-3.5, 3.5, 8), 7)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

im, _ = heatmap(my_heatmap3, y, x, ax=ax,
                cmap=plt.get_cmap("PiYG", 4), norm=norm,
                cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
                cbarlabel="Quality Rating")

annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
                 textcolors=("red", "black"))




plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')








import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,1,figsize=(18,8))
my_colors=[(0.2,0.3,0.3),(0.4,0.5,0.4),(0.1,0.7,0),(0.1,0.7,0)]

sns.heatmap(dt_tweet_cnt, cmap=my_colors, square=True, linewidth=0.1, linecolor=(0.1,0.2,0.2), ax=ax)

colorbar = ax.collections[0].colorbar
M=dt_tweet_cnt.max().max()
colorbar.set_ticks([1/8*M,3/8*M,6/8*M])
colorbar.set_ticklabels(['low','med','high'])

plt.show()



for 
my_heatmap3

value_to_int = {j:i for i,j in enumerate(pd.unique(df.values.ravel()))} # like you did
n = len(value_to_int)     






















def heatmap_binary(df,
            edgecolors='w',
            #cmap=mpl.cm.RdYlGn,
            log=False):    
    width = len(df.columns)/7*10
    height = len(df.index)/7*10
    #
    fig, ax = plt.subplots(figsize=(20,10))#(figsize=(width,height))
    #
    cmap, norm = mcolors.from_levels_and_colors([0, 0.05, 1],['Teal', 'MidnightBlue'] ) # ['MidnightBlue', Teal]['Darkgreen', 'Darkred']

    heatmap = ax.pcolor(df ,
                        edgecolors=edgecolors,  # put white lines between squares in heatmap
                        cmap=cmap,
                        norm=norm)
    data = df.values
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x + 0.5 , y + 0.5, '%.4f' % data[y, x], #data[y,x] +0.05 , data[y,x] + 0.05
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='w')


    ax.autoscale(tight=True)  # get rid of whitespace in margins of heatmap
    ax.set_aspect('equal')  # ensure heatmap cells are square
    ax.xaxis.set_ticks_position('top')  # put column labels at the top
    ax.tick_params(bottom='off', top='off', left='off', right='off')  # turn off ticks

    ax.set_yticks(np.arange(len(df.index)) + 0.5)
    ax.set_yticklabels(df.index, size=20)
    ax.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax.set_xticklabels(df.columns, rotation=90, size= 15)

    # ugliness from http://matplotlib.org/users/tight_layout_guide.html
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="1%")
    fig.colorbar(heatmap, cax=cax)

df1 = pd.DataFrame(np.random.choice([0, 0.75], size=(4,5)), columns=list('ABCDE'), index=list('WXYZ'))
heatmap_binary(df1)












# 아예 dot plot 으로 시도 




ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_train'])]
ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]


ABCS_test_result = ABCS_test[['DrugCombCCLE','type','CID_CID', 'CID_CID_CCLE']]
ABCS_test_result['tissue'] = [a.split('_')[1] for a in ABCS_test_result.DrugCombCCLE]
ABCS_test_result['ANS'] = np.random.uniform(-10,10, 35931)
ABCS_test_result['PRED'] = np.random.uniform(-10,10, 35931)
ABCS_test_result['DIFF'] = abs(ABCS_test_result.ANS - ABCS_test_result.PRED)
ABCS_test_result['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )


# test 에 들어간 애들로만 봐서 지금 제대로 안갈리는것 같음 

ABCS_test_result = A_B_C_S_SET_SM[['DrugCombCCLE','type','CID_CID', 'CID_CID_CCLE']]
ABCS_test_result['tissue'] = [a.split('_')[1] for a in ABCS_test_result.DrugCombCCLE]
ABCS_test_result['ANS'] = np.random.uniform(-10,10, 184503)
ABCS_test_result['PRED'] = np.random.uniform(-10,10, 184503)
ABCS_test_result['DIFF'] = abs(ABCS_test_result.ANS - ABCS_test_result.PRED)
ABCS_test_result['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )




my_heatmap_dot = pd.DataFrame(
    data = "NA", #np.inf,
    columns=list(set(ABCS_test_result.DrugCombCCLE)),
    index=list(set(ABCS_test_result.CID_CID))
)

c_c_c_list = list(ABCS_test_result.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : # 대충 18만줄 대상으로 30분 소요 
    if c_ind%1000 == 0 : 
        print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
        datetime.now() 
    c_c_c = c_c_c_list_set[c_ind]
    tmp_res = ABCS_test_result[ABCS_test_result.CID_CID_CCLE==c_c_c]
    c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
    c = c_c_c.split('___')[2]
    if tmp_res.shape[0] ==1 :
        tmp_result = tmp_res['DIFF'].item()
    else  :
        tmp_result = np.mean(tmp_res['DIFF'])
    # 
    if tmp_result < 5 : 
        my_heatmap_dot.at[c_c, c] = "under5"
    elif  tmp_result < 10 : 
        my_heatmap_dot.at[c_c, c] = "under10"
    else :
        my_heatmap_dot.at[c_c, c] = "over10"
    

# my_heatmap_dot.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/test.csv', index =False, sep ='\t')


tissue_order = ['SKIN', 'BREAST', 'LUNG', 'OVARY', 'LARGE_INTESTINE', 'CENTRAL_NERVOUS_SYSTEM', 'KIDNEY', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','PROSTATE', 'BONE', 'PLEURA']
tiss_cell_dict ={a : list(cells_info[cells_info.tissue==a].index) for a in tissue_order }


my_map_skin = my_heatmap_dot[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap_dot[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap_dot[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap_dot[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap_dot[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap_dot[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap_dot[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap_dot[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap_dot[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap_dot[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap_dot[tiss_cell_dict['PLEURA']]






# seaborn 으로 해결? 

# row index

my_heatmap_dot_c = my_heatmap_dot.replace(value_to_int)
nach = my_heatmap_dot_c.sum(axis=1)
new_ind = nach.sort_values(ascending = False).index
my_heatmap_dot_c = my_heatmap_dot_c.loc[new_ind]


# col index 만드는 중이었음 
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist

linked = linkage(my_map_skin, 'complete')

gg = sns.clustermap(my_map_skin, row_cluster=True, col_cluster = True) 

my_map_skin = my_heatmap_dot_c[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap_dot_c[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap_dot_c[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap_dot_c[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap_dot_c[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap_dot_c[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap_dot_c[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap_dot_c[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap_dot_c[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap_dot_c[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap_dot_c[tiss_cell_dict['PLEURA']]





cmap = [mcolors.to_rgba('#f5f8fb'),mcolors.to_rgba('#ffe28a'),mcolors.to_rgba("#ffc1cb"),mcolors.to_rgba("#ff647e")]

fig, ax = plt.subplots(1,1,figsize=(18,18))
sns.heatmap(my_heatmap_dot_c, ax = ax, cmap=cmap) 
colorbar = ax.collections[0].colorbar 
r = colorbar.vmax - colorbar.vmin 
colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
colorbar.set_ticklabels(list(value_to_int.keys()))            


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')
plt.close()








# seaborn 으로 해결? 2 

my_heatmap_dot

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


tissue_map = pd.DataFrame(my_heatmap_dot.columns, columns= ['cell'])
tissue_map['tissue'] = ['_'.join(a.split('_')[1:]) for a in tissue_map.cell]
tissue_map['col'] = tissue_map['tissue'].map(color_dict)

col_colors = list(tissue_map['col'])


#fig = plt.subplots()

# value_to_int = {j:i for i,j in enumerate(pd.unique(my_heatmap_dot.values.ravel()))} # like you did
n = len(value_to_int)     
# cmap = sns.color_palette("Pastel2", n) 
cmap = [mcolors.to_rgba('#f5f8fb'),mcolors.to_rgba('#ffe28a'),mcolors.to_rgba("#ffc1cb"),mcolors.to_rgba("#ff647e")]

gg = sns.clustermap(
    my_heatmap_dot.replace(value_to_int),  cmap=cmap, 
    figsize=(20,20),
    row_cluster=True, col_cluster = True, 
    col_colors = col_colors, dendrogram_ratio=0.2, yticklabels=False) 


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example1.pdf", bbox_inches='tight')
plt.close()





# skin 이 지랄이라서 skin 만 빼고 돌려보기 

my_heatmap_dot2 = my_heatmap_dot.drop(tiss_cell_dict['SKIN'], axis =1)
my_heatmap_dot3 = my_heatmap_dot2.replace(value_to_int)
nach = my_heatmap_dot3.sum(axis=1)
naind = [a for a in my_heatmap_dot3.index if nach[a] != 0]
my_heatmap_dot4 = my_heatmap_dot3.loc[naind]

gg = sns.clustermap(
    my_heatmap_dot4,  cmap=cmap, 
    figsize=(20,20),
    row_cluster=True, col_cluster = True, 
    col_colors = col_colors, dendrogram_ratio=0.2, yticklabels=False) 


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example2.pdf", bbox_inches='tight')
plt.close()




