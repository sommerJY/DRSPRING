
# correlation plot 

import seaborn as sns
from scipy import stats

aa = list(np.random.randn(10))
bb = list(np.random.randn(10))
tmp_df = pd.DataFrame({'A' : aa ,'B' : bb })

sns.scatterplot(x="FlyAsh", y="Strength", data=con);

ax = sns.scatterplot(x="A", y="B", data=tmp_df)
ax.set_title("test")
ax.set_xlabel("A")
path = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_3_2/'

plotname = 'test' 
ax.figure.savefig('{}/{}.corrplot1.png'.format(path, plotname), bbox_inches = 'tight')


LMplot = sns.lmplot(x="A", y="B", data=tmp_df, palette="Set1")
path = 
plotname = 
LMplot.savefig('{}/{}.corrplot.png'.format(path, plotname), bbox_inches = 'tight')




regplot = sns.regplot(x="A", y="B", ci=68, data=tmp_df)
regplot.figure.savefig('{}/{}.corrplot2.png'.format(path, plotname), bbox_inches = 'tight')

def r2(x,y):
    return(stats.pearsonr(x,y)[0]**2)


jplot = sns.jointplot(x="A", y="B", ci=68, data=tmp_df, kind='reg')
pr,_ = stats.pearsonr(tmp_df['A'], tmp_df['B'])
sr,_ = stats.spearmanr(tmp_df['A'], tmp_df['B'])
jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(tmp_df['A'].min()+ 0.01,tmp_df['B'].max()- 0.01 ),ha='left', va='center',)
jplot.ax_joint.scatter(tmp_df['A'], tmp_df['B'])
jplot.set_axis_labels(xlabel='a', ylabel='b', size=15)
jplot.figure.savefig('{}/{}.corrplot2.png'.format(path, plotname), bbox_inches = 'tight')


stats.pearsonr(con['Strength'], con['FlyAsh'])

g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
               palette="Set1")
