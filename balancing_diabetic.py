from pandas import read_csv
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

filename = 'data/Scaling/my_diabetic_data_scaled_minmax.csv'
file = "unbalanced"
original = read_csv(filename, index_col=0)
class_var = 'readmitted'
target_count = original[class_var].value_counts()
middle_class = 1
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()
#ind_positive_class = target_count.index.get_loc(positive_class)
print('Minority class=', positive_class, ':', target_count[positive_class])
print('Majority class=', negative_class, ':', target_count[negative_class])
print('Middle class=', middle_class, ':', target_count[middle_class])
values = {'Original': [target_count[positive_class], target_count[middle_class], target_count[negative_class]]}

figure()
bar_chart(target_count.index, target_count.values, title='Class balance')
savefig(f'images/balancing/{file}_balance.png')

df_positives = original[original[class_var] == positive_class]
df_negatives = original[original[class_var] == negative_class]
df_middle = original[original[class_var] == middle_class] 

from pandas import concat, DataFrame

df_pos_sample_1 = DataFrame(df_positives.sample(len(df_negatives), replace=True))
df_pos_sample_2 = DataFrame(df_middle.sample(len(df_negatives), replace=True))
df_over = concat([df_pos_sample_1, df_pos_sample_2, df_negatives], axis=0)
df_over.to_csv(f'data/Balancing/{file}_over.csv', index=False)
values['OverSample'] = [len(df_pos_sample_1),len(df_pos_sample_2), len(df_negatives)]
print('Minority class=', positive_class, ':', len(df_pos_sample_1))
print('Middle class=', middle_class, ':', len(df_pos_sample_2))
print('Majority class=', negative_class, ':', len(df_negatives))
print('Proportion:', round(len(df_pos_sample_1) / len(df_negatives), 2), ': 1')