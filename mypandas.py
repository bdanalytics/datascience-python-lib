import pandas as pd
import numpy as np

def mygroupby(df, groupby_cols, grouped_cols, agg_funcs):

	# MultiIndex data structure for rows:
	#	n			: # of rows in series / dataframe
	#	p			: # of groupby vars
	#	values(p)	: distinct values of var p
	#
	# 	names	: list of p
	#	labels	: list of p x n
	# 	levels	: list of p x values(p)
	#
	# MultiIndex data structure for cols:
	#	n			: # of grouped cols
	#	p			: # of vars in index
	#	agg(p)		: distinct agg of var p
	#
	# 	names	: list of p
	#	labels	: list of p x n
	# 	levels	: list of p x values(p) / agg(p)
	#

	grpd_df = df.groupby(groupby_cols)[grouped_cols].agg(agg_funcs)
	labels_arr = np.array(grpd_df.index.labels).transpose()
	index_arr = [' '.join(str(label)) for label in labels_arr]
	grpd_df.index = index_arr
	grpd_df.index.name = 'groupby_cols'
	levels_arr = np.array(grpd_df.columns.levels)
	colnames_lst = []
	for var_ix in range(levels_arr[0].shape[0]):
		for agg_ix in range(levels_arr[1].shape[0]):
			colnames_lst.append(levels_arr[0][var_ix] + '_' + levels_arr[1][agg_ix])
	
	grpd_df.columns = colnames_lst
	grpd_df[glb_predict_varname + '_size_pct'] = (100.0 * grpd_df[glb_predict_varname + '_size']) \
													/ np.sum(grpd_df[glb_predict_varname + '_size'])
	grpd_df[glb_predict_varname + '_sum_pct'] = (100.0 * grpd_df[glb_predict_varname + '_sum']) \
													/ np.sum(grpd_df[glb_predict_varname + '_sum'])

	return grpd_df
	
def mypivot_table():

	colnames_lst = []
	colnames_lst.append(names_arr[0] + '_' + str(levels_arr[0][0])
				+ '_' + names_arr[1] + '_' + str(levels_arr[1][0])
				+ '_' + names_arr[2] + '_' + str(levels_arr[2][0]))
	colnames_lst.append(names_arr[0] + '_' + str(levels_arr[0][0])
				+ '_' + names_arr[1] + '_' + str(levels_arr[1][1])
				+ '_' + names_arr[2] + '_' + str(levels_arr[2][0]))

	colnames_lst.append(names_arr[0] + '_' + str(levels_arr[0][1])
				+ '_' + names_arr[1] + '_' + str(levels_arr[1][0])
				+ '_' + names_arr[2] + '_' + str(levels_arr[2][0]))
	colnames_lst.append(names_arr[0] + '_' + str(levels_arr[0][1])
				+ '_' + names_arr[1] + '_' + str(levels_arr[1][1])
				+ '_' + names_arr[2] + '_' + str(levels_arr[2][0]))

	return 0	