import matplotlib.pyplot as plt

def myscatter(df, x1var_name, x2var_name, pred_varname, show=False, exp_prefix=None):
	plt.scatter( df[df[pred_varname] == 1][x1var_name]
				,df[df[pred_varname] == 1][x2var_name]
				,marker='D', color='r', label="y == 1")
	plt.scatter( df[df[pred_varname] == 0][x1var_name]
				,df[df[pred_varname] == 0][x2var_name]
				,marker='o', color='b', label="y == 0")
	plt.xlabel(x1var_name)
	plt.ylabel(x2var_name)
	plt.legend(loc='best', shadow=True)
	if show:
		plt.show()
		
	if exp_prefix is not None:	
		exp_filename = exp_prefix + "scatter_" + x1var_name + "_" + x2var_name + "_" + '.png'
		print "		exporting plot:{0} ...".format(exp_filename)
		plt.savefig(exp_filename, dpi=200)
		
def mybuild_colormesh(df, j, b, i, a, common, clf):	
	import numpy as np
	
	## 			Plot the decision boundary. For that, we will assign a color to each
	## 			point in the mesh [xb_min, xb_max]x[xa_min, xa_max].
	xb_min, xb_max = df[b][common].min() - 1, df[b][common].max() + 1
	xa_min, xa_max = df[a][common].min() - 1, df[a][common].max() + 1
	xb_step = max(int((xb_max - xb_min) / 10), 1)
	xa_step = max(int((xa_max - xa_min) / 10), 1)
	xbb, xaa = np.meshgrid(np.arange(xb_min, xb_max, xb_step),
						   np.arange(xa_min, xa_max, xa_step))
	random_X = np.empty([len(df.columns), xbb.ravel().shape[0]])
	for feat_ix, feat in enumerate(df.columns):
		random_X[feat_ix] = np.random.randint(df[feat][common].min(), df[feat][common].max(), 
												xbb.ravel().shape[0])

	random_X[j] = xbb.ravel()
	random_X[i] = xaa.ravel()
	Z = clf.predict(random_X.transpose())
	# 			Put the result into a color plot
	Z = Z.reshape(xbb.shape)

	return(xbb, xaa, Z)
                    					   			
def myscatter_matrix(frame, pred_values=[], mypred_values=[], clf=None, alpha=0.5, figsize=None, ax=None, grid=False, diagonal='hist', marker='.', density_kwds={},
					 hist_kwds={}, **kwds):

	# enhanced pd.tools.plotting.scatter_matrix
	import pandas as pd
	import pandas.core.common as com
	import numpy as np
	from matplotlib.colors import ListedColormap

	#print "in myscatter_matrix..."
	#print "		clf=%s" % clf
	#if len(mypred_values) == 0:
	#	print "mypred_values is empty"
	#else:
	#	print "mypred_values:{0}".format(mypred_values)

	from matplotlib.artist import setp

	df = frame._get_numeric_data()
	n = df.columns.size
	fig, axes = pd.tools.plotting._subplots(nrows=n, ncols=n, figsize=figsize, ax=ax,
							squeeze=False)

	# no gaps between subplots
	fig.subplots_adjust(wspace=0, hspace=0)

	mask = com.notnull(df)

	marker = pd.tools.plotting._get_marker_compat(marker)

	for i, a in zip(range(n), df.columns):
		for j, b in zip(range(n), df.columns):
			ax = axes[i, j]

			if i == j:
				values = df[a].values[mask[a].values]

				# Deal with the diagonal by drawing a histogram there.
				if diagonal == 'hist':
					ax.hist(values, **hist_kwds)
				elif diagonal in ('kde', 'density'):
					from scipy.stats import gaussian_kde
					y = values
					gkde = gaussian_kde(y)
					ind = np.linspace(y.min(), y.max(), 1000)
					ax.plot(ind, gkde.evaluate(ind), **density_kwds)
			else:
				common = (mask[a] & mask[b]).values

				# begin my mods
				#ax.scatter(df[b][common], df[a][common],
				#           marker=marker, alpha=alpha, **kwds)
				if len(pred_values) == 0 and len(mypred_values) == 0:
					ax.scatter(df[b][common], df[a][common],
					           marker=marker, alpha=alpha, **kwds)
				elif len(pred_values) > 0 and len(mypred_values) == 0:
					ax.scatter(df[b][common][pred_values == 1], df[a][common][pred_values == 1],
							   marker='D', color='r', label="y == 1", alpha=alpha, **kwds)
					ax.scatter(df[b][common][pred_values == 0], df[a][common][pred_values == 0],
							   marker='o', color='b', label="y == 0", alpha=alpha, **kwds)
					#print "myscatter_matrix: obs_n(y == 0)={0:,}".format(len(df[a][common][pred_values == 0]))
					#print "myscatter_matrix: obs_n(y == 1)={0:,}".format(len(df[a][common][pred_values == 1]))		   
					myhandles, mylabels = ax.get_legend_handles_labels()
				else:
					## 			Create color maps
					cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
					cmap_bold  = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
					xbb, xaa, Z = mybuild_colormesh(df, j, b, i, a, common, clf)
					ax.pcolormesh(xbb, xaa, Z, cmap=cmap_light)
				
					#label="true  +ve", c='g', marker="d"
					#label="false -ve", c='r', marker="x"
					#label="false +ve", c='k', marker="*"
					#label="true  -ve", c='b', marker="o"	# not displayed
					ax.scatter(	df[b][common][np.logical_and(pred_values == 1, mypred_values == 1)], 
								df[a][common][np.logical_and(pred_values == 1, mypred_values == 1)],
								marker='d', color='g', label=" true +ve", cmap=cmap_bold, alpha=alpha, **kwds)
					ax.scatter(	df[b][common][np.logical_and(pred_values == 1, mypred_values == 0)], 
								df[a][common][np.logical_and(pred_values == 1, mypred_values == 0)],
								marker='x', color='r', label="false -ve", cmap=cmap_bold, alpha=alpha, **kwds)
					ax.scatter(	df[b][common][np.logical_and(pred_values == 0, mypred_values == 1)], 
								df[a][common][np.logical_and(pred_values == 0, mypred_values == 1)],
								marker='*', color='k', label="false +ve", cmap=cmap_bold, alpha=alpha, **kwds)
					#ax.grid(True)
					myhandles, mylabels = ax.get_legend_handles_labels()

				# end my mods

			ax.set_xlabel('')
			ax.set_ylabel('')

			pd.tools.plotting._label_axis(ax, kind='x', label=b, position='bottom', rotate=True)
			ax.set_xlabel(b, fontsize=4)
			pd.tools.plotting._label_axis(ax, kind='y', label=a, position='left')
			ax.set_ylabel(a, fontsize=4)

			if j!= 0:
				ax.yaxis.set_visible(False)
			if i != n-1:
				ax.xaxis.set_visible(False)

	for ax in axes.flat:
		setp(ax.get_xticklabels(), fontsize=2) # was 8 in original version
		setp(ax.get_yticklabels(), fontsize=4)

	# begin my mods
	if len(pred_values) > 0 or len(mypred_values) > 0:
		fig.legend(myhandles, mylabels, loc='best', shadow=True, fontsize='x-small')
	# end my mods

	return axes
	
def myplot_vbar(df, bar_varname, prefix_filename, png_prefix, xlabel_str=''):
	x_pos = np.arange(df.shape[0])
	plt.bar(x_pos, df[bar_varname].values, align='center', alpha=0.4)
	plt.xticks(x_pos, df.index)
	plt.title(entity_data_filename)
	plt.ylabel(bar_varname)
	plt.xlabel(xlabel_str)
	#plt.figtext(df.ix[0, hbar_varname], y_pos[0], str(df.ix[0, label_varname]))
	exp_filename = prefix_filename + 'plt_vbar_' + png_prefix + '.png'
	print "		exporting plot:{0} ...".format(exp_filename)
	plt.savefig(exp_filename)
	if plt_disp:
		sys.stderr.write("*** displaying a plot...\n")
		plt.show()
		
def myplot_vbar_group():
	index = np.arange(5)
	bar_width = 0.35
	rects_fn = plt.bar(index + bar_width * 0.0, entity_grpd_df[:5]['mrp_csymhash_0.0_csymhash_1.0_npct']
					   ,bar_width, color='r', label='mrp_csymhash_0.0_csymhash_1.0_n%:min_tp_gap')
	rects_fp = plt.bar(index + bar_width * 1.0, entity_grpd_df[:5]['mrp_csymhash_1.0_csymhash_0.0_npct']
					   ,bar_width, color='k', label='mrp_csymhash_1.0_csymhash_0.0_n%')
	rects_tp = plt.bar(index + bar_width * 2.0, entity_grpd_df[:5]['mrp_csymhash_1.0_csymhash_1.0_npct']
					   ,bar_width, color='g', label='mrp_csymhash_1.0_csymhash_1.0_n%')
	plt.xlabel('features')
	plt.ylabel('n%')
	plt.xticks(index + bar_width, list(entity_grpd_df[:5].index))
	plt.show()
		

def myplot_hbar(df, bar_varname, prefix_filename, png_prefix, ylabel_str=''):
	y_pos = np.arange(df.shape[0])
	plt.barh(y_pos, df[bar_varname].values, align='center', alpha=0.4)
	plt.yticks(y_pos, df.index)
	plt.title(entity_data_filename)
	plt.xlabel(bar_varname)
	plt.ylabel(ylabel_str)
	#plt.figtext(df.ix[0, hbar_varname], y_pos[0], str(df.ix[0, label_varname]))
	exp_filename = prefix_filename + 'plt_hbar_' + png_prefix + '.png'
	print "		exporting plot:{0} ...".format(exp_filename)
	plt.savefig(exp_filename)
	if plt_disp:
		sys.stderr.write("*** displaying a plot...\n")
		plt.show()
		
def myplot_hbar_group(df, cols, colors=None, legend_suffix=None, ylabel=None, xlabel=None
					  ,show=False, exp_prefix=None):
					  
	import numpy as np
					  
	plt.figure()
	index = np.arange(df.shape[0])
	if df.shape[0] <= 5:
		bar_width = 0.30
	else:
		bar_width = 0.30

	for col_ix, col in enumerate(cols):
		plt.barh(index + bar_width * col_ix, df[col]
                       ,bar_width, color=colors[col_ix], label=col + legend_suffix[col_ix])

	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.yticks(index + bar_width, list(df.index), fontsize=4)
	plt.legend(loc='best', shadow=True, fontsize='xx-small')
	plt.title(list(df[-1:].index)[0], fontsize='x-small')
	if show:
		sys.stderr.write("*** displaying a plot...\n")
		plt.show()

	if exp_prefix is not None:
		exp_filename = exp_prefix + "hbar_grp" + '.png'
		print "		exporting plot:{0} ...".format(exp_filename)
		plt.savefig(exp_filename, dpi=200)		

def myplot_histogram(entity_df):	
	n, bins, patches = plt.hist(entity_df[random_varname], 50, facecolor='b', alpha=0.75)
	plt.xlabel(random_varname)
	plt.ylabel('Frequency')
	plt.title("Histogram of " + random_varname)
	plt.grid(True)
	if plt_disp:
		sys.stderr.write("*** displaying a plot...\n")
		plt.show()


