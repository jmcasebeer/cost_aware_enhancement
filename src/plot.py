""" 
Plot generator for monitoring models that are training.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_plot(per_db_results, idx, metric='ponder', title='Per-db'):
    fig, ax = plt.subplots()
    
    # Sort keys in ascending order
    db_lvls = list(per_db_results.keys())
    db_lvls = sorted(db_lvls, key=lambda x: float(x))
    
    # Get data for plotting and statstics per-dB
    x_coord = np.arange(len(db_lvls))
    stats = {}
    stats['mean'] = np.asarray([np.asarray(per_db_results[key][metric]).mean() for key in db_lvls])
    stats['var'] =  np.asarray([np.asarray(per_db_results[key][metric]).var() for key in db_lvls])
    stats['mins'] =  np.asarray([np.asarray(per_db_results[key][metric]).min() for key in db_lvls])
    stats['maxes'] =  np.asarray([np.asarray(per_db_results[key][metric]).max() for key in db_lvls])
    stats['medians'] =  np.asarray([np.median(np.asarray(per_db_results[key][metric])) for key in db_lvls])
    
    # Plot the data
    plot_labels = ["{0:.3f}".format(float(x)) for x in db_lvls]
    ax.errorbar(x_coord, stats['mean'], stats['var'], 
                fmt='ok', ecolor='blue', lw=5)
    ax.errorbar(x_coord, stats['mean'], 
                [stats['mean']-stats['mins'], stats['maxes']-stats['mean']], 
                fmt='.k', ecolor='blue', alpha=0.5, lw=2)
    ax.plot(x_coord, stats['medians'], marker='.', linestyle='', alpha=0.5, color='red')
    ax.set_xticks(x_coord)
    ax.set_xticklabels(plot_labels, rotation=30)
    title = ' '.join([title, metric.capitalize()])
    ax.set_title('{} at {}'.format(title, idx))

    return fig, ax, stats
