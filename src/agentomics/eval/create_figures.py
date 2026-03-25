import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import defaultdict

mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "path.simplify": False,
})

colors = [
    "B22222", #red
    "FFC107", #yellow
    "4682B4", #blue
    "6F8480", #gray
    "004D40", #green

]
def hex_to_rgb(hex_code):
    return tuple(int(hex_code[i:i+2], 16)/255 for i in (0, 2, 4))

colors_rgb = [hex_to_rgb(hex) for hex in colors]

def plot_both_arch_bars(
    sums_a, total_runs_a, fm_array_a,
    sums_b, total_runs_b, fm_array_b,
    label_a="A", label_b="B", figname='unnamed',
):


    # --- Configuration ---
    BAR_COLOR_A = colors_rgb[2]
    BAR_COLOR_B = colors_rgb[0]

    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 20
    TICK_FONT_SIZE = 20

    # print(f"normalized based on {total_runs_a} samples ({label_a})")
    # print(f"normalized based on {total_runs_b} samples ({label_b})")

    # FM stats
    fm_a_percentage = (fm_array_a.sum()/len(fm_array_a)) * 100
    fm_b_percentage = (fm_array_b.sum()/len(fm_array_b)) * 100
    # print(f"FM {label_a}: {fm_a_percentage:.2f}%")
    # print(f"FM {label_b}: {fm_b_percentage:.2f}%")

    # --- Normalize exactly as before ---
    sums_a = sums_a[sums_a > 0]
    sums_b = sums_b[sums_b > 0]

    sums_a = (sums_a / total_runs_a) * 100
    sums_b = (sums_b / total_runs_b) * 100

    # print('sum of a', sum(sums_a))
    # print('sum of b', sum(sums_b))

    # Align classes
    all_classes = sums_a.index.union(sums_b.index)
    sums_a = sums_a.reindex(all_classes, fill_value=0)
    sums_b = sums_b.reindex(all_classes, fill_value=0)

    # Sort by combined magnitude (DESCENDING - highest at top)
    order = (sums_a + sums_b).sort_values(ascending=True).index  # ascending=True for descending visual order
    sums_a = sums_a.loc[order]
    sums_b = sums_b.loc[order]

    # Long-form DataFrame for architecture bars
    df_plot = pd.concat(
        [
            sums_a.rename("Percent of runs").to_frame().assign(Group=label_a),
            sums_b.rename("Percent of runs").to_frame().assign(Group=label_b),
        ]
    ).reset_index().rename(columns={"index": "arch"})

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, len(order) * 0.4 + 1.5))
    
    # Plot architecture bars
    sns.barplot(
        data=df_plot,
        y="arch",
        x="Percent of runs",
        hue="Group",
        palette=[BAR_COLOR_A, BAR_COLOR_B],
        ax=ax,
        order=order  # Keep the descending order
    )
    
    # Get number of architecture bars
    n_arch_bars = len(order)
    gap_size = 0.8  # Gap between architecture and FM bars
    fm_position = -1 - gap_size  # Negative position = bottom of chart
    
    # Manually add FM bars at the bottom
    bar_height = 0.4
    group_offset = 0.2
    
    ax.barh(fm_position - group_offset, fm_a_percentage, 
            height=bar_height, color=BAR_COLOR_A, label=None)
    ax.barh(fm_position + group_offset, fm_b_percentage, 
            height=bar_height, color=BAR_COLOR_B, label=None)
    
    # Update y-axis to include FM bars at bottom
    ax.set_ylim(fm_position - 1, n_arch_bars - 0.5)
    
    # Set y-tick positions and labels
    ytick_positions = [fm_position] + list(range(len(order)))
    ytick_labels = ['FM Embeddings'] + list(order)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=TICK_FONT_SIZE)
    
    # Add horizontal line to separate categories (centered in the gap)
    separator_y = -0.5  # Midpoint between index -1 and 0
    ax.axhline(y=separator_y, color='black', 
               linestyle='--', linewidth=2, alpha=1.0)
    
    ax.tick_params(
        axis="both",
        which="major",
        length=6,
        width=1.5,
    )

    ax.set_xlabel("Architecture Usage (%)", fontsize=LABEL_FONT_SIZE)
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax.set_ylabel("")
    
    # Fix legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='center right', fontsize=LABEL_FONT_SIZE, frameon=False)

    sns.despine(top=True, right=True, ax=ax)
    plt.tight_layout()

    plt.savefig(f'./paper_tables/{figname}.svg', format='svg')

def create_arch_plot():
    arch_df_extra = pd.read_csv('./paper_tables/architectures.csv')
    sum_all_iters = arch_df_extra['arch'].value_counts()
    only_best_iters = arch_df_extra[arch_df_extra['iteration_number'] == arch_df_extra['best_iteration_logged']]
    sum_best_iters = only_best_iters['arch'].value_counts()
    plot_both_arch_bars(
        sums_a = sum_best_iters, 
        total_runs_a =len(only_best_iters), 
        fm_array_a = only_best_iters['foundation_model_embeddings_frozen'],
        sums_b=sum_all_iters, 
        total_runs_b=len(arch_df_extra), 
        fm_array_b = arch_df_extra['foundation_model_embeddings_frozen'],
        label_a="Output Iterations", 
        label_b="All Iterations",
        figname='archs_fig',
    )

def create_time_plots(fin_df):
    import ast
    def add_time_to_norm_wouldbetest(row, to_normalize_col='would_be_test'):
        would_be_test = ast.literal_eval(row[to_normalize_col])
        time_marks = ast.literal_eval(row['cumulative_iterations_duration'])
        time_to_wbtest = zip(time_marks, would_be_test)
        # some iters that were new_best failed to run as test -> would output nothing, filtering those here
        non_nan_time_to_wbtest = [(time, wbtest) for time,wbtest in time_to_wbtest if not pd.isna(wbtest)]
        non_nan_wbtest = np.array([wbtest for time,wbtest in non_nan_time_to_wbtest])

        lower_is_better = row['main_metric'] == 'MAE'
        worst_test_metric = max(non_nan_wbtest) if lower_is_better else min(non_nan_wbtest)
        best_test_metric  = min(non_nan_wbtest) if lower_is_better else max(non_nan_wbtest)
        normalized_would_be_test = (worst_test_metric - non_nan_wbtest) / (worst_test_metric - best_test_metric) if lower_is_better \
            else (non_nan_wbtest - worst_test_metric) / (best_test_metric - worst_test_metric)
        if(best_test_metric == worst_test_metric):
            assert all(pd.isna(normalized_would_be_test)) or len(set(normalized_would_be_test)) == 1, normalized_would_be_test
            normalized_would_be_test = [1 for _ in normalized_would_be_test]

        assert len(normalized_would_be_test) == len(non_nan_time_to_wbtest)
        time_to_normalized_wbtest = {time: normalized_would_be_test[i] for i,(time,wbtest) in enumerate(non_nan_time_to_wbtest)}
        return time_to_normalized_wbtest
    
    def unfold_time(row, col='time_to_norm_wouldbetest'):
        times_to_fill = all_timestamps #across all experiments
        new_time_to_norm_wouldbetest = {}
        old_time_to_norm_wouldbetest = row[col]
        last_norm_wouldbetest = None
        for time in times_to_fill:
            if time in old_time_to_norm_wouldbetest.keys():
                new_time_to_norm_wouldbetest[time] = old_time_to_norm_wouldbetest[time]
                last_norm_wouldbetest = old_time_to_norm_wouldbetest[time]
            elif last_norm_wouldbetest is None:
                continue
            else:
                new_time_to_norm_wouldbetest[time] = last_norm_wouldbetest
        return new_time_to_norm_wouldbetest  
    def get_timestep_to_avg(df, col='unfolded_time_to_norm_wouldbetest'):
        time_to_values = defaultdict(list)
        time_to_avg = {}
        for time_to_val in df[col].values:
            for time,val in time_to_val.items():
                time_to_values[time].append(val)
        for time,values in time_to_values.items():
            assert not pd.isna(time) and not pd.isna(np.mean(values))
            time_to_avg[time] = np.mean(values)
        return time_to_avg
    
    def plot_test_progression(col='unfolded_time_to_norm_wouldbetest', groupby='domain', legend=True, ylabel='Normalized potential generalization'):
        plt.figure(figsize=(5,5))
        fontsize = 20
        for i,group in enumerate(st_fin_df[groupby].unique()):
            sub_df = st_fin_df[st_fin_df[groupby] == group]
            time_to_avg = dict(sorted(get_timestep_to_avg(sub_df, col=col).items()))
            time_to_avg = {time/(60*60):avg for time,avg in time_to_avg.items()}

            plt.plot(list(time_to_avg.keys()), list(time_to_avg.values()), label=group, color=colors_rgb[i%len(colors_rgb)], linewidth=2.5)
        if(legend):
            plt.legend(loc='lower right', fontsize=18, frameon=False)
        plt.xlabel('Time passed (h)', fontsize=fontsize)
        plt.xticks([0,4,8], fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.yticks([0,0.5,1], fontsize=fontsize)
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.savefig(f"./paper_tables/{ylabel}.svg", format="svg")
    
    max_dur = 60*60*8 # 8 hours in seconds
    st_fin_df = fin_df[~fin_df['would_be_test'].isna()] #only exps that have stealth test
    st_fin_df['time_to_norm_wouldbetest'] = st_fin_df.apply(lambda x: add_time_to_norm_wouldbetest(x, to_normalize_col='would_be_test'), axis=1)
    st_fin_df['time_to_norm_val_mainmetric'] = st_fin_df.apply(lambda x: add_time_to_norm_wouldbetest(x, to_normalize_col='best_val_metric_so_far'), axis=1)

    def check_monotone(row):
        import re as _re
        col = 'best_val_metric_so_far'
        s = row[col]
        if pd.isna(s):
            return
        s = _re.sub(r'np\.float64\(([^)]+)\)', r'\1', s)
        s = _re.sub(r'\bnan\b', 'None', s)
        vals = [v for v in ast.literal_eval(s) if v is not None]
        if len(vals) < 2:
            return
        lower_is_better = row.get('main_metric') == 'MAE'
        for i in range(1, len(vals)):
            if lower_is_better:
                assert vals[i] <= vals[i-1] + 1e-9, \
                    f"{col} not non-increasing at index {i} for run {row.get('run_name')}: {vals[i-1]} -> {vals[i]}"
            else:
                assert vals[i] >= vals[i-1] - 1e-9, \
                    f"{col} not non-decreasing at index {i} for run {row.get('run_name')}: {vals[i-1]} -> {vals[i]}"
        print('monotonicity OK')

    st_fin_df.apply(check_monotone, axis=1)


    all_timestamps_lists_test = st_fin_df.apply(lambda x: list(x['time_to_norm_wouldbetest'].keys()), axis=1).values
    all_timestamps_lists_val = st_fin_df.apply(lambda x: list(x['time_to_norm_val_mainmetric'].keys()), axis=1).values

    all_timestamps_test = [float(x) for sublist in all_timestamps_lists_test for x in sublist]
    all_timestamps_val = [float(x) for sublist in all_timestamps_lists_val for x in sublist]
    all_timestamps = all_timestamps_test + all_timestamps_val
    all_timestamps.append(max_dur)
    all_timestamps.sort()

    st_fin_df['unfolded_time_to_norm_wouldbetest'] = st_fin_df.apply(lambda x: unfold_time(x, col='time_to_norm_wouldbetest'), axis=1)
    st_fin_df['unfolded_time_to_norm_val_mainmetric'] = st_fin_df.apply(lambda x: unfold_time(x, col='time_to_norm_val_mainmetric'), axis=1)

    plot_test_progression(col='unfolded_time_to_norm_wouldbetest', groupby='domain', ylabel='Normalized test score')
    plot_test_progression(col='unfolded_time_to_norm_val_mainmetric', groupby='domain', ylabel='Normalized validation score')
    plot_test_progression(col='unfolded_time_to_norm_val_mainmetric', groupby='dataset', legend=False, ylabel='Normalized per dset validation score')
    plot_test_progression(col='unfolded_time_to_norm_val_mainmetric', groupby='run_name', legend=False, ylabel='Normalized per run validation score')


def create_corr_dist_plot(fin_df):
    st_fin_df = fin_df[~fin_df['would_be_test'].isna()] #only exps that have stealth test
    plt.plot(figsize=(5,5))
    sns.displot(st_fin_df, x="val_stealth_mainmetric_corr", color=colors_rgb[2], bins=30, stat='percent', edgecolor='none')
    plt.xlabel("Validation-Test Correlation", fontsize=20)
    plt.ylabel("Runs (%)", fontsize=20)
    plt.xticks([-0.25, 0 , 0.5,  1], fontsize=20)
    plt.yticks([0, 10, 50], fontsize=20)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig("./paper_tables/corr_distribution.svg", format="svg", bbox_inches="tight")

def main():
    create_arch_plot()
    agentomics_all_df = pd.read_csv('./paper_tables/agentomics_all.csv')
    create_time_plots(agentomics_all_df)
    create_corr_dist_plot(agentomics_all_df)

if __name__ == '__main__':
    main()