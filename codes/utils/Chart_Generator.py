import matplotlib

matplotlib.use('TkAgg')

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score


def create_publication_chart(base_dir: str = '../../data/evaluation', output_dir: str = 'charts'):
    """
    Generates a publication-quality chart using Matplotlib and saves it
    as both PDF and PNG.
    """
    # 1. --- Data Processing ---
    print("📊 Part 1: Generating aggregated publication chart...")
    print("🔍 Reading and aggregating data...")
    models = ['deepseek-r1', 'gemma3']
    categories = ['Algorithmic', 'Creative-Writing', 'Math']
    statuses = ['Consistent', 'Inconsistent (Position Bias)', 'Inconsistent (Unstable Evaluation)']
    all_results = []
    # (Data loading logic is the same as before)
    for model in models:
        for category in categories:
            category_path = Path(base_dir) / model / category
            if not category_path.is_dir(): continue
            for file_name in os.listdir(category_path):
                if file_name.endswith('.json'):
                    file_path = category_path / file_name
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for exp_value in data.get('experiments', {}).values():
                            all_results.append(
                                {"model": model, "category": category, "status": exp_value.get('status')})

    if not all_results:
        print(f"Error: No data found in {base_dir}. Please check the path.")
        return
    df = pd.DataFrame(all_results)
    df_filtered = df[df['status'].isin(statuses)]
    status_counts = df_filtered.groupby(['model', 'category', 'status']).size().unstack(fill_value=0)
    status_counts = status_counts.reindex(columns=statuses)

    # 2. --- Chart Generation (Matplotlib) ---
    print("🎨 Creating plot...")
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

    x = np.arange(len(categories))
    width = 0.25
    colors = ['#4daf4a', '#ff7f00', '#e41a1c']
    hatches = ['/', '\\', '.']

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharey=True, sharex=True)
    fig.tight_layout(pad=4.0, h_pad=3.0)

    for i, model in enumerate(models):
        ax = axes[i]
        model_data = status_counts.loc[model]
        offsets = [-width, 0, width]

        for j, status in enumerate(statuses):
            counts = model_data[status].reindex(categories).values
            rects = ax.bar(x + offsets[j], counts, width, label=status,
                           color=colors[j], hatch=hatches[j], edgecolor='black')
            ax.bar_label(rects, padding=3, fontsize=9)

        ax.set_title(f'Model: {model}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Count')
        ax.set_xticks(x, categories)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98), frameon=False)

    # 3. --- Save the Chart in Both Formats ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define file paths
    pdf_file = output_path / "EMNLP_status_comparison.pdf"
    png_file = output_path / "EMNLP_status_comparison.png"

    # Save as PDF (for the paper)
    print(f"💾 Saving high-quality PDF chart to: {pdf_file}")
    plt.savefig(pdf_file, bbox_inches='tight')

    # Save as PNG (for presentations or quick viewing)
    print(f"💾 Saving high-resolution PNG chart to: {png_file}")
    plt.savefig(png_file, bbox_inches='tight', dpi=300)  # dpi=300 for high resolution

    plt.show()


def create_literal_per_experiment_barcharts(base_dir: str = '../../data/evaluation', output_dir: str = 'charts'):
    """
    Generates a separate figure for each model. Each figure contains a grid
    of bar charts, with each chart showing the non-aggregated outcome
    (counts of 1 or 0, or more if multiple files exist) for a single experiment.
    """
    print("📊 Generating literal, per-experiment bar charts...")

    # 1. --- Data Processing ---
    print("🔍 Reading experiment-level data...")
    models = ['deepseek-r1', 'gemma3']
    categories = ['Algorithmic', 'Creative-Writing', 'Math']
    statuses = ['Consistent', 'Inconsistent (Position Bias)', 'Inconsistent (Unstable Evaluation)']
    all_results = []

    # Load data including the experiment key
    for model in models:
        for category in categories:
            category_path = Path(base_dir) / model / category
            if not category_path.is_dir(): continue
            for file_name in os.listdir(category_path):
                if file_name.endswith('.json'):
                    file_path = category_path / file_name
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for exp_key, exp_value in data.get('experiments', {}).items():
                            all_results.append({
                                "model": model,
                                "category": category,
                                "experiment": int(exp_key),
                                "status": exp_value.get('status')
                            })

    if not all_results:
        print(f"Error: No data found in {base_dir}. Please check the path.")
        return

    df = pd.DataFrame(all_results).set_index(['model', 'category', 'experiment'])
    # FIX: Sort the index to remove the PerformanceWarning and improve lookup speed.
    df.sort_index(inplace=True)

    # 2. --- Chart Generation (One Figure Per Model) ---
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

    status_colors = {
        'Consistent': '#4daf4a',
        'Inconsistent (Position Bias)': '#ff7f00',
        'Inconsistent (Unstable Evaluation)': '#e41a1c'
    }

    for model in models:
        print(f"🎨 Creating figure for model: {model}...")
        fig, axes = plt.subplots(len(categories), 4, figsize=(12, 7), sharey=True)
        fig.suptitle(f'Per-Experiment Outcomes for Model: {model}', fontsize=16, fontweight='bold')

        # Find the max count for this model to set a consistent Y-axis limit
        max_count_for_model = df.loc[model].groupby(
            ['category', 'experiment', 'status']).size().max() if model in df.index.get_level_values('model') else 1

        for i, category in enumerate(categories):
            for j in range(4):
                ax = axes[i, j]

                try:
                    # This can return a single string or a Series of strings
                    outcomes = df.loc[(model, category, j), 'status']

                    # FIX: Properly count the statuses, whether it's a single value or a Series.
                    if isinstance(outcomes, pd.Series):
                        # If multiple files exist for this experiment, count the values
                        counts = outcomes.value_counts()
                    else:
                        # If only one file exists, the count is 1 for that status
                        counts = pd.Series([1], index=[outcomes])

                    # Ensure all statuses are present in the data, filling missing ones with 0
                    counts = counts.reindex(statuses, fill_value=0)

                except KeyError:
                    # If no data exists for this combination, all counts are 0
                    counts = pd.Series([0, 0, 0], index=statuses)

                bars = ax.bar(counts.index, counts.values, color=status_colors.values())
                ax.bar_label(bars, fontsize=9)

                if i == 0:
                    ax.set_title(f'Experiment {j + 1}', fontsize=11)
                if j == 0:
                    ax.set_ylabel(category, fontsize=12, fontweight='bold', labelpad=10)

                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # Set Y-axis ticks and limit dynamically
                ax.set_ylim(0, max_count_for_model * 1.2)
                ax.set_yticks(np.arange(0, max_count_for_model + 1, 1))

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"FINAL_outcomes_{model}.pdf"
        print(f"💾 Saving final chart to: {filename}")
        plt.savefig(filename, bbox_inches='tight')
        plt.show()


def create_results_table_image(base_dir: str = '../../data/evaluation', output_dir: str = 'charts'):
    """
    Counts per-experiment outcomes and presents them in a clean, publication-quality
    table, which is then saved as an image.

    Args:
        base_dir (str): The root directory containing the evaluation data.
        output_dir (str): The directory where the output table image will be saved.
    """
    print("📊 Generating results table...")

    # 1. --- Data Processing ---
    print("🔍 Counting per-experiment outcomes...")
    models = ['deepseek-r1', 'gemma3']
    categories = ['Algorithmic', 'Creative-Writing', 'Math']
    statuses = ['Consistent', 'Inconsistent (Position Bias)', 'Inconsistent (Unstable Evaluation)']
    all_results = []

    # This data loading part is identical to the previous function
    for model in models:
        for category in categories:
            category_path = Path(base_dir) / model / category
            if not category_path.is_dir(): continue
            for file_name in os.listdir(category_path):
                if file_name.endswith('.json'):
                    file_path = category_path / file_name
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for exp_key, exp_value in data.get('experiments', {}).items():
                            all_results.append({
                                "Model": model,
                                "Category": category,
                                "Experiment": f'Exp {int(exp_key) + 1}',
                                "status": exp_value.get('status')
                            })

    if not all_results:
        print("Error: No data was found.")
        return

    df = pd.DataFrame(all_results)

    # --- Aggregate data into the final table structure ---
    # This groups by everything and counts the status occurrences
    table_df = df.groupby(['Model', 'Category', 'Experiment', 'status']).size().unstack(fill_value=0)

    # Ensure all status columns exist, even if one never occurred
    table_df = table_df.reindex(columns=statuses, fill_value=0)
    table_df.reset_index(inplace=True)

    # 2. --- Table Rendering (Matplotlib) ---
    print("🎨 Rendering table as an image...")
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figsize as needed
    ax.axis('off')  # Hide the axes

    # Create the table object
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center',
        colColours=["#f2f2f2"] * len(table_df.columns)  # Header color
    )

    # --- Style the table for a professional look ---
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)  # Adjust column width and row height

    # Style the header
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='black')

    # 3. --- Save the Figure ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_filename = output_path / "FINAL_results_table.pdf"
    png_filename = output_path / "FINAL_results_table.png"

    print(f"💾 Saving final table to: {pdf_filename} and {png_filename}")
    plt.savefig(pdf_filename, bbox_inches='tight', dpi=300)
    plt.savefig(png_filename, bbox_inches='tight', dpi=300)
    plt.show()


def load_and_prepare_data(base_dir: str):
    """Loads and merges data from both models for head-to-head comparison."""
    all_results = []
    models = ['deepseek-r1', 'gemma3']

    # The data loading logic remains the same
    for model in models:
        categories = [c for c in os.listdir((Path(base_dir) / model))]

        for category in categories:
            category_path = Path(base_dir) / model / category
            if not category_path.is_dir(): continue
            for file_name in os.listdir(category_path):
                if file_name.endswith('.json'):
                    file_path = category_path / file_name
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for exp_key, exp_value in data.get('experiments', {}).items():
                            instance_id = f"{category}-{file_name}-exp{exp_key}"
                            all_results.append({
                                "instance_id": instance_id,
                                "model": model,
                                "status": exp_value.get('status'),
                                "choice": exp_value.get('result')
                            })

    if not all_results:
        return None

    df = pd.DataFrame(all_results)
    print(df.head())
    # Find duplicated (col1, col2) combinations
    duplicates = df.groupby(['instance_id', 'model']) \
        .filter(lambda x: len(x) > 1)

    # Mark all duplicates
    dupes = df[df.duplicated(['instance_id', 'model'], keep=False)]

    # Keep only those with exactly two rows in group
    counts = dupes.groupby(['instance_id', 'model'])['instance_id'].transform('size')
    result = dupes[counts == 2]
    print(result)
    # print(duplicates)
    # --- THIS IS THE FIX ---
    # Instead of pivot, we set a multi-level index and then unstack the 'model' level.
    # This correctly handles the duplicated 'instance_id' values.
    pivot_df = df.set_index(['instance_id', 'model']).unstack('model')

    # The rest of the function remains the same
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df.rename(columns={
        'status_deepseek-r1': 'status_deepseek', 'choice_deepseek-r1': 'choice_deepseek',
        'status_gemma3': 'status_gemma', 'choice_gemma3': 'choice_gemma'
    }, inplace=True)

    return pivot_df


# The two analysis functions below remain unchanged.

# --- This function is updated to produce two charts ---
# --- This function is updated to remove emojis ---
def create_agreement_charts(data_df: pd.DataFrame, output_dir: str = '../../data/charts'):
    """
    Creates two charts: a donut chart for overall agreement and a bar chart
    breaking down the 'Agree' category.
    """
    print("📊 Generating Agreement Donut Chart and Breakdown Bar Chart...")

    consistent_df = data_df[
        (data_df['status_deepseek'] == 'Consistent') &
        (data_df['status_gemma'] == 'Consistent')
        ].copy()

    # --- FIX: Removed emojis from labels to prevent font warnings ---
    def classify_agreement(row):
        ds_choice, gemma_choice = row['choice_deepseek'], row['choice_gemma']
        if ds_choice == gemma_choice and ds_choice in ['A', 'B', 'C']:
            return 'Agree'
        elif (ds_choice == 'C' and gemma_choice in ['A', 'B']) or \
                (gemma_choice == 'C' and ds_choice in ['A', 'B']):
            return 'Partially Agree'
        else:
            return 'Disagree'

    consistent_df['agreement_type'] = consistent_df.apply(classify_agreement, axis=1)
    agreement_counts = consistent_df['agreement_type'].value_counts()

    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    # fig.suptitle('Head-to-Head Agreement Analysis (Consistent Results Only)', fontsize=18, weight='bold')

    # Plot 1: Overall Agreement Donut Chart (on axes[0])
    ax1 = axes
    wedges, _, autotexts = ax1.pie(
        agreement_counts,
        autopct=lambda pct: f"{int(round(pct / 100 * sum(agreement_counts)))}",
        startangle=90,
        pctdistance=0.85,
        colors=['#4daf4a', '#ff7f00', '#e41a1c'],
        wedgeprops=dict(width=0.4, edgecolor='w')
    )
    ax1.legend(wedges, agreement_counts.index, title="Overall Agreement", loc="upper left")
    plt.setp(autotexts, size=12, weight="bold", color="white")
    # ax1.set_title('Agreement Distribution', fontsize=14)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "agreement_chart1.png"
    print(f"💾 Saving combined charts to: {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

    # # Plot 2: Breakdown of 'Agree' Category (on axes[1])
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    ax2 = axes
    agree_df = consistent_df[consistent_df['agreement_type'] == 'Agree']
    agree_breakdown_counts = agree_df['choice_deepseek'].value_counts().sort_index()

    bars = ax2.bar(
        agree_breakdown_counts.index,
        agree_breakdown_counts.values,
        color=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    ax2.bar_label(bars, fontsize=12)
    # ax2.set_title("Breakdown of 'Agree' Outcomes", fontsize=14)
    ax2.set_ylabel("Number of Agreements")
    ax2.set_xlabel("Agreed-Upon Choice")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "agreement_chart2.png"
    print(f"💾 Saving combined charts to: {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


# --- NEW FUNCTION TO ANALYZE 'DISAGREE' OUTCOMES ---
def create_disagreement_breakdown_chart(data_df: pd.DataFrame, output_dir: str = '../../data/charts'):
    """
    Analyzes the 'Disagree' category to show the direction of disagreement
    and visualizes it as a bar chart.
    """
    print("\n📊 Generating Disagreement Breakdown Bar Chart...")

    consistent_df = data_df[
        (data_df['status_deepseek'] == 'Consistent') & (data_df['status_gemma'] == 'Consistent')].copy()

    # Filter for only the 'Disagree' cases
    disagree_df = consistent_df[
        (consistent_df['choice_deepseek'] != consistent_df['choice_gemma']) &
        (consistent_df['choice_deepseek'] != 'C') &
        (consistent_df['choice_gemma'] != 'C')
        ].copy()

    # Classify the direction of disagreement
    def classify_disagreement_direction(row):
        if row['choice_deepseek'] == 'A' and row['choice_gemma'] == 'B':
            return 'DeepSeek (A) vs Gemma (B)'
        elif row['choice_deepseek'] == 'B' and row['choice_gemma'] == 'A':
            return 'DeepSeek (B) vs Gemma (A)'
        return 'Other'

    disagree_df['direction'] = disagree_df.apply(classify_disagreement_direction, axis=1)
    disagreement_counts = disagree_df['direction'].value_counts()

    if disagreement_counts.empty:
        print("No 'Disagree' cases found to analyze.")
        return

    # --- Visualization ---
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(6, 6))

    bars = plt.bar(disagreement_counts.index, disagreement_counts.values, color=['#1f77b4', '#ff7f0e'])
    plt.bar_label(bars, fontsize=12)

    # plt.title("Breakdown of 'Disagree' Outcomes", fontsize=16, weight='bold')
    plt.ylabel("Number of Disagreements")
    plt.xlabel("Direction of Disagreement")
    plt.xticks(rotation=10)  # Rotate labels slightly for better fit
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Save the Figure ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "disagreement_breakdown_chart.png"
    print(f"💾 Saving disagreement breakdown chart to: {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


def create_agreement_matrix_heatmap(data_df: pd.DataFrame, output_dir: str = '../../data/charts'):
    """
    Creates a heatmap (agreement matrix) showing every combination of outcomes
    from both models, including inconsistent results.
    """
    print("📊 Generating Full Agreement Matrix Heatmap...")

    df = data_df.copy()
    for model_name in ['deepseek', 'gemma']:
        df[f'outcome_{model_name}'] = np.where(
            df[f'status_{model_name}'] == 'Consistent',
            df[f'choice_{model_name}'],
            'Inconsistent'
        )

    contingency_table = pd.crosstab(df['outcome_deepseek'], df['outcome_gemma'])

    all_outcomes = ['A', 'B', 'C', 'Inconsistent']
    contingency_table = contingency_table.reindex(index=all_outcomes, columns=all_outcomes, fill_value=0)

    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(6, 6))

    sns.heatmap(
        contingency_table,
        annot=True, fmt='d', cmap='cividis',
        linewidths=.5, annot_kws={"size": 12},
        cbar=False
    )

    # plt.title('Full Agreement Matrix (Including Inconsistencies)', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Gemma3 Outcome', fontsize=12)
    plt.ylabel('DeepSeek-R1 Outcome', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / "agreement_matrix_heatmap.png"
    print(f"💾 Saving heatmap to: {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


def analyze_linguistic_consistency(base_dir: str, output_dir: str = '../../data/charts'):
    """
    Compares Experiment 0 (Persian) vs. Experiment 2 (English) for each
    model and category to analyze linguistic consistency.

    Generates a separate 2x2 composite figure for each category.
    """
    print("## Starting Linguistic Consistency Analysis (Exp 0 vs. Exp 2) ##")

    # 1. --- Load and Prepare Data ---
    all_results = []
    models = ['deepseek-r1', 'gemma3']
    categories = ['Algorithmic', 'Creative-Writing', 'Math']

    for model in models:
        for category in categories:
            category_path = Path(base_dir) / model / category
            if not category_path.is_dir(): continue
            for file_name in os.listdir(category_path):
                if file_name.endswith('.json'):
                    with open(category_path / file_name, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # We only care about experiments 0 and 2
                        for exp_key in ['0', '2']:
                            exp_value = data.get('experiments', {}).get(exp_key)
                            if exp_value:
                                all_results.append({
                                    "model": model, "category": category,
                                    "question_file": file_name,
                                    "experiment": f"exp{exp_key}",
                                    "outcome": exp_value.get('result', 'Inconsistent') if exp_value.get(
                                        'status') == 'Consistent' else 'Inconsistent'
                                })

    if not all_results:
        print("Could not find data for experiments 0 or 2.")
        return

    # Pivot the data to get exp0 and exp2 outcomes side-by-side for each question
    df = pd.DataFrame(all_results)
    pivot_df = df.pivot_table(
        index=['model', 'category', 'question_file'],
        columns='experiment',
        values='outcome',
        aggfunc='first'
    ).reset_index()
    pivot_df.dropna(subset=['exp0', 'exp2'], inplace=True)  # Ensure we have pairs to compare

    # 2. --- Main plotting loop: One figure per category ---
    for category in categories:
        print(f"\nProcessing and plotting for Category: {category}...")

        # Create a new 2x2 figure for this category
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        # fig.suptitle(f'Linguistic Consistency Analysis: {category} (Persian vs. English)', fontsize=16, weight='bold')

        for i, model in enumerate(models):
            # Filter for the current model and category
            model_cat_df = pivot_df[(pivot_df['model'] == model) & (pivot_df['category'] == category)].copy()

            if model_cat_df.empty:
                continue

            # --- Plot 1: Stability Bar Chart (Top Row) ---
            ax_bar = axes[0, i]

            def classify_stability(row):
                is_c0 = row['exp0'] != 'Inconsistent'
                is_c2 = row['exp2'] != 'Inconsistent'
                if is_c0 and is_c2: return 'Stable Consistent'
                if not is_c0 and not is_c2: return 'Stable Inconsistent'
                if is_c0 and not is_c2: return 'Became Inconsistent'
                if not is_c0 and is_c2: return 'Became Consistent'

            model_cat_df['stability'] = model_cat_df.apply(classify_stability, axis=1)
            stability_counts = model_cat_df['stability'].value_counts()
            plot_order = ['Stable Consistent', 'Stable Inconsistent', 'Became Consistent', 'Became Inconsistent']
            stability_counts = stability_counts.reindex(plot_order, fill_value=0)

            bars = ax_bar.bar(stability_counts.index, stability_counts.values)
            ax_bar.bar_label(bars, fontsize=9)
            ax_bar.set_title(f'Model: {model}', fontsize=14)
            ax_bar.tick_params(axis='x', rotation=45, labelsize=9)
            if i == 0: ax_bar.set_ylabel("Count of Questions")

            # --- Plot 2: Detailed Transition Matrix (Bottom Row) ---
            ax_heatmap = axes[1, i]

            transition_matrix = pd.crosstab(model_cat_df['exp0'], model_cat_df['exp2'])
            all_outcomes = ['A', 'B', 'C', 'Inconsistent']
            transition_matrix = transition_matrix.reindex(index=all_outcomes, columns=all_outcomes, fill_value=0)

            sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_heatmap, cbar=False,
                        annot_kws={"size": 10})
            ax_heatmap.set_title("Detailed Transition Matrix", fontsize=12)
            ax_heatmap.set_ylabel("Outcome in Persian (Exp 0)")
            ax_heatmap.set_xlabel("Outcome in English (Exp 2)")

        # Display the figure for the current category
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"linguistic_{category}.png"
        print(f"💾 Saving heatmap to: {filename}")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()

        plt.show()


# def analyze_model_human_agreement_per_file(base_dir: str, human_labels_dir: str, output_dir: str = '../../data/charts'):
#     """
#     For each labeled file, creates a composite 2x2 figure containing:
#     - Top row: Bar charts of model-human agreement types.
#     - Bottom row: Confusion matrices (heatmaps) for detailed analysis.
#     """
#     print("## Starting Detailed Model vs. Human Analysis (Per-File Basis) ##")
#
#     # 1. --- Load all model data ---
#     all_model_results = []
#     models = ['deepseek-r1', 'gemma3']
#     for model in models:
#         category_dirs = [d for d in (Path(base_dir) / model).glob('*') if d.is_dir()]
#         for category_path in category_dirs:
#             for file_name in os.listdir(category_path):
#                 if file_name.endswith('.json'):
#                     with open(category_path / file_name, 'r', encoding='utf-8') as f:
#                         data = json.load(f)
#                         for exp_key, exp_value in data.get('experiments', {}).items():
#                             all_model_results.append({
#                                 "model": model, "category": category_path.name,
#                                 "experiment": int(exp_key), "question_file": file_name,
#                                 "status": exp_value.get('status'), "model_choice": exp_value.get('result')
#                             })
#     model_df = pd.DataFrame(all_model_results)
#
#     # 2. --- Iterate through each potential group and generate a plot if labels exist ---
#     unique_groups = model_df[['category', 'experiment']].drop_duplicates().to_records(index=False)
#     files_found = 0
#     label_files = defaultdict(list)
#     for file_path in Path(human_labels_dir).glob('*.txt'):
#         print(file_path)
#         parts = file_path.stem.split('_')
#         category, experiment = parts[2], int(parts[3].replace('exp', ''))
#         label_files[(category, experiment)].append(file_path)
#
#     for category, experiment in unique_groups:
#         if category == 'Creative-Writing' and experiment > 2:
#             continue
#         tmp_exp = experiment
#         if experiment == 2:
#             tmp_exp = 0
#         label_filepath = label_files[(category, tmp_exp)][0]
#         label_filename = os.path.basename(label_filepath)
#         print(label_filename)
#         # (category, experiment), files in label_files.items()
#         # label_filename = f"human_labels_{category}_exp{experiment}.txt"
#         # label_filepath = Path(human_labels_dir) / label_filename
#         print(label_filepath)
#         if not label_filepath.exists():
#             continue
#
#         files_found += 1
#         print(f"\nProcessing and plotting for: {label_filename}...")
#
#         with open(label_filepath, 'r') as f:
#             human_labels = [line.split('-')[1].strip() for line in f if '-' in line]
#
#         group_model_df = model_df[
#             (model_df['category'] == category) & (model_df['experiment'] == experiment)].sort_values(
#             by='question_file').reset_index(drop=True)
#
#         if len(group_model_df) / len(models) != len(human_labels):
#             print(f"Warning: Mismatch between questions and labels for {label_filename}. Skipping.")
#             continue
#
#         repeated_labels = np.tile(human_labels, len(models))
#         group_model_df['human_label'] = repeated_labels
#
#         def classify(row):
#             if row['status'] != 'Consistent': return 'Model Inconsistent'
#             if row['model_choice'] == row['human_label']: return 'Full Agreement'
#             if row['model_choice'] == 'C':
#                 return 'Model Undecided'
#             else:
#                 return 'Disagreement'
#
#         group_model_df['agreement_type'] = group_model_df.apply(classify, axis=1)
#
#         # --- Create a new 2x2 figure for this specific file ---
#         plt.style.use('seaborn-v0_8-paper')
#         fig, axes = plt.subplots(2, 2, figsize=(10, 7))  # 2 rows, 2 columns
#         fig.suptitle(f'Model vs. Human Analysis: {category} - Experiment {experiment}', fontsize=16, weight='bold')
#
#         for i, model in enumerate(models):
#             model_data = group_model_df[group_model_df['model'] == model]
#
#             # --- Plot 1: Bar Chart (Top Row) ---
#             ax_bar = axes[0, i]
#             agreement_counts = model_data['agreement_type'].value_counts()
#             plot_order = ['Full Agreement', 'Disagreement', 'Model Undecided', 'Model Inconsistent']
#             agreement_counts = agreement_counts.reindex(plot_order, fill_value=0)
#
#             bars = ax_bar.bar(agreement_counts.index, agreement_counts.values,
#                               color=['#4daf4a', '#e41a1c', '#ff7f00', '#999999'])
#             ax_bar.bar_label(bars, fontsize=9)
#             ax_bar.set_title(f'Model: {model}', fontsize=14)
#             ax_bar.tick_params(axis='x', rotation=45, labelsize=9)
#             ax_bar.spines['top'].set_visible(False)
#             ax_bar.spines['right'].set_visible(False)
#             if i == 0: ax_bar.set_ylabel("Count of Outcomes")
#
#             # --- Plot 2: Confusion Matrix (Bottom Row) ---
#             ax_heatmap = axes[1, i]
#             # Create the confusion matrix data
#             conf_matrix = pd.crosstab(model_data['human_label'], model_data['model_choice'])
#             # Ensure all possible outcomes are present for a consistent matrix layout
#             all_outcomes = ['A', 'B', 'C']
#             model_outcomes = ['A', 'B', 'C', 'Inconsistent'] if 'Inconsistent' in model_data[
#                 'model_choice'].unique() else ['A', 'B', 'C']
#             conf_matrix = conf_matrix.reindex(index=all_outcomes, columns=model_outcomes, fill_value=0)
#
#             sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', ax=ax_heatmap, cbar=False,
#                         annot_kws={"size": 10})
#             ax_heatmap.set_title("Agreement Matrix", fontsize=12)
#             ax_heatmap.set_ylabel("Human Label")
#             ax_heatmap.set_xlabel("Model Choice")
#             ax_heatmap.tick_params(axis='x', rotation=0)
#             ax_heatmap.tick_params(axis='y', rotation=0)
#
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
#         filename = output_path / f"gold_{category}_exp{experiment}.png"
#         print(f"💾 Saving disagreement breakdown chart to: {filename}")
#         plt.savefig(filename, bbox_inches='tight', dpi=300)
#         # plt.show()
#
#     if files_found == 0:
#         print("\nAnalysis complete. No matching human label files were found.")


def analyze_model_human_agreement_per_file(base_dir: str, human_labels_dir: str, output_dir: str = '../../data/charts'):
    """
    Compares model responses to human labels for each valid category-experiment pair,
    implementing special rules for Creative-Writing exp3 and the exp0/exp2 label mapping.
    """
    print("## Starting Detailed Model vs. Human Analysis (Corrected) ##")

    # 1. --- Load all model data ---
    all_model_results = []
    models = ['deepseek-r1', 'gemma3']
    for model in models:
        category_dirs = [d for d in (Path(base_dir) / model).glob('*') if d.is_dir()]
        for category_path in category_dirs:
            for file_name in os.listdir(category_path):
                if file_name.endswith('.json'):
                    with open(category_path / file_name, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for exp_key, exp_value in data.get('experiments', {}).items():
                            all_model_results.append({
                                "model": model, "category": category_path.name,
                                "experiment": int(exp_key), "question_file": file_name,
                                "status": exp_value.get('status'), "model_choice": exp_value.get('result')
                            })
    model_df = pd.DataFrame(all_model_results)

    # 2. --- Pre-load all available human label file paths ---
    label_files = defaultdict(list)
    for file_path in Path(human_labels_dir).glob('*.txt'):
        parts = file_path.stem.split('_')
        category, experiment = parts[2], int(parts[3].replace('exp', ''))
        label_files[(category, experiment)].append(file_path)

    # 3. --- Iterate through each model data group and perform analysis ---
    unique_groups = model_df[['category', 'experiment']].drop_duplicates().to_records(index=False)

    for category, experiment in unique_groups:
        # RULE 1: Ignore Creative-Writing experiment 3
        if category == 'Creative-Writing' and experiment == 3:
            continue

        # RULE 2: Map model experiment 2 to human labels from experiment 0
        label_exp_key = 0 if experiment == 2 else experiment

        # Safely check if the required label file exists
        if (category, label_exp_key) not in label_files:
            continue

        label_filepath = label_files[(category, label_exp_key)][0]
        print(f"\nProcessing {category}-exp{experiment} using labels from {label_filepath.name}...")

        # Load human labels
        with open(label_filepath, 'r') as f:
            human_labels = [line.split('-')[1].strip() for line in f if '-' in line]

        # Filter model data for the current group
        group_model_df = model_df[(model_df['category'] == category) & (model_df['experiment'] == experiment)]

        # --- ROBUST MERGE LOGIC (Fixes sparse matrix bug) ---
        sorted_filenames = sorted(group_model_df['question_file'].unique())

        if len(sorted_filenames) != len(human_labels):
            print(
                f"  Warning: Mismatch between question files ({len(sorted_filenames)}) and labels ({len(human_labels)}). Skipping.")
            continue

        question_map = pd.DataFrame({'question_file': sorted_filenames, 'question_index': range(len(sorted_filenames))})
        human_labels_df = pd.DataFrame({'human_label': human_labels, 'question_index': range(len(human_labels))})

        merged_df = pd.merge(group_model_df, question_map, on='question_file')
        merged_df = pd.merge(merged_df, human_labels_df, on='question_index')

        # --- Classification and Plotting (Your logic, now on correct data) ---
        def classify(row):
            if row['status'] != 'Consistent': return 'Model Inconsistent'
            if row['model_choice'] == row['human_label']: return 'Full Agreement'
            if row['model_choice'] == 'C':
                return 'Model Undecided'
            else:
                return 'Disagreement'

        merged_df['agreement_type'] = merged_df.apply(classify, axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Model vs. Human Analysis: {category} - Experiment {experiment}', fontsize=16, weight='bold')

        for i, model in enumerate(models):
            model_data = merged_df[merged_df['model'] == model]
            ax_bar = axes[0, i]
            agreement_counts = model_data['agreement_type'].value_counts().reindex(
                ['Full Agreement', 'Disagreement', 'Model Undecided', 'Model Inconsistent'], fill_value=0)
            bars = ax_bar.bar(agreement_counts.index, agreement_counts.values,
                              color=['#4daf4a', '#e41a1c', '#ff7f00', '#999999'])
            ax_bar.bar_label(bars, fontsize=9)
            ax_bar.set_title(f'Model: {model}', fontsize=14)
            ax_bar.tick_params(axis='x', rotation=45, labelsize=9)
            if i == 0: ax_bar.set_ylabel("Count of Outcomes")

            ax_heatmap = axes[1, i]
            conf_matrix = pd.crosstab(model_data['human_label'], model_data['model_choice'])
            all_outcomes = ['A', 'B', 'C']
            model_outcomes = ['A', 'B', 'C', 'Inconsistent'] if 'Inconsistent' in model_data[
                'model_choice'].unique() else ['A', 'B', 'C']
            conf_matrix = conf_matrix.reindex(index=all_outcomes, columns=model_outcomes, fill_value=0)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', ax=ax_heatmap, cbar=False,
                        annot_kws={"size": 10})
            ax_heatmap.set_ylabel("Human Label")
            ax_heatmap.set_xlabel("Model Choice")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"gold_comparison_{category}_exp{experiment}.png"
        print(f"  -> Saving analysis to: {filename}")
        plt.savefig(filename, dpi=300)
        plt.close(fig)


def analyze_model_and_annotator_agreement(base_dir: str, human_labels_dir: str, output_dir: str = '../../data/charts'):
    """
    Correctly analyzes inter-annotator agreement and model performance against human labels.
    This version uses a robust merging strategy and handles single-annotator files.
    """
    print("## Starting Full Annotator and Model Agreement Analysis (Corrected) ##")

    # 1. --- Load all model data first ---
    all_model_results = []
    models = ['deepseek-r1', 'gemma3']
    for model in models:
        category_dirs = [d for d in (Path(base_dir) / model).glob('*') if d.is_dir()]
        for category_path in category_dirs:
            for file_name in os.listdir(category_path):
                if file_name.endswith('.json'):
                    with open(category_path / file_name, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for exp_key, exp_value in data.get('experiments', {}).items():
                            all_model_results.append({
                                "model": model, "category": category_path.name,
                                "experiment": int(exp_key), "question_file": file_name,
                                "model_outcome": exp_value.get('result', 'Inconsistent') if exp_value.get(
                                    'status') == 'Consistent' else 'Inconsistent'
                            })
    model_df = pd.DataFrame(all_model_results)

    # 2. --- Group human label files by category and experiment ---
    label_files = defaultdict(list)
    for file_path in Path(human_labels_dir).glob('*.txt'):
        parts = file_path.stem.split('_')
        category, experiment = parts[2], int(parts[3].replace('exp', ''))
        label_files[(category, experiment)].append(file_path)

    # 3. --- Main Loop: Analyze and Plot for each group ---
    for (category, experiment), files in label_files.items():
        print(f"\nProcessing {category}-exp{experiment} with {len(files)} annotator(s)...")

        # Load all annotators for the group into a single DataFrame
        annotator_data = {f.stem.split('-')[-1]: [line.split('-')[1].strip() for line in open(f, 'r') if '-' in line]
                          for f in files}
        annotator_df = pd.DataFrame(annotator_data)
        annotator_names = annotator_df.columns.tolist()

        # Filter model data for the current group
        group_model_df = model_df[(model_df['category'] == category) & (model_df['experiment'] == experiment)]

        # --- ROBUST MERGE LOGIC ---
        # Create a definitive order based on sorted question filenames
        sorted_filenames = sorted(group_model_df['question_file'].unique())

        if len(sorted_filenames) != len(annotator_df):
            print(
                f"Warning: Mismatch between number of question files ({len(sorted_filenames)}) and labels ({len(annotator_df)}). Skipping.")
            continue

        # Create a temporary mapping DataFrame to ensure correct joins
        question_map = pd.DataFrame({
            'question_file': sorted_filenames,
            'question_index': range(len(sorted_filenames))
        })
        annotator_df['question_index'] = annotator_df.index

        # Merge map with model data and then with annotator data
        merged_df = pd.merge(group_model_df, question_map, on='question_file')
        merged_df = pd.merge(merged_df, annotator_df, on='question_index')

        # --- Dynamic Figure Layout based on number of annotators ---
        num_annotators = len(annotator_names)
        num_subplots = 3 if num_annotators >= 2 else 2
        figsize = (18, 5) if num_annotators >= 2 else (12, 5)

        plt.style.use('seaborn-v0_8-paper')
        fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
        fig.suptitle(f'Analysis for: {category} - Experiment {experiment}', fontsize=16, weight='bold')

        # --- Plotting Panels ---
        ax_offset = 0
        if num_annotators >= 2:
            # Panel 1: Inter-Annotator Agreement (IAA)
            ax_offset = 1
            ax_iaa = axes[0]
            labels1, labels2 = annotator_df[annotator_names[0]], annotator_df[annotator_names[1]]
            kappa = cohen_kappa_score(labels1, labels2)
            iaa_matrix = pd.crosstab(labels1, labels2).reindex(index=['A', 'B', 'C'], columns=['A', 'B', 'C'],
                                                               fill_value=0)

            sns.heatmap(iaa_matrix, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax_iaa)
            ax_iaa.set_title(f'Inter-Annotator Agreement\n(Cohen\'s Kappa: {kappa:.2f})')
            ax_iaa.set_xlabel(f'Annotator: {annotator_names[1]}')
            ax_iaa.set_ylabel(f'Annotator: {annotator_names[0]}')

            # Define gold standard from consensus
            merged_df['gold_label'] = np.where(labels1 == labels2, labels1, np.nan)
            comparison_df = merged_df.dropna(subset=['gold_label'])
            gold_label_col = 'gold_label'
            human_label_source = "Human Consensus"
        else:
            # For single annotator, the "gold label" is just their labels
            comparison_df = merged_df
            gold_label_col = annotator_names[0]
            human_label_source = f"Human Label ({annotator_names[0]})"

        # Model vs. Gold Standard Panels
        for i, model in enumerate(models):
            ax = axes[i + ax_offset]
            model_vs_gold_df = comparison_df[comparison_df['model'] == model]

            model_matrix = pd.crosstab(model_vs_gold_df[gold_label_col], model_vs_gold_df['model_outcome'])
            human_outcomes = sorted(model_vs_gold_df[gold_label_col].unique())
            model_outcomes = ['A', 'B', 'C', 'Inconsistent']
            model_matrix = model_matrix.reindex(index=human_outcomes, columns=model_outcomes, fill_value=0)

            # Calculate accuracy on A, B, C choices
            valid_labels = [l for l in ['A', 'B', 'C'] if l in model_matrix.index]
            correct_preds = np.diag(model_matrix.loc[valid_labels, valid_labels]).sum()
            total_preds = model_matrix.loc[valid_labels, :].values.sum()
            accuracy = correct_preds / total_preds if total_preds > 0 else 0

            sns.heatmap(model_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'{model} vs. Human\n(Accuracy: {accuracy:.2f})')
            ax.set_xlabel('Model Outcome')
            ax.set_ylabel(human_label_source)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"full_analysis_{category}_exp{experiment}.png"
        print(f"💾 Saving composite analysis chart to: {filename}")
        plt.savefig(filename, dpi=300)
        plt.show()


def visualize_inter_annotator_agreement(human_labels_dir: str, output_dir: str = '../../data/charts/IAA'):
    """
    Calculates and visualizes Inter-Annotator Agreement (IAA) for pairs
    of annotators, saving each comparison as a separate heatmap chart.

    Args:
        human_labels_dir (str): The directory containing human label .txt files.
        output_dir (str): The directory where the output charts will be saved.
    """
    print("## Visualizing Inter-Annotator Agreement (IAA) for Paired Annotators ##")

    # 1. --- Group human label files by category and experiment ---
    label_files = defaultdict(list)
    for file_path in Path(human_labels_dir).glob('*.txt'):
        try:
            parts = file_path.stem.split('_')
            category = parts[2]
            experiment = int(parts[3].replace('exp', ''))
            label_files[(category, experiment)].append(file_path)
        except (IndexError, ValueError):
            print(f"Warning: Skipping malformed filename: {file_path.name}")
            continue

    # 2. --- Loop through groups, calculate metrics, and plot for pairs ---
    files_processed = 0
    for (category, experiment), files in label_files.items():
        if len(files) != 2:
            continue

        files_processed += 1
        annotator_names = [f.stem.split('-')[-1] for f in files]
        print(f"Processing and plotting for {category}-exp{experiment}...")

        labels1 = [line.split('-')[1].strip() for line in open(files[0], 'r') if '-' in line]
        labels2 = [line.split('-')[1].strip() for line in open(files[1], 'r') if '-' in line]

        if len(labels1) != len(labels2) or not labels1:
            print(f"  Warning: Label mismatch or empty file for {category}-exp{experiment}. Skipping.")
            continue

        # Calculate metrics
        agreement_count = sum(1 for i in range(len(labels1)) if labels1[i] == labels2[i])
        percent_agreement = (agreement_count / len(labels1)) * 100
        kappa = cohen_kappa_score(labels1, labels2)

        # Create the confusion matrix
        iaa_matrix = pd.crosstab(pd.Series(labels1, name=annotator_names[0]),
                                 pd.Series(labels2, name=annotator_names[1]))
        all_outcomes = ['A', 'B', 'C']
        iaa_matrix = iaa_matrix.reindex(index=all_outcomes, columns=all_outcomes, fill_value=0)

        # --- Visualization ---
        plt.style.use('seaborn-v0_8-paper')
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(iaa_matrix, annot=True, fmt='d', cmap='Greens', cbar=True, ax=ax,
                    annot_kws={"size": 12}, linewidths=.5)

        title = (
            f'Inter-Annotator Agreement: {category} - Experiment {experiment}\n'
            f'Agreement: {percent_agreement:.1f}%  |  Cohen\'s Kappa (κ): {kappa:.3f}'
        )
        ax.set_title(title, fontsize=14, weight='bold')
        plt.tight_layout()

        # --- Save the Figure ---
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"IAA_{category}_exp{experiment}.png"

        plt.savefig(filename, dpi=300)
        plt.close(fig)  # Close the figure to avoid displaying it

    if files_processed == 0:
        print("\nNo pairs with exactly two annotators were found to visualize.")
    else:
        print(f"\nProcessing complete. Saved {files_processed} charts to '{output_dir}'.")


### How to Run


if __name__ == '__main__':
    data_directory = '../../data/evaluation'
    # create_publication_chart(base_dir=data_directory)
    # create_literal_per_experiment_barcharts(base_dir=data_directory)
    prepared_df = load_and_prepare_data(base_dir=data_directory)
    # create_agreement_charts(prepared_df)
    # create_disagreement_breakdown_chart(prepared_df)
    # create_agreement_matrix_heatmap(prepared_df)
    # analyze_linguistic_consistency(base_dir=data_directory)

    human_labels_directory = '../../data/human_labels/g'
    analyze_model_human_agreement_per_file(
        base_dir=data_directory,
        human_labels_dir=human_labels_directory
    )
    # analyze_model_and_annotator_agreement(
    #     base_dir=data_directory,
    #     human_labels_dir=human_labels_directory,
    #     output_dir='../../data/charts'
    # )
    # visualize_inter_annotator_agreement(
    #     human_labels_dir=human_labels_directory,
    #     output_dir='../../data/charts/IAA'  # A dedicated subdirectory for these charts
    # )

    # prepared_df = load_and_prepare_data(base_dir=data_directory)
    #
    # if prepared_df is not None:
    #     create_agreement_charts(prepared_df)
    #     # Generates the new, separate 'Disagree' breakdown bar chart
    #     create_disagreement_breakdown_chart(prepared_df)
    #
    #     create_agreement_matrix_heatmap(prepared_df)
    # else:
    #     print("Data loading failed. Please check the directory path.")

    # human_labels_directory = '../../data/human_labels'  # Assumed path
    #
    # if not Path(human_labels_directory).exists():
    #     print(f"Error: Human labels directory not found at '{human_labels_directory}'")
    # else:
    #     analyze_model_human_agreement_per_file(
    #         base_dir=data_directory,
    #         human_labels_dir=human_labels_directory
    #     )
