import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def generate_latex_report(csv_path = "./tests_results.csv", output_dir="."):
    # Load the CSV
    df = pd.read_csv(csv_path)
    df['experiment_id'] = df['experiment_id'].astype(int)
    
    # Define sections
    sections = [
        {
            "title": "Baseline + ROC Curve",
            "ids": list(range(1, 9)),
            "description": "Initial baseline architecture and ROC evaluation."
        },
        {
            "title": "Spectrogram Augmentations",
            "ids": list(range(10, 18)),
            "description": "Time mask, shift, Gaussian noise, and frequency masking."
        },
        {
            "title": "VLTP Enabled",
            "ids": list(range(19, 20)),
            "description": "Training with VLTP preprocessing."
        },
        {
            "title": "VLTP Removed (BS=32)",
            "ids": list(range(29, 30)),
            "description": "VLTP removed, batch size increased for validation and testing."
        },
        {
            "title": "Random Sampler",
            "ids": list(range(23, 24)),
            "description": "Random sampler added for improved class balancing."
        },
        {
            "title": "Group Normalization",
            "ids": list(range(25, 30)),
            "description": "Architectural variant using GroupNorm."
        },
        {
            "title": "Pooling Config (1×2)",
            "ids": list(range(31, 34)),
            "description": "All 2D poolings changed to (1×2) plus avg pooling."
        },
        {
            "title": "Pooling Config (1×1)",
            "ids": list(range(35, 37)),
            "description": "Returned avg pooling back to (1×1)."
        }
    ]

    metrics = ['test_f1_macro', 'val_f1_macro', 'train_f1_macro']
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    latex_content = [
        "\\documentclass{article}",
        
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{booktabs}",
        "\\usepackage{geometry}",
        "\\usepackage{graphicx}",
        "\\geometry{a4paper, margin=1in}",
        "\\title{Comprehensive Model Architecture and Training Report}",
        "\\author{Automated Analysis System}",
        "\\date{\\today}",
        "\\begin{document}",
        "\\maketitle",
        ""
    ]

    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    for sec in sections:
        filtered_df = df[df['experiment_id'].isin(sec['ids'])].copy()
        if filtered_df.empty:
            continue

        filtered_df['notes'] = filtered_df['notes'].fillna('ResNet 18 Base' if 45 in sec['ids'] else 'Baseline')
        report_data = []

        stats = {"name": sec["title"]}
        for metric in metrics:
            stats[f"{metric}_mean"] = filtered_df[metric].mean()
            stats[f"{metric}_std"] = filtered_df[metric].std()

        report_data.append(stats)

        # Boxplot data: one distribution only
        box_values = filtered_df["test_f1_macro"].dropna().values

        # Generate Boxplot
        plt.figure(figsize=(12, 6))
        plt.boxplot([box_values], tick_labels=["All Experiments"])
        plt.title(f"Test F1 Macro – {sec['title']}")
        plt.ylabel("F1 Macro")
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        safe_title = re.sub(r'[^a-zA-Z0-9_]', '', sec['title'].lower().replace(' ', '_'))
        plot_filename = f"boxplot_{safe_title}.png"
        plt.savefig(os.path.join(output_dir, 'plots', plot_filename))
        plt.close()

        # LaTeX Section
        latex_content.append(f"\\section{{{sec['title']}}}")
        latex_content.append(sec['description'])
        latex_content.append("")

        # Table
        latex_content.extend([
            "\\begin{table}[h!]",
            "\\centering",
            f"\\caption{{Summary table for {sec['title']}}}",
            "\\resizebox{\\textwidth}{!}{",
            "\\begin{tabular}{|l|c|c|c|}",
            "\\hline",
            "\\textbf{Experiment (Notes)} & \\textbf{Test F1 Macro} & \\textbf{Val F1 Macro} & \\textbf{Train F1 Macro} \\\\",
            "\\hline"
        ])

        def fmt(mean, std):
            if pd.isna(mean): return "N/A"
            if pd.isna(std) or std == 0: return f"{mean:.4f}"
            return f"{mean:.4f} $\\pm$ {std:.4f}"

        row = report_data[0]
        line = f"{row['name']} & "
        line += f"{fmt(row['test_f1_macro_mean'], row['test_f1_macro_std'])} & "
        line += f"{fmt(row['val_f1_macro_mean'], row['val_f1_macro_std'])} & "
        line += f"{fmt(row['train_f1_macro_mean'], row['train_f1_macro_std'])} \\\\ \\hline"
        latex_content.append(line)

        latex_content.extend([
            "\\end{tabular}",
            "}",
            "\\end{table}",
            ""
        ])

        # Figure
        latex_content.extend([
            "\\begin{figure}[h!]",
            "\\centering",
            f"\\includegraphics[width=0.8\\textwidth]{{plots/{plot_filename}}}",
            f"\\caption{{Distribution of Test F1 Macro for {sec['title']}}}",
            "\\end{figure}",
            "\\clearpage",
            ""
        ])

    latex_content.append("\\end{document}")
    
    tex_path = os.path.join(output_dir, 'report.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_content))
    
    print(f"Report generated at {tex_path}")

if __name__ == "__main__":
    csv_file = r'C:\Users\Hoang\Desktop\Uni\Semester 5\ML\Project\results-single-val\tests_results.csv'
    output_dir = r'C:\Users\Hoang\Desktop\Uni\Semester 5\ML\reports2'
    generate_latex_report(csv_file, output_dir)
