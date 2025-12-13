import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from matplotlib.lines import Line2D
from collections import defaultdict

def process_jbi_case_report(df: pd.DataFrame) -> pd.DataFrame:
    if "Author,Year" not in df.columns:
        if "Author, Year" in df.columns:
            df = df.rename(columns={"Author, Year": "Author,Year"})
        elif "Author" in df.columns and "Year" in df.columns:
            df["Author,Year"] = df["Author"].astype(str) + " " + df["Year"].astype(str)
        else:
            raise ValueError("Missing required columns: 'Author,Year' or 'Author' + 'Year'")

    required_columns = [
        "Author,Year",
        "Demographics", "History", "ClinicalCondition", "Diagnostics",
        "Intervention", "PostCondition", "AdverseEvents", "Lessons",
        "Total", "Overall RoB"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "Demographics", "History", "ClinicalCondition", "Diagnostics",
        "Intervention", "PostCondition", "AdverseEvents", "Lessons"
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric (0 or 1).")
        if df[col].min() < 0 or df[col].max() > 1:
            raise ValueError(f"Column {col} contains invalid values (0 or 1 allowed).")

    df["ComputedTotal"] = df[numeric_cols].sum(axis=1)
    mismatches = df[df["ComputedTotal"] != df["Total"]]
    if not mismatches.empty:
        print("⚠️ Warning: Total Score mismatches detected:")
        print(mismatches[["Author,Year", "Total", "ComputedTotal"]])

    return df

def stars_to_rob(score):
    return "Low" if score == 1 else "High"

def map_color(score, colors):
    return colors.get(stars_to_rob(score), "#BBBBBB")

def professional_jbi_plot(df: pd.DataFrame, output_file: str, theme: str = "default"):
    theme_options = {
        "default": {"Low":"#06923E","High":"#DC2525"},
        "blue": {"Low":"#3a83b7","High":"#084582"},
        "gray": {"Low":"#FF884DFF","High":"#5B6D80"},
        "smiley": {"Low":"#06923E","High":"#DC2525"},
        "smiley_blue": {"Low":"#3a83b7","High":"#084582"}
    }

    if theme not in theme_options:
        raise ValueError(f"Theme {theme} not available. Choose from {list(theme_options.keys())}")
    colors = theme_options[theme]

    domains = ["Demographics", "History", "ClinicalCondition", "Diagnostics",
               "Intervention", "PostCondition", "AdverseEvents", "Lessons", "Overall RoB"]

    n_studies = len(df)
    per_study_height = 0.5      
    min_first_plot_height = 4.0
    second_plot_height = 4.5    
    gap_between_plots = 3.0   
    top_margin = 1.0            
    bottom_margin = 0.5        
    
    first_plot_height = max(min_first_plot_height, n_studies * per_study_height)
    total_height = first_plot_height + gap_between_plots + second_plot_height + top_margin + bottom_margin
    
    fig = plt.figure(figsize=(18, total_height))

    ax0_bottom = (bottom_margin + second_plot_height + gap_between_plots) / total_height
    ax0_height = first_plot_height / total_height
    
    ax1_bottom = bottom_margin / total_height
    ax1_height = second_plot_height / total_height
    
    ax0 = fig.add_axes([0.12, ax0_bottom, 0.75, ax0_height])
    ax1 = fig.add_axes([0.12, ax1_bottom, 0.75, ax1_height])
    
    domain_pos = {d:i for i,d in enumerate(domains)}
    author_pos = {a:i for i,a in enumerate(df["Author,Year"].tolist())}

    for y in range(len(author_pos)):
        ax0.axhline(y, color='lightgray', linewidth=0.8, zorder=0)
    ax0.axhline(-0.5, color='lightgray', linewidth=0.8, zorder=0)
    ax0.axhline(len(author_pos)-0.5, color='lightgray', linewidth=0.8, zorder=0)

    if theme.startswith("smiley"):
        for _, row in df.iterrows():
            author = row["Author,Year"]
            y_pos = author_pos[author]
            
            for domain in domains[:-1]:
                x_pos = domain_pos[domain]
                symbol = "☺" if row[domain] == 1 else "☹"
                color = colors[stars_to_rob(row[domain])]
                ax0.text(x_pos, y_pos, symbol, fontsize=30, ha='center', va='center', 
                         color=color, fontweight='bold', zorder=1)
            
            x_pos = domain_pos["Overall RoB"]
            symbol = "☺" if row["Overall RoB"] == "Low" else "☹"
            color = colors.get(row["Overall RoB"], "#BBBBBB")
            ax0.text(x_pos, y_pos, symbol, fontsize=30, ha='center', va='center', 
                     color=color, fontweight='bold', zorder=1)
        
        ax0.set_xticks(range(len(domains)))
        ax0.set_xticklabels(domains, fontsize=14, fontweight="bold", rotation=45, ha='right')
        ax0.set_yticks(list(author_pos.values()))
        ax0.set_yticklabels(list(author_pos.keys()), fontsize=11, fontweight="bold", rotation=0)
        ax0.set_ylim(-0.5, len(author_pos)-0.5)
        ax0.set_xlim(-0.5, len(domains)-0.5)
        ax0.set_facecolor('white')
    else:
        x_coords = []
        y_coords = []
        colors_list = []
        
        for _, row in df.iterrows():
            author = row["Author,Year"]
            y_pos = author_pos[author]
            
            for domain in domains[:-1]:
                x_coords.append(domain_pos[domain])
                y_coords.append(y_pos)
                colors_list.append(map_color(row[domain], colors))
            
            x_coords.append(domain_pos["Overall RoB"])
            y_coords.append(y_pos)
            colors_list.append(colors.get(row["Overall RoB"], "#BBBBBB"))
        
        ax0.scatter(x_coords, y_coords, c=colors_list, s=800, marker="s", zorder=1)
        ax0.set_xticks(range(len(domains)))
        ax0.set_xticklabels(domains, fontsize=14, fontweight="bold", rotation=45, ha='right')
        ax0.set_yticks(list(author_pos.values()))
        ax0.set_yticklabels(list(author_pos.keys()), fontsize=11, fontweight="bold", rotation=0)
        ax0.set_ylim(-0.5, len(author_pos)-0.5)

    ax0.set_title("JBI Case Report Traffic-Light Plot", fontsize=18, fontweight="bold")
    ax0.set_xlabel("")
    ax0.set_ylabel("")
    ax0.grid(axis='x', linestyle='--', alpha=0.25)

    risk_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        for domain in domains[:-1]:
            risk = stars_to_rob(row[domain])
            risk_counts[domain][risk] += 1
        
        risk_counts["Overall RoB"][row["Overall RoB"]] += 1
    
    inverted_domains = domains[::-1]
    high_counts = []
    low_counts = []
    
    for domain in inverted_domains:
        high_counts.append(risk_counts[domain].get("High", 0))
        low_counts.append(risk_counts[domain].get("Low", 0))
    
    totals = [h + l for h, l in zip(high_counts, low_counts)]
    high_percent = [h / t * 100 if t > 0 else 0 for h, t in zip(high_counts, totals)]
    low_percent = [l / t * 100 if t > 0 else 0 for l, t in zip(low_counts, totals)]
    
    y_positions = range(len(inverted_domains))
    
    ax1.barh(y_positions, high_percent, color=colors["High"], edgecolor='black', label='High')
    ax1.barh(y_positions, low_percent, left=high_percent, color=colors["Low"], edgecolor='black', label='Low')
    
    for i, (hp, lp) in enumerate(zip(high_percent, low_percent)):
        if hp > 0:
            ax1.text(hp/2, i, f"{hp:.0f}%", ha='center', va='center', 
                     color='black', fontsize=14, fontweight='bold')
        if lp > 0:
            ax1.text(hp + lp/2, i, f"{lp:.0f}%", ha='center', va='center', 
                     color='black', fontsize=14, fontweight='bold')
    
    ax1.set_xlim(0,100)
    ax1.set_xticks([0,20,40,60,80,100])
    ax1.set_xticklabels([0,20,40,60,80,100], fontsize=14, fontweight='bold')  
    ax1.set_yticks(range(len(inverted_domains)))
    ax1.set_yticklabels(inverted_domains, fontsize=14, fontweight='bold') 
    ax1.set_xlabel("Percentage of Studies (%)", fontsize=16, fontweight="bold") 
    ax1.set_ylabel("")
    ax1.set_title("Distribution of Risk-of-Bias Judgments by Domain", fontsize=18, fontweight="bold")
    ax1.grid(axis='x', linestyle='--', alpha=0.25)
    
    for y in range(len(inverted_domains)):
        ax1.axhline(y-0.5, color='lightgray', linewidth=0.8, zorder=0)

    legend_elements = [
        Line2D([0],[0], marker='s', color='w', label='Low Risk', markerfacecolor=colors["Low"], markersize=12),
        Line2D([0],[0], marker='s', color='w', label='High Risk', markerfacecolor=colors["High"], markersize=12)
    ]
    legend = ax0.legend(
        handles=legend_elements,
        title="Domain Risk",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=14,
        title_fontsize=16,
        frameon=True,
        fancybox=True,
        edgecolor='black'
    )
 
    plt.setp(legend.get_title(), fontweight='bold')
    for text in legend.get_texts():
        text.set_fontweight('bold')

    valid_ext = [".png", ".pdf", ".svg", ".eps"]
    ext = os.path.splitext(output_file)[1].lower()
    if ext not in valid_ext:
        raise ValueError(f"Unsupported file format: {ext}. Use one of {valid_ext}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Professional JBI plot saved to {output_file}")

def read_input_file(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(file_path, engine='c')
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file format: {ext}. Provide a CSV or Excel file.")

if __name__ == "__main__":
    if len(sys.argv) not in [3,4]:
        print("Usage: python3 jbi_plot.py input_file output_file.(png|pdf|svg|eps) [theme]")
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]
    theme = sys.argv[3] if len(sys.argv) == 4 else "default"

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    df = read_input_file(input_file)
    df = process_jbi_case_report(df)
    professional_jbi_plot(df, output_file, theme)
    del df