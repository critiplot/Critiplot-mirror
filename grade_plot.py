import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

def process_grade(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "Other Considerations": "Publication Bias"
    }
    df = df.rename(columns=column_map)
    
   
    domain_columns = ["Risk of Bias", "Inconsistency", "Indirectness", "Imprecision", "Publication Bias", "Overall Certainty"]
    for col in domain_columns:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    
    required_columns = ["Outcome","Study","Risk of Bias","Inconsistency","Indirectness","Imprecision","Publication Bias","Overall Certainty"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

   
    df["Outcome_Display"] = df["Outcome"] + " (" + df["Study"] + ")"

    df['Original_Order'] = range(len(df))
    return df

def map_color(certainty, colors):
    return colors.get(certainty, "grey")

def grade_plot(df: pd.DataFrame, output_file: str, theme="default"):
    theme_options = {
        "green": {  
            "High":"#276B37",
            "Moderate":"#56AF29",
            "Low":"#3376AD",
            "Very Low":"#7D7D7D",
            "None":"#B5B5B5"
        },
        "default": {  
            "High":"#3A896F",
            "Moderate":"#AEBF2B",
            "Low":"#FFBB00",
            "Very Low":"#B42222",
            "None":"#818181"
        },
        "blue": {  
            "High":"#006699",
            "Moderate":"#3399CC",
            "Low":"#FFCC66",
            "Very Low":"#CC3333",
            "None":"#B0B0B0"
        }
    }

    if theme not in theme_options:
        raise ValueError("Invalid theme.")
    colors = theme_options[theme]

    fig_height = max(6, 0.7*len(df) + 5)
    fig = plt.figure(figsize=(18, fig_height))
    gs = GridSpec(2,1, height_ratios=[len(df)*0.7, 1.5], hspace=0.4)


    domains = ["Risk of Bias","Inconsistency","Indirectness","Imprecision","Publication Bias", "Overall Certainty"]
    

    ax0 = fig.add_subplot(gs[0])
    
   
    df_copy = df.copy()
    
  
    outcome_order = df_copy["Outcome_Display"].tolist()
    
   
    plot_data = []
    for _, row in df_copy.iterrows():
        for domain in domains:
            plot_data.append({
                "Outcome_Display": row["Outcome_Display"],
                "Domain": domain,
                "Certainty": row[domain],
                "Original_Order": row["Original_Order"]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
   
    plot_df["Color"] = plot_df["Certainty"].apply(lambda x: map_color(x, colors))
    
  
    plot_df['Outcome_Display'] = pd.Categorical(
        plot_df['Outcome_Display'], 
        categories=outcome_order, 
        ordered=True
    )
    
   
    sns.scatterplot(
        data=plot_df, 
        x="Domain", 
        y="Outcome_Display",
        hue="Color", 
        palette={c:c for c in plot_df["Color"].unique()},
        s=350, 
        marker="s", 
        legend=False, 
        ax=ax0
    )
    

    ax0.set_yticks(range(len(outcome_order)))
    ax0.set_yticklabels(outcome_order, fontsize=10, fontweight="bold")
    

    for y in range(len(outcome_order)+1):
        ax0.axhline(y-0.5, color='lightgray', linewidth=0.8, zorder=0)


    ax0.set_xticks(range(len(domains)))
    ax0.set_xticklabels(domains, fontsize=12, fontweight="bold")
    ax0.set_xlim(-0.5, len(domains)-0.5)
    ax0.set_ylim(-0.5, len(outcome_order)-0.5)
    ax0.set_facecolor("white")


    ax0.set_title("GRADE Traffic-Light Plot", fontsize=18, fontweight="bold")
    ax0.set_xlabel("GRADE Domains", fontsize=12, fontweight="bold")
    ax0.set_ylabel("", fontsize=12, fontweight="bold")
    ax0.tick_params(axis='y', labelsize=10)


    legend_elements = [Patch(facecolor=colors[c], edgecolor='black', label=c) for c in ["High","Moderate","Low","Very Low","None"]]
    leg = ax0.legend(handles=legend_elements, title="Certainty", bbox_to_anchor=(1.02,1), loc='upper left', frameon=True, borderpad=1)
    plt.setp(leg.get_texts(), fontweight="bold")
    plt.setp(leg.get_title(), fontweight="bold")

 
    ax1 = fig.add_subplot(gs[1])
    

    bar_data = []
    for domain in domains:
       
        certainty_counts = df_copy[domain].value_counts()
        total = certainty_counts.sum()
        
        for certainty in ["High", "Moderate", "Low", "Very Low", "None"]:
            count = certainty_counts.get(certainty, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            bar_data.append({
                "Domain": domain,
                "Certainty": certainty,
                "Count": count,
                "Percentage": percentage
            })
    
    bar_df = pd.DataFrame(bar_data)
    
    
    bar_df['Domain'] = pd.Categorical(
        bar_df['Domain'], 
        categories=domains, 
        ordered=True
    )
    
    
    bottom = pd.Series([0.0] * len(domains), index=domains)
    for cert in ["Very Low", "Low", "Moderate", "High", "None"]:
        cert_data = bar_df[bar_df['Certainty'] == cert]
        if not cert_data.empty:
            
            cert_series = pd.Series(0.0, index=domains)
            for _, row in cert_data.iterrows():
                cert_series[row['Domain']] = row['Percentage']
            
            ax1.barh(
                range(len(domains)), 
                cert_series, 
                left=bottom,
                color=colors[cert], 
                edgecolor="black", 
                linewidth=1.5, 
                label=cert
            )
            
            
            for i, domain in enumerate(domains):
                val = cert_series[domain]
                if val > 0: 
                    ax1.text(
                        bottom[i] + val/2, 
                        i, 
                        f"{val:.1f}%", 
                        va='center', 
                        ha='center', 
                        fontsize=10, 
                        color='black', 
                        fontweight="bold"
                    )
            
            bottom = bottom + cert_series

  
    ax1.set_xlim(0,100)
    ax1.set_xlabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("", fontsize=12, fontweight="bold")
    ax1.set_title("Distribution of GRADE Judgments by Domain", fontsize=18, fontweight="bold")
    
  
    ax1.set_yticks(range(len(domains)))
    ax1.set_yticklabels(domains, fontsize=12, fontweight="bold")
   
   
    for y in range(len(domains)):
        ax1.axhline(y-0.5, color='lightgray', linewidth=0.8, zorder=0)

    
    for label in ax1.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")

   
    fig.subplots_adjust(left=0.05, right=0.78, top=0.95, bottom=0.05, hspace=0.4)

    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ GRADE plot saved to {output_file}")

def read_input_file(input_file: str) -> pd.DataFrame:
    if input_file.endswith(".csv"):
        return pd.read_csv(input_file)
    elif input_file.endswith(".xlsx") or input_file.endswith(".xls"):
        return pd.read_excel(input_file)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx/.xls")

if __name__ == "__main__":
    if len(sys.argv) not in [3,4]:
        print("Usage: python3 grade_plot.py input_file output_file.(png|pdf|svg|eps) [theme]")
        sys.exit(1)
    input_file, output_file = sys.argv[1], sys.argv[2]
    theme = sys.argv[3] if len(sys.argv)==4 else "default"
    df = read_input_file(input_file)
    df = process_grade(df)
    grade_plot(df, output_file, theme)