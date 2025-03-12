import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import cmcrameri.cm as cmc  # Import the cmcrameri colormaps

def load_benchmark_results(csv_file):
    """Load benchmark results from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} benchmark results.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def create_summary_tables(df):
    """Create and print summary tables of the benchmark results."""
    # Print overall summary
    print("\nBenchmark Results Summary:")
    print(tabulate(df.describe(), headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create summary table grouped by interpolation method
    print("\nSummary by Interpolation Method:")
    interp_summary = df.groupby('Interpolation').agg({
        'Speedup': ['mean', 'min', 'max']
    })
    print(tabulate(interp_summary, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create summary table grouped by image size
    print("\nSummary by Image Size:")
    size_summary = df.groupby('Image Size').agg({
        'Speedup': ['mean', 'min', 'max']
    })
    print(tabulate(size_summary, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create summary table grouped by SIP usage if it exists
    if 'SIP' in df.columns:
        print("\nSummary by SIP Distortion:")
        sip_summary = df.groupby('SIP').agg({
            'Speedup': ['mean', 'min', 'max'],
            'Reproject Time (s)': 'mean',
            'DFReproject Time (s)': 'mean'
        })
        print(tabulate(sip_summary, headers='keys', tablefmt='grid', floatfmt='.4f'))

def plot_execution_times(df, output_file='execution_times.png'):
    """Plot execution times for both packages."""
    plt.figure(figsize=(12, 8))
    
    # Create a categorical x-axis combining size and interpolation
    if 'Size_Interp' not in df.columns:
        df['Size_Interp'] = df['Image Size'] + '\n' + df['Interpolation']
    
    # Sort by Image Size and Interpolation
    df_sorted = df.sort_values(['Image Size', 'Interpolation'])
    
    # Create the bar chart
    bar_positions = np.arange(len(df_sorted))
    bar_width = 0.35
    
    # Use specific values from the colormap to get colors
    reproject_color = cmc.hawaii_r(0.3)
    dfreproject_color = cmc.hawaii_r(0.7)
    
    plt.bar(bar_positions - bar_width/2, df_sorted['Reproject Time (s)'], 
            bar_width, label='reproject', color=reproject_color)
    plt.bar(bar_positions + bar_width/2, df_sorted['DFReproject Time (s)'], 
            bar_width, label='dfreproject', color=dfreproject_color)
    
    plt.xlabel('Image Size and Interpolation Method')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(bar_positions, df_sorted['Size_Interp'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Execution time plot saved to {output_file}")

def plot_speedup_heatmap(df, output_file='speedup_heatmap.png'):
    """Create a heatmap of speedup factors."""
    plt.figure(figsize=(12, 8))
    
    # Reshape data for heatmap
    heatmap_data = df.pivot_table(
        index='Interpolation', 
        columns='Image Size', 
        values='Speedup'
    )
    
    # Create the heatmap
    # Convert the colormap to a string name if needed or use a compatible cmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu',
                linewidths=.5, cbar_kws={'label': 'Speedup Factor'})
    
    plt.title('Speedup Factor (reproject/dfreproject)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Speedup heatmap saved to {output_file}")

def plot_speedup_by_size(df, output_file='speedup_by_size.png'):
    """Create a line plot showing speedup by image size."""
    plt.figure(figsize=(14, 8))
    
    # Create a list of colors from the colormap
    n_interp = len(df['Interpolation'].unique())
    colors = [cmc.glasgow(i / (n_interp - 1) if n_interp > 1 else 0.5) for i in range(n_interp)]
    
    # Create a grouped line plot
    for i, interp in enumerate(df['Interpolation'].unique()):
        interp_data = df[df['Interpolation'] == interp]
        
        # Convert image size to numeric for proper scaling
        interp_data['Size_Numeric'] = interp_data['Image Size'].apply(
            lambda x: int(x.split('x')[0]) * int(x.split('x')[1])
        )
        
        # Sort by numeric size
        interp_data = interp_data.sort_values('Size_Numeric')
        
        # Use a color from the pre-generated list
        plt.plot(interp_data['Image Size'], interp_data['Speedup'], 
                marker='o', linewidth=2, label=interp, 
                color=colors[i])
    
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Image Size')
    plt.ylabel('Speedup Factor (reproject/dfreproject)')
    plt.title('Speedup Factor by Image Size and Interpolation Method')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Speedup by size plot saved to {output_file}")

def plot_sip_comparison(df, output_file_prefix='sip_comparison'):
    """Create visualizations comparing performance with and without SIP."""
    if 'SIP' not in df.columns:
        print("SIP column not found in data. Skipping SIP comparison plots.")
        return
    
    # 1. Side-by-side heatmaps for with/without SIP
    plt.figure(figsize=(16, 7))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, sip_value in enumerate(df['SIP'].unique()):
        sip_data = df[df['SIP'] == sip_value]
        
        # Reshape data for heatmap
        heatmap_data = sip_data.pivot_table(
            index='Interpolation', 
            columns='Image Size', 
            values='Speedup'
        )
        
        # Create the heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap=cmc.lajolla,
                    linewidths=.5, ax=axes[i],
                    cbar_kws={'label': 'Speedup Factor'})
        
        axes[i].set_title(f'Speedup Factor {sip_value}')
    
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot of speedup by SIP and interpolation
    plt.figure(figsize=(12, 8))
    # Convert the colormap to a list of colors for the number of SIP categories
    n_colors = len(df['SIP'].unique())
    colors = [cmc.managua(i / (n_colors - 1) if n_colors > 1 else 0.5) for i in range(n_colors)]
    sns.boxplot(x='Interpolation', y='Speedup', hue='SIP', data=df, palette=colors)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.title('Distribution of Speedup by Interpolation Method and SIP')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Line plot with SIP as groups
    plt.figure(figsize=(14, 10))

    interp_line_styles = ['-', '--', '-.']
    # Create a grouped line plot for each SIP + interpolation combination
    for sip in df['SIP'].unique():
        sip_data = df[df['SIP'] == sip]
        
        for interp in sip_data['Interpolation'].unique():
            filter_data = sip_data[sip_data['Interpolation'] == interp]
            
            # Convert image size to numeric for proper scaling
            filter_data['Size_Numeric'] = filter_data['Image Size'].apply(
                lambda x: int(x.split('x')[0]) * int(x.split('x')[1])
            )
            
            # Sort by numeric size
            filter_data = filter_data.sort_values('Size_Numeric')
            
            # Calculate color index for gradient
            interp_idx = list(df['Interpolation'].unique()).index(interp)
            sip_idx = list(df['SIP'].unique()).index(sip)
            color_idx = (interp_idx * len(df['SIP'].unique()) + sip_idx) / (len(df['Interpolation'].unique()) * len(df['SIP'].unique()) - 1)
            
            plt.plot(filter_data['Image Size'], filter_data['Speedup'], 
                    marker='o', linewidth=2, 
                    label=f"{interp} ({sip})",
                    color=cmc.managua(color_idx),)
                     # linestyle=interp_line_styles[interp_idx])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylim(3, 50)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Image Size', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('Speedup Factor (dfreproject/reproject)' , fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Speedup Factor by Image Size, Interpolation Method, and SIP Distortion', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_file_prefix}_line.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SIP comparison plots saved with prefix {output_file_prefix}")

def main():
    # Get the CSV file from user
    csv_file = input("Enter the path to the benchmark results CSV file: ")
    
    # Load the benchmark results
    df = load_benchmark_results(csv_file)
    if df is None:
        return
    
    # Create summary tables
    create_summary_tables(df)
    
    # Generate basic plots
    plot_execution_times(df)
    plot_speedup_heatmap(df)
    plot_speedup_by_size(df)
    
    # Generate SIP comparison plots if applicable
    plot_sip_comparison(df)
    
    print("\nVisualization complete. All plots saved to current directory.")

if __name__ == "__main__":
    main()
