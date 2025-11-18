import os
import json
import glob
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def main(args):
    save_path = os.path.join(args.results_dir, f"{args.method}")
    folder_path = os.path.join(save_path, "context")

    plt.rcParams.update({'font.size': 23})

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}/*.json")

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower().split()
            needle = json_data.get("needle", None).lower()
            expected_answer = args.expected_answer.lower().split()
            score = len(set(model_response).intersection(set(expected_answer))) / len(set(expected_answer))
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)
    locations = list(df["Context Length"].unique())
    locations.sort()

    # df = df.drop(df[df['Context Length'] > 120000].index)  # 120000
    print("Average Score: %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    ## Paper (text version) ###
    for i in range(len(pivot_table.columns)):
        pivot_table = pivot_table.rename(columns={pivot_table.columns[i]: str((pivot_table.columns[i]+200)//1000) + 'k'})
    ###########################

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    #f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    f = plt.figure(figsize=(13.5, 8))
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )

    # More aesthetics
    plt.title(f'Average Score = {df["Score"].mean():.3f}')  # Adds a title
    plt.xlabel('Context Length')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Save the results
    save_png = os.path.join(save_path, "result_plot.png")
    plt.savefig(save_png)
    save_pdf = os.path.join(save_path, "result_plot.pdf")
    plt.savefig(save_pdf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/needle")
    parser.add_argument("--method", type=str,  default=None, choices=["fullkv", "fastkv", "snapkv", "h2o", "streamingllm", "gemfilter", "pyramidinfer"])
    parser.add_argument("--expected_answer", type=str, 
                        default="eat a sandwich and sit in Dolores Park on a sunny day.", help="Path to save the output")
    args = parser.parse_args()
    main(args)