import argparse
import torch
from torch.utils.data import DataLoader
import splice
import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-out_folder', type=str)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('-l1_penalty', type=float, default=0.2)
    parser.add_argument('-class_label', type=int)
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-plot_topk', type=int, default=10)
    args = parser.parse_args()

    preprocess = splice.get_preprocess(args.model)

    dataset = datasets.load(args.dataset, preprocess, args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    splicemodel = splice.load(args.model, args.vocab, 10000, args.device, l1_penalty=args.l1_penalty, return_weights=True)

    if args.class_label is None:
        if args.verbose:
            print("Decomposing " + str(args.dataset) + "...")
        weights, l0_norm, cosine = splice.decompose_dataset(dataloader, splicemodel, args.device)
    else:
        if args.verbose:
            print("Decomposing class " + str(args.class_label) + "...")
        class_weights, l0_norm, cosine = splice.decompose_classes(dataloader, args.class_label, splicemodel, args.device)
        weights = class_weights[args.class_label]

    vocab = splice.get_vocabulary(args.vocab, 10000)

    _, indices = torch.sort(weights, descending=True)

    concept_names = []
    concept_weights = []

    with open(os.path.join(args.out_folder, "weights.txt"), "w") as f:
        
        f.write("Concept Decomposition: \n")
        print("Concept Decomposition:")

        for idx in indices.squeeze():
            if weights[idx.item()].item() == 0:
                break

            f.write("\t" + str(vocab[idx.item()]) + "\t" + str(round(weights[idx.item()].item(), 4)) + "\n")

            if args.verbose:
                print("\t" + str(vocab[idx.item()]) + "\t" + str(round(weights[idx.item()].item(), 4)))

            concept_names.append(str(vocab[idx.item()]))
            concept_weights.append(weights[idx.item()].item())

        if args.verbose:
            f.write("Average Decomposition L0 Norm: \t" + str(l0_norm) + "\n")
            print("Average Decomposition L0 Norm: \t" + str(l0_norm))

            f.write("Average CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)) + "\n")
            print("Average CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)))


    df = pd.DataFrame({"concept":concept_names[:args.plot_topk], "weight":concept_weights[:args.plot_topk]})
    sns.set_style("darkgrid", {"axes.facecolor": "whitesmoke"})
    colors = ["#e86276ff", "#629d1eff"]
    custom_palette = sns.color_palette(colors)
    sns.set_palette(custom_palette, 2)
    ax = sns.barplot(y="concept", x="weight", data=df, label="concept", orient = 'h')

    title = str(args.dataset)
    if args.class_label:
        title += " Class " + str(args.class_label)
    title += " Decomposition"
    plt.title(title, fontsize=20)
    plt.xlabel('Weight', fontsize=16)
    plt.ylabel('Concept', fontsize=16)
    ax.get_legend().remove()
    sns.despine(bottom=True)
    plt.tight_layout()
    titlepath="_".join(title.split(" ")).lower()
    plt.savefig(os.path.join(args.out_folder, titlepath+".pdf"))
    
if __name__ == "__main__":
    main()