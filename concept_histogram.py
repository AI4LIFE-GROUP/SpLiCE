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
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--l1_penalty', type=float, default=0.2)
    parser.add_argument('--class_label', type=int)
    parser.add_argument('-device', type=str, default="cpu")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--plot', action="store_true")
    args = parser.parse_args()

    preprocess = splice.get_preprocess(args.model)

    dataset = datasets.load(args.dataset, preprocess, args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    splicemodel = splice.load(args.model, args.vocab, 10000, args.device, l1_penalty=args.l1_penalty, return_weights=True)

    if args.class_label is None:
        if args.verbose:
            print("Decomposing full dataset...")
        weights, l0 = splice.decompose_dataset(dataloader, splicemodel, args.device)
    else:
        if args.verbose:
            print("Decomposing class " + str(args.class_label) + "...")
        class_scores, l0 = splice.decompose_classes(dataloader, args.class_label, splicemodel, args.device)
        weights = class_scores[args.class_label]

    vocab = splice.get_vocabulary(args.vocab, 10000)

    _, indices = torch.sort(weights, descending=True)

    if args.plot:
        concept_names = []
        concept_weights = []

    with open(os.path.join(args.out_path, "weights.txt"), "w") as f:
        if args.verbose:
            f.write("Average l0 norm of decompositions: " + str(l0) + "\n")
            print("Average l0 norm of decompositions:", l0)
        for idx in indices.squeeze():
            if weights[idx.item()].item() == 0:
                break

            f.write(str(vocab[idx.item()]) + ", " + str(weights[idx.item()].item()) + "\n")

            if args.verbose:
                print(str(vocab[idx.item()]) + ", " + str(weights[idx.item()].item()))

            if args.plot:
                concept_names.append(str(vocab[idx.item()]))
                concept_weights.append(weights[idx.item()].item())


    if args.plot:
        sns.set_theme(style="darkgrid")
        df = pd.DataFrame({"concept":concept_names[:20], "weight":concept_weights[:20]})
        
        sns.set_color_codes("pastel")
        ax = sns.barplot(y="concept", x="weight", data=df, label="concept", orient = 'h')

        title = str(args.dataset).capitalize()
        if args.class_label:
            title += " Class " + str(args.class_label)
        title += " Decomposition"
        ax.set(title=title, ylabel="Concept",xlabel="Weight")
        ax.get_legend().remove()
        sns.despine(bottom=True)
        plt.tight_layout()
        titlepath="_".join(title.split(" ")).lower()
        plt.savefig(os.path.join(args.out_path, titlepath+".pdf"))
    
if __name__ == "__main__":
    main()