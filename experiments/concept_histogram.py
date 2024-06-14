import argparse
import splice
import torch
import datasets
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def concept_histogram(splicemodel, dataloader, class_indices, concepts, vocab, device="cpu"):
    splicemodel.eval()

    class_weights={}
    class_totals={}

    splicemodel.return_weights = True
    splicemodel.return_cosine = True

    concept_indices = []
    for concept in concepts:
        concept_indices.append(vocab.index(concept))

    l0 = 0
    cosine = 0
    total = 0

    for idx, (image, label) in enumerate(dataloader):
        idx = torch.zeros(label.shape[0])
        for class_idx in class_indices:
            idx = torch.logical_or((label == class_idx).to(torch.int64), idx)

        idx = torch.argwhere(idx >= 0).squeeze()
        if idx.nelement() == 0:
            continue

        image = image[idx]
        label = label[idx]

        if idx.nelement() == 1:
            imagel, label = image.unsqueeze(0), label.unsqueeze(0)
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)

            (weights, batch_cosine) = splicemodel.encode_image(image)
            target_weights = weights[:, concept_indices].tolist()

            for i in range(image.shape[0]):
                weights_i, label_i = target_weights[i], label[i].item()
                if label_i in class_weights:
                    class_weights[label_i].append(weights_i)
                    class_totals[label_i] += 1
                elif label_i in class_indices:
                    class_weights[label_i] = [weights_i]
                    class_totals[label_i] = 1

            l0 += torch.linalg.vector_norm(weights, dim=1, ord=0).sum().item()
            cosine += batch_cosine.item()
            total += image.shape[0]

    return class_weights, l0/total, cosine/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out_path', type=str)
    parser.add_argument('-l1_penalty', type=float)
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    parser.add_argument('-vocab_size', type=int, default=10000)
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-batch_size', type=int, default=512)
    args = parser.parse_args()

    classes = ['man', 'woman']
    concepts = ['trunks', 'underwear', 'bra', 'swimwear']

    preprocess = splice.get_preprocess(args.model)

    ## train
    dataset = datasets.load(args.dataset, preprocess, args.data_path, train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    class_indices = []
    for clss in classes:
        class_indices.append(dataset.class_to_idx[clss])

    splicemodel = splice.load(args.model, args.vocab, args.vocab_size, args.device, l1_penalty=args.l1_penalty, return_weights=True)

    vocab = splice.get_vocabulary(args.vocab, args.vocab_size)

    class_weights, l0_norm, cosine = concept_histogram(splicemodel, dataloader, class_indices, concepts, vocab, args.device)

    flat_classes = []
    flat_weights = []
    for class_label in class_weights.keys():
        weightlist = torch.mean(torch.tensor(class_weights[class_label]), dim=-1).tolist()
        flat_weights += weightlist
        for _ in range(len(weightlist)):
            flat_classes.append(classes[class_indices.index(class_label)])

    ## test
    test_dataset = datasets.load(args.dataset, preprocess, args.data_path, train=False)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    class_weights, l0_norm, cosine = concept_histogram(splicemodel, dataloader, class_indices, concepts, vocab, args.device)

    for class_label in class_weights.keys():
        weightlist = torch.mean(torch.tensor(class_weights[class_label]), dim=-1).tolist()
        flat_weights += weightlist
        for _ in range(len(weightlist)):
            flat_classes.append(classes[class_indices.index(class_label)])



    ## Plot histogram!
    
    df = pd.DataFrame({"class":flat_classes, "weight":flat_weights})
    sns.set_style("darkgrid", {"axes.facecolor": "whitesmoke"})
    colors = ["#e86276ff", "#629d1eff"]
    custom_palette = sns.color_palette(colors[::-1])
    sns.set_palette(custom_palette, 2)
    ax = sns.histplot(x="weight", data=df, hue="class", bins=20)

    title = str(args.dataset)
    title += " Histogram"
    plt.title(title, fontsize=20)
    plt.xlabel('Weight', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.yscale('log')
    # ax.get_legend().remove()
    sns.despine(bottom=True)
    plt.tight_layout()
    titlepath="_".join(title.split(" ")).lower()
    plt.savefig(os.path.join(args.out_path, titlepath+".pdf"))

if __name__ == "__main__":
    main()