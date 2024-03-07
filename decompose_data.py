import argparse
import torch
import splice
from PIL import Image
from torch.utils.data import DataLoader
import datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out_path', type=str)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('-l1_penalty', type=float)
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-class_label', type=int)
    parser.add_argument('-batch_size', type=int, default=512)

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
            print("Decomposing class " + str(args.class_label) + " from " + str(args.dataset) +"...")
        class_weights, l0_norm, cosine = splice.decompose_classes(dataloader, args.class_label, splicemodel, args.device)
        weights = class_weights[args.class_label]

    vocab = splice.get_vocabulary(args.vocab, 10000)

    _, indices = torch.sort(weights, descending=True)

    with open(args.out_path, "w") as f:

        f.write("Concept Decomposition:" + "\n")
        print("Concept Decomposition:")

        for idx in indices.squeeze():
            if str(round(weights[idx.item()].item(), 4)) == "0.0":
                break
            f.write("\t" + str(vocab[idx.item()]) + "\t" + str(round(weights[idx.item()].item(), 4)) + "\n")
            if args.verbose:
                print("\t" + str(vocab[idx.item()]) + "\t" + str(round(weights[idx.item()].item(), 4)))

        f.write("Average Decomposition L0 Norm: \t" + str(l0_norm)+ "\n")
        print("Average Decomposition L0 Norm: \t" + str(l0_norm))

        f.write("Average CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)) + "\n")
        print("Average CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)))

if __name__ == "__main__":
    main()