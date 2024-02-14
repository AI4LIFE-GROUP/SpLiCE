import argparse
import torch
import splice
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str)
    parser.add_argument('-out_path', type=str)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('-l1_penalty', type=float)
    parser.add_argument('-device', type=str, default="cpu")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    args = parser.parse_args()

    splicemodel = splice.load(args.model, args.vocab, 10000, args.device, l1_penalty = args.l1_penalty, return_weights=True)
    preprocess = splice.get_preprocess(args.model)
    img = preprocess(Image.open(args.path)).to(args.device).unsqueeze(0)

    weights, l0_norm, cosine = splice.decompose_image(img, splicemodel, args.device)

    vocab = splice.get_vocabulary(args.vocab, 10000)

    _, indices = torch.sort(weights, descending=True)

    with open(args.out_path, "w") as f:

        f.write("Concept Decomposition of " + str(args.path) + ": \n")
        print("Concept Decomposition of " + str(args.path) + ":")

        for idx in indices.squeeze():
            if weights[0, idx.item()].item() == 0:
                break
            f.write("\t" + str(vocab[idx.item()]) + "\t" + str(round(weights[0, idx.item()].item(), 4)) + "\n")
            if args.verbose:
                print("\t" + str(vocab[idx.item()]) + "\t" + str(round(weights[0, idx.item()].item(), 4)))

        f.write("Decomposition L0 Norm: \t" + str(l0_norm) + "\n")
        print("Decomposition L0 Norm: \t" + str(l0_norm))

        f.write("CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)) + "\n")
        print("CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)))

if __name__ == "__main__":
    main()