import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import splice
import datasets

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def zero_shot_eval(model, dataloader, label_embeddings):
    """zero_shot_eval Runs zero shot evaluation over a dataloader

    Parameters
    ----------
    model : torch.nn.Module
        A CLIP or SpLiCE model
    dataloader : torch.utils.data.Dataloader
        Dataloader to run eval over
    label_embeddings : torch.tensor
        A {num_labels x CLIP dimensionality} tensor of zero-shot label embeddings for each class ("A photo of a {}").

    Returns
    -------
    avg_accuracy, avg_sparsity, avg_cosine_similarity
    """
    total = 0
    correct = 0
    l0 = 0
    cosine = 0
    model.eval()

    for idx, (image, label) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            original_embedding = model.clip.encode_image(image)
            weights = model.encode_image(image)
            embedding = model.recompose_image(weights)

        cosine_matrix = torch.nn.functional.normalize(embedding, dim=1)@torch.nn.functional.normalize(original_embedding, dim=1).T

        preds = find_closest(embedding, label_embeddings)

        cosine += torch.sum(torch.diag(cosine_matrix)).item()
        l0 += torch.sum(torch.linalg.norm(weights,ord=0,dim=1)).item()
        correct += torch.sum((preds == label)).item()
        total += image.shape[0]
    return correct/total, l0/total, cosine/total

def find_closest(embedding, label_embeddings):
    dot_product = embedding@label_embeddings.T
    return torch.argmax(dot_product, dim=-1)

# def compute_zero_shot(clipmodel, tokenizer, preprocess):
#     dataset_test = CIFAR100("./datasets/", download=True, train=False, transform=preprocess)

#     test_dataloader = DataLoader(dataset_test, batch_size=1024, shuffle=False)

#     label_embeddings = []

#     idx_to_class = dict((v,k) for k,v in dataset_test.class_to_idx.items())    
#     for key in idx_to_class:
#         label_embeddings.append(clipmodel.encode_text(tokenizer("A photo of a {}".format(idx_to_class[key])).to(device)))
    
#     label_embeddings = torch.stack(label_embeddings).squeeze()
#     label_embeddings /= torch.linalg.norm(label_embeddings, dim=-1).view(-1, 1)
    
#     return zero_shot_eval(clipmodel, test_dataloader, label_embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l1_penalty', type=float)
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    parser.add_argument('-vocab_size', type=int, default=10000)
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-data_path', type=str, default="./datasets/")
    parser.add_argument('-batch_size', type=int, default=512)
    args = parser.parse_args()
    print(args, flush=True)

    ## Load SpLiCE Components
    preprocess = splice.get_preprocess(args.model)
    tokenizer = splice.get_tokenizer(args.model)
    splicemodel = splice.load(args.model, args.vocab, args.vocab_size, args.device, l1_penalty=args.l1_penalty, return_weights=True)
    vocab = splice.get_vocabulary(args.vocab, args.vocab_size)

    ## Load dataset
    dataset = datasets.load(args.dataset, preprocess, args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    ## Construct zero shot embeddings. Requires dataset to have 'class_to_idx' field.
    label_embeddings = []

    idx_to_class = dict((v,k) for k,v in dataset.class_to_idx.items())    
    for key in idx_to_class:
        label_embeddings.append(splicemodel.encode_text(tokenizer("A photo of a {}".format(idx_to_class[key])).to(device)))
    
    label_embeddings = torch.stack(label_embeddings).squeeze()
    label_embeddings /= torch.linalg.norm(label_embeddings, dim=-1).view(-1, 1)

    zero_shot_acc = zero_shot_eval(splicemodel, dataloader, label_embeddings)

    print("Zero shot accuracy, sparsity, and cosine similarity of reconstruction with SpLiCE: {}".format(zero_shot_acc))
    
if __name__ == "__main__":
    main()