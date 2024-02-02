import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import splice

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def zero_shot_eval(model, dataloader, label_embeddings):
    total = 0
    correct = 0
    model.eval()

    for idx, (image, label) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        label = label.to(device)

        embedding = model.encode_image(image)

        if embedding.shape[1] == 512: ## if model outputs embeddings in clipspace
            embedding /= torch.linalg.norm(embedding, dim=-1).view(-1, 1)

        preds = find_closest(embedding, label_embeddings)

        correct += torch.sum((preds == label)).item()
        total += image.shape[0]
    return correct/total

def find_closest(embedding, label_embeddings):
    dot_product = embedding@label_embeddings.T
    return torch.argmax(dot_product, dim=-1)

def compute_zero_shot(clipmodel, tokenizer, preprocess, dataset="CIFAR100"):
    dataset_test = CIFAR100("/n/holylabs/LABS/hlakkaraju_lab/Lab/datasets/", download=True, train=False, transform=preprocess)

    test_dataloader = DataLoader(dataset_test, batch_size=8192, shuffle=False)

    label_embeddings = []

    idx_to_class = dict((v,k) for k,v in dataset_test.class_to_idx.items())    
    for key in idx_to_class:
        print(idx_to_class[key])
        label_embeddings.append(clipmodel.encode_text(tokenizer("A photo of a {}".format(idx_to_class[key])).to(device)))
    
    label_embeddings = torch.stack(label_embeddings).squeeze()
    label_embeddings /= torch.linalg.norm(label_embeddings, dim=-1).view(-1, 1)
    
    return zero_shot_eval(clipmodel, test_dataloader, label_embeddings)

def main():
    ## load model
    # clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    # clipmodel = clipmodel.to(device)
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # zero_shot_acc = compute_zero_shot(clipmodel, tokenizer, preprocess, dataset="Tiny-Imagenet")
    # print("Zero shot accuracy with base CLIP: {}".format(zero_shot_acc))
    
    splicemodel = splice.load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000, device="cuda")
    tokenizer = splice.get_tokenizer("open_clip:ViT-B-32")
    preprocess = splice.get_preprocess("open_clip:ViT-B-32")
    zero_shot_acc = compute_zero_shot(splicemodel, tokenizer, preprocess, dataset="CIFAR100")
    print("Zero shot accuracy with SpLiCE: {}".format(zero_shot_acc))
    
if __name__ == "__main__":
    main()