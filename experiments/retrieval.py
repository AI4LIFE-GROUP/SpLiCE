import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import argparse
import splice
import os

class CLIPDataset(torch.utils.data.Dataset):
    """
    Custom Torch dataset for our CLIP model that loads instances from disk
    """

    def __init__(self, ids, data_folder):
        self.ids = ids
        self.data_folder = data_folder

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        img_captions = torch.load(os.path.join(self.data_folder, f"{id.zfill(12)}.pth"))
        image = img_captions[0]
        captions = img_captions[1:]
        caption = captions[torch.randperm(captions.shape[0])[0]]
        return image, caption

def retrieve(splicemodel, dataloader, args):
    logit_scale = splicemodel.clip.logit_scale
    splicemodel.clip = None
    batched_loss = []
    batched_img_top_1 = []
    batched_text_top_1 = []
    batched_img_top_5 = []
    batched_text_top_5 = []
    batched_img_top_10 = []
    batched_text_top_10 = []

    for batch_i, (batch_x, batch_y) in enumerate(dataloader):
        with torch.no_grad():
            image_features = batch_x.to(args.device).to(torch.float32)
            text_features = batch_y.to(args.device).to(torch.float32)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            if not args.clip:
                image_features = splicemodel.encode_image(image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits

            logits_per_image = logit_scale * image_features @ text_features.t()
            image_preds_ind = torch.argsort(logits_per_image, dim=1)
            logits_per_text = logits_per_image.t()
            text_preds_ind = torch.argsort(logits_per_text, dim=1)

            batch_size = batch_x.shape[0]
            labels = torch.arange(batch_size, device=args.device).long()

            batched_img_top_1.append(torch.Tensor([l in image_preds_ind[l, -1:] for l in range(batch_size)]).mean())
            batched_text_top_1.append(torch.Tensor([l in text_preds_ind[l, -1:] for l in range(batch_size)]).mean())

            batched_img_top_5.append(torch.Tensor([l in image_preds_ind[l, -5:] for l in range(batch_size)]).mean())
            batched_text_top_5.append(torch.Tensor([l in text_preds_ind[l, -5:] for l in range(batch_size)]).mean())

            batched_img_top_10.append(torch.Tensor([l in image_preds_ind[l, -10:] for l in range(batch_size)]).mean())
            batched_text_top_10.append(torch.Tensor([l in text_preds_ind[l, -10:] for l in range(batch_size)]).mean())


            total_loss = (
                torch.nn.functional.cross_entropy(logits_per_image, labels, reduction="mean") +
                torch.nn.functional.cross_entropy(logits_per_text, labels, reduction="mean")
            ) / 2
            
        print(total_loss.item(), flush=True)
        batched_loss.append(total_loss)

        if batch_i == 8:
            break

    batched_loss = torch.Tensor(batched_loss)
    batched_img_top_1 = torch.Tensor(batched_img_top_1)
    batched_text_top_1 = torch.Tensor(batched_text_top_1)
    batched_img_top_5 = torch.Tensor(batched_img_top_5)
    batched_text_top_5 = torch.Tensor(batched_text_top_5)
    batched_img_top_10 = torch.Tensor(batched_img_top_10)
    batched_text_top_10 = torch.Tensor(batched_text_top_10)

    print("BATCH OVER 10 LOSSES:", (batched_loss.mean()).item(), (batched_loss.std()).item())
    print("IMG RECALL TOP 1:", (batched_img_top_1.mean()).item(), (batched_img_top_1.std()).item())
    print("TEXT RECALL TOP 1:", (batched_text_top_1.mean()).item(), (batched_text_top_1.std()).item())
    print("IMG RECALL TOP 5:", (batched_img_top_5.mean()).item(), (batched_img_top_5.std()).item())
    print("TEXT RECALL TOP 5:", (batched_text_top_5.mean()).item(), (batched_text_top_5.std()).item())
    print("IMG RECALL TOP 10:", (batched_img_top_10.mean()).item(), (batched_img_top_10.std()).item())
    print("TEXT RECALL TOP 10:", (batched_text_top_10.mean()).item(), (batched_text_top_10.std()).item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l1_penalty', type=float)
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    parser.add_argument('-vocab_size', type=int, default=10000)
    parser.add_argument('-data_path', type=str, default="./datasets/")
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-clip', action="store_true")
    args = parser.parse_args()
    print(args, flush=True)

    ## Load SpLiCE Components
    splicemodel = splice.load(args.model, args.vocab, args.vocab_size, args.device, l1_penalty=args.l1_penalty, return_weights=False)

    ## Load dataset
    filepath = os.path.join(args.data_path, args.model)
    with open("data/mscoco_ids.txt") as file:
        lines = [line.rstrip() for line in file]
    dataset = CLIPDataset(lines, filepath)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    retrieve(splicemodel, dataloader, args)

if __name__ == "__main__":
    main()