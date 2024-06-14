import torch
from PIL import Image
import json
import argparse
from tqdm.auto import tqdm
import splice
import os

def data_dict(args):
    f = open(os.path.join(args.data_path,'annotations/captions_train2017.json'))
    data = json.load(f)
    f.close()


    img_id_set = {}
    for x in data['annotations']:

        img_id = x["image_id"]

        if img_id in img_id_set:
            img_id_set[img_id].append(x["caption"])
        else:
            img_id_set[img_id] = [x["caption"]]

    return img_id_set

def embed_images(model, preprocess, args):
    img_id_set = data_dict(args)

    img_ids = list(img_id_set.keys())
    for i in tqdm(range(0, len(img_id_set), args.batch_size)):

        batch_ids = img_ids[i: i+args.batch_size]
        img = torch.zeros((len(batch_ids), 3, 224, 224)).to(args.device)

        for j in range(len(batch_ids)):
            img_path = os.path.join(args.data_path, "train2017/", str(batch_ids[j]).zfill(12) + ".jpg")
            img[j] = preprocess(Image.open(img_path)).to(args.device)


        with torch.no_grad():
            image_features = model.encode_image(img)
            # image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features.to_sparse_csr()

        for j in range(len(batch_ids)):
            torch.save(image_features[j], os.path.join(args.out_path, str(batch_ids[j]).zfill(12) + "_image.pth"))

def embed_text(model, tokenizer, args):
    caption_set = data_dict(args)

    for id, captions in caption_set.items():
        text = tokenizer(captions).to(args.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(text)
            # text_features /= text_features.norm(dim=-1, keepdim=True)

        torch.save(text_features, os.path.join(args.out_path, str(id) + "_caption.pth"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default="/datasets/")
    parser.add_argument('-out_path', type=str, default="/datasets/")
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-batch_size', type=int, default=512)
    args = parser.parse_args()

    preprocess = splice.get_preprocess(args.model)
    tokenizer = splice.get_tokenizer(args.model)
    splicemodel = splice.load(args.model, 'laion', 1, args.device, l1_penalty=1, return_weights=False)

    args.out_path = os.path.join(args.out_path, args.model)
    
    embed_images(splicemodel.clip, preprocess, args)
    embed_text(splicemodel.clip, tokenizer, args)

if __name__ == "__main__":
    main()