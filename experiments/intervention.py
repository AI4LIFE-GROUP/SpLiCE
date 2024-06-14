import splice
from datasets import CelebA, WaterbirdDataset
import torch
import time
import argparse
from torch.utils.data import DataLoader
from zero_shot import find_closest
from sklearn import linear_model
import numpy as np

def zero_shot_eval(model, dataloader, label_embeddings, obj, device, intervene=None):
    total = 0
    correct = 0
    model.eval()

    image_l0s = []
    image_l1s = []

    class_0_acc = {"c":0, "t":0}
    class_1_acc = {"c":0, "t":0}

    for idx, (image, objlabel, adjlabel) in enumerate(dataloader):
        with torch.no_grad():
            image = image.to(device)

            label = objlabel if obj else adjlabel
            label = label.to(device)

            if intervene == None:
                embedding = model.encode_image(image)
            else:
                embedding = model.intervene_image(image.to(device), intervene).to(device)

            if embedding.shape[1] == 512: ## if model outputs embeddings in clipspace
                embedding /= torch.linalg.norm(embedding, dim=-1).view(-1, 1)

            image_l0s.append(torch.mean(torch.linalg.vector_norm(embedding, 0, 1)))
            image_l1s.append(torch.mean(torch.linalg.vector_norm(embedding, 1, 1)))

            preds = find_closest(embedding, label_embeddings)

            correct += torch.sum((preds == label).to(torch.int64)).item()
            total += image.shape[0]

            for i, y in enumerate(label):
                if y == 1:
                    if (preds[i] == 1):
                        class_1_acc["c"] += 1
                    class_1_acc["t"] += 1
                elif y == 0:
                    if (preds[i] == 0):
                        class_0_acc["c"] += 1
                    class_0_acc["t"] += 1


    return correct/total, class_0_acc["c"]/class_0_acc["t"], class_1_acc["c"]/class_1_acc["t"]

def concept_zero_shot_eval(tokenizer, clipmodel, test_loader, labels, obj, device, intervene=None):
    label_embeddings = []
    for l in labels:
        e = clipmodel.encode_text(tokenizer(l).to(device))
        label_embeddings.append(e)

    label_embeddings = torch.stack(label_embeddings).squeeze()
    label_embeddings /= torch.linalg.norm(label_embeddings, dim=-1).view(-1, 1)

    obj_acc, obj_0_acc, obj_1_acc = zero_shot_eval(clipmodel, test_loader, label_embeddings, obj=obj, device=device, intervene=intervene)
    return round(obj_acc, 3), round(obj_0_acc, 3), round(obj_1_acc, 3)

def train_cbm(X_train, y_train1, reg):
    cbm1 = linear_model.LogisticRegression(penalty="l1", C = reg, solver="saga", fit_intercept=False)
    cbm1.fit(X_train, y_train1)
    return cbm1

def test_cbm_subgroup_accs(cbm, X_test, y_test1, y_test2):
    indices_y1_0_y2_0 = np.nonzero(np.logical_and((y_test1 == 0), (y_test2 == 0)).astype(np.int64))
    indices_y1_1_y2_0 = np.nonzero(np.logical_and((y_test1 == 1), (y_test2 == 0)).astype(np.int64))
    indices_y1_0_y2_1 = np.nonzero(np.logical_and((y_test1 == 0), (y_test2 == 1)).astype(np.int64))
    indices_y1_1_y2_1 = np.nonzero(np.logical_and((y_test1 == 1), (y_test2 == 1)).astype(np.int64))

    return cbm.score(X_test, y_test1), cbm.score(X_test[indices_y1_0_y2_0], y_test1[indices_y1_0_y2_0]), cbm.score(X_test[indices_y1_1_y2_0], y_test1[indices_y1_1_y2_0]), cbm.score(X_test[indices_y1_0_y2_1], y_test1[indices_y1_0_y2_1]), cbm.score(X_test[indices_y1_1_y2_1], y_test1[indices_y1_1_y2_1])

def test_cbm(cbm, X_test, y_test1):
    indices_y1_0 = np.nonzero((y_test1 == 0).astype(np.int64))
    indices_y1_1 = np.nonzero((y_test1 == 1).astype(np.int64))

    return cbm.score(X_test, y_test1), cbm.score(X_test[indices_y1_0], y_test1[indices_y1_0]), cbm.score(X_test[indices_y1_1], y_test1[indices_y1_1])

def intervene_celeba(splicemodel, preprocess, tokenizer, vocab, intervening_indices, args):
    dataset_train = CelebA(args.data_path, train=False, transform=preprocess)
    dataset_test = CelebA(args.data_path, train=True, transform=preprocess)

    train_loader = DataLoader(dataset_train, batch_size=1024, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1024, shuffle=False)

    obj_labels = ["A picture of a man", "A picture of a woman"]
    adj_labels = ["A picture of person without glasses", "A picture of person with glasses"]

    print("======ZERO SHOT========")

    print("CLIP ZERO SHOT:", flush=True)
    print("Gender Zero Shot Full, Female, Male: ", concept_zero_shot_eval(tokenizer, splicemodel.clip, test_loader, obj_labels, obj=True, device=args.device))
    print("Glasses Zero Shot Full, No Glasses, Glasses", concept_zero_shot_eval(tokenizer, splicemodel.clip, test_loader, adj_labels, obj=False, device=args.device))

    print("SpLiCE ZERO SHOT:", flush=True)
    print("Gender Zero Shot Full, Female, Male: ", concept_zero_shot_eval(tokenizer, splicemodel, test_loader, obj_labels, obj=True, device=args.device))
    print("Glasses Zero Shot Full, No Glasses, Glasses", concept_zero_shot_eval(tokenizer, splicemodel, test_loader, adj_labels, obj=False, device=args.device))

    print("INTERVENED SpLiCE ZERO SHOT:", flush=True)
    print("Gender Zero Shot Full, Female, Male: ",concept_zero_shot_eval(tokenizer, splicemodel, test_loader, obj_labels, obj=True, intervene=intervening_indices, device=args.device))
    print("Glasses Zero Shot Full, No Glasses, Glasses",concept_zero_shot_eval(tokenizer, splicemodel, test_loader, adj_labels, obj=False, intervene=intervening_indices, device=args.device))

    print("======PROBING========")

    splicemodel.return_weights = True

    X_full = None
    y_full1 = None
    y_full2 = None

    for i, (X, y1, y2) in enumerate(train_loader):
        X = X.to(args.device)
        y1 = y1.to(args.device)
        y2 = y2.to(args.device)
        with torch.no_grad():
            if X_full is None:
                X_full = splicemodel.encode_image(X)
            else:
                X_full = torch.cat((X_full, splicemodel.encode_image(X)))
        if y_full1 is None:
            y_full1 = y1
        else:
            y_full1 = torch.cat((y_full1, y1))
        if y_full2 is None:
            y_full2 = y2
        else:
            y_full2 = torch.cat((y_full2, y2))
    
    X_train, y_train1, y_train2 = (X_full).cpu().numpy(), (y_full1).cpu().numpy(), (y_full2).cpu().numpy()

    cbm_gender = train_cbm(X_train, y_train1, reg = args.probe_regularization)
    cbm_glasses = train_cbm(X_train, y_train2, reg = args.probe_regularization)

    X_full = None
    y_full1 = None
    y_full2 = None

    for X, y1, y2 in test_loader:
        X = X.to(args.device)
        y1 = y1.to(args.device)
        y2 = y2.to(args.device)
        with torch.no_grad():
            if X_full is None:
                X_full = splicemodel.encode_image(X.cuda())
            else:
                X_full = torch.cat((X_full, splicemodel.encode_image(X.cuda())))
        if y_full1 is None:
            y_full1 = y1
        else:
            y_full1 = torch.cat((y_full1, y1))
        if y_full2 is None:
            y_full2 = y2
        else:
            y_full2 = torch.cat((y_full2, y2))

    X_test, y_test1, y_test2 = (X_full).cpu().numpy(), (y_full1).cpu().numpy(), (y_full2).cpu().numpy()

    print("Gender Linear Probe Full, Female, Male:", test_cbm(cbm_gender, X_test, y_test1))
    print("Glasses Linear Probe Full, No Glasses, Glasses:", test_cbm(cbm_glasses, X_test, y_test2))

    print("Male vs Female Probe Weights")
    for ind in np.argsort(np.abs(cbm_gender.coef_[0, :]))[:-20:-1]:
        print('\t',vocab[ind],'\t', cbm_gender.coef_[0, ind], '\t', ind, flush=True)
    print("No glasses vs Glasses Probe Weights")
    for ind in np.argsort(np.abs(cbm_glasses.coef_[0, :]))[:-20:-1]:
        print('\t',vocab[ind], '\t', cbm_glasses.coef_[0, ind], '\t', ind, flush=True)

    for index in intervening_indices:
        cbm_gender.coef_[0, index] = 0
    for index in intervening_indices:
        cbm_glasses.coef_[0, index] = 0

    print("Gender Intervened Linear Probe Full, Female, Male:", test_cbm(cbm_gender, X_test, y_test1))
    print("Glasses Intervened Linear Probe Full, No Glasses, Glasses:", test_cbm(cbm_glasses, X_test, y_test2))


    print("Male vs Female Probe Weights")
    for ind in np.argsort(np.abs(cbm_gender.coef_[0, :]))[:-20:-1]:
        print('\t',vocab[ind], '\t', cbm_gender.coef_[0, ind],'\t', ind, flush=True)
    print("No glasses vs Glasses Probe Weights")
    for ind in np.argsort(np.abs(cbm_glasses.coef_[0, :]))[:-20:-1]:
        print('\t',vocab[ind],'\t', cbm_glasses.coef_[0, ind],'\t', ind, flush=True)

def intervene_waterbirds(splicemodel, preprocess, tokenizer, vocab, intervening_indices, args):
    splicemodel.return_weights = True

    waterbirds = WaterbirdDataset(args.data_path, preprocess, train=True)
    train_loader = torch.utils.data.DataLoader(waterbirds, batch_size=512, shuffle=True)

    X_full = None
    y_full1 = None
    y_full2 = None

    for X, y1, y2 in train_loader:
        with torch.no_grad():
            if X_full is None:
                X_full = splicemodel.encode_image(X.cuda())
            else:
                X_full = torch.cat((X_full, splicemodel.encode_image(X.cuda())))
        if y_full1 is None:
            y_full1 = y1
        else:
            y_full1 = torch.cat((y_full1, y1))
        if y_full2 is None:
            y_full2 = y2
        else:
            y_full2 = torch.cat((y_full2, y2))

    waterbirds_test = WaterbirdDataset(args.data_path, preprocess, train=False)
    test_loader = torch.utils.data.DataLoader(waterbirds_test, batch_size=512, shuffle=True)

    X_train, y_train1, y_train2 = (X_full).cpu().numpy(), (y_full1).cpu().numpy(), (y_full2).cpu().numpy()

    X_full = None
    y_full1 = None
    y_full2 = None

    for X, y1, y2 in test_loader:
        with torch.no_grad():
            if X_full is None:
                X_full = splicemodel.encode_image(X.cuda())
            else:
                X_full = torch.cat((X_full, splicemodel.encode_image(X.cuda())))
        if y_full1 is None:
            y_full1 = y1
        else:
            y_full1 = torch.cat((y_full1, y1))
        if y_full2 is None:
            y_full2 = y2
        else:
            y_full2 = torch.cat((y_full2, y2))

    X_test, y_test1, y_test2 = (X_full).cpu().numpy(), (y_full1).cpu().numpy(), (y_full2).cpu().numpy()

    cbm = train_cbm(X_train, y_train1, reg = args.probe_regularization)

    print("Landbird vs Waterbird Probe Weights")
    for ind in np.argsort(np.abs(cbm.coef_[0, :]))[:-30:-1]:
        print('\t',vocab[ind],'\t', cbm.coef_[0, ind],'\t', ind, flush=True)

    # print("Train metrics")
    # print("Landbird vs Waterbird Train Full, Landbird on Land, Landbird on Water, Waterbird on Land, Waterbird on Water")
    # print(test_cbm_subgroup_accs(cbm, X_train, y_train1, y_train2))
    # print("Test metrics")
    print("Landbird vs Waterbird Test Full, Landbird on Land, Waterbird on Land, Landbird on Water, Waterbird on Water")
    print(test_cbm_subgroup_accs(cbm, X_test, y_test1, y_test2))
    
    
    for index in intervening_indices:
        cbm.coef_[0, index] = 0

    # print("Train metrics, intervened")
    # print("Landbird vs Waterbird Train Full, Landbird on Land, Landbird on Water, Waterbird on Land, Waterbird on Water")
    # print(test_cbm_subgroup_accs(cbm, X_train, y_train1, y_train2))
    # print("Test metrics, intervened")
    print("Intervened Landbird vs Waterbird Test Full, Landbird on Land, Waterbird on Land, Landbird on Water, Waterbird on Water")
    print(test_cbm_subgroup_accs(cbm, X_test, y_test1, y_test2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l1_penalty', type=float)
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    parser.add_argument('-vocab_size', type=int, default=10000)
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-probe_regularization', type=float, default=1)
    args = parser.parse_args()
    print(args)

    preprocess = splice.get_preprocess(args.model)
    tokenizer = splice.get_tokenizer(args.model)

    splicemodel = splice.load(args.model, args.vocab, args.vocab_size, args.device, l1_penalty=args.l1_penalty, return_weights=False)

    vocab = splice.get_vocabulary(args.vocab, args.vocab_size)

    if args.dataset == "CelebA":
        intervening_indices = [2756, 9194, 9287, 1660]
        print("Intervening Concepts: ", [vocab[i] for i in intervening_indices])
        intervene_celeba(splicemodel, preprocess, tokenizer, vocab, intervening_indices, args)
    elif args.dataset == "Waterbirds":
        intervening_indices = [2118, 13929, 1281, 422, 4187, 5127, 3316, 11443, 14702, 13617, 4106, 3929, 6579, 5279, 4472, 381, 2483]
        print("Intervening Concepts: ", [vocab[i] for i in intervening_indices])
        intervene_waterbirds(splicemodel, preprocess, tokenizer, vocab, intervening_indices, args)

if __name__ == "__main__":
    main()