import torch
from .model import SPLICE
import os
import urllib

GITHUB_HOST_LINK = "https://raw.githubusercontent.com/AI4LIFE-GROUP/SpLiCE/main/data/"

SUPPORTED_MODELS = {
    "clip": [
        "ViT-B/32",
        "ViT-B/16",
        "RN50"
    ],
    "open_clip": [
        "ViT-B-32"
    ]
}

SUPPORTED_VOCAB = [
    "laion",
    "mscoco"
]

def available_models():
    """Returns supported models."""
    return SUPPORTED_MODELS

def _download(url: str, root: str, subfolder: str):
    """_download

    Parameters
    ----------
    url : str
        Link to download files from
    root : str
        Destination folder
    subfolder : str
        Subfolder (either /vocab or /means)

    Returns
    -------
    str
        A path to the desired file
    """
    root_subfolder = os.path.join(root, subfolder)
    os.makedirs(root_subfolder, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root_subfolder, filename)

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
    return download_target

def load(name: str, vocabulary: str, vocabulary_size: int = 10000, device = "cuda" if torch.cuda.is_available() else "cpu", download_root = None, **kwargs):
    """load SpLiCE

    Parameters
    ----------
    name : str
        the name of the CLIP backbone used
    vocabulary : str
        the vocabulary set used for dictionary learning
    device : Union[str, torch.device], optional
        torch device
    download_root : str
        path to download vocabulary and mean data to, otherwise "~/.cache/splice"
    """
    if ":" not in name:
        raise RuntimeError("Please define your CLIP backbone with the syntax \'[library]:[model]\'")

    library, model_name = name.split(":")
    if library in SUPPORTED_MODELS.keys():
        if model_name in SUPPORTED_MODELS[library]:
            if library == "clip":
                import clip
                clip_backbone, _ = clip.load(model_name, device=device)
                tokenizer = clip.tokenize
            elif library == "open_clip":
                import open_clip
                clip_backbone = open_clip.create_model(model_name, device=device, pretrained='laion2b_s34b_b79k') ##FIXME maybe allow specifying pretrained? probs not though
                tokenizer = open_clip.get_tokenizer(model_name)
            else:
                raise RuntimeError("Only CLIP and Open CLIP supported at this time. Try manual construction instead.")
        else:
            raise RuntimeError(f"Model type {model_name} not supported. Try manual construction instead.")
    else:
        raise RuntimeError(f"Library {name} not supported. Try manual construction instead.")
    
    if vocabulary in SUPPORTED_VOCAB:
        concepts = []
        vocab = []

        vocab_path = _download(os.path.join(GITHUB_HOST_LINK, "vocab", vocabulary + ".txt"), download_root or os.path.expanduser("~/.cache/splice/"), "vocab")

        concept_root = download_root or os.path.expanduser("~/.cache/splice/")
        os.makedirs(os.path.join(concept_root, "embeddings"), exist_ok=True)
        concept_path = os.path.join(concept_root, f"embeddings/{name}_{vocabulary}_{vocabulary_size}_embeddings.pt")

        if os.path.isfile(concept_path):
            concepts = torch.load(concept_path)
        else:
            with open(vocab_path, "r") as f:
                lines = f.readlines()[-vocabulary_size:]
                for line in lines:
                    line = line.strip().split(", ")[0]
                    vocab.append(line)
                    tokens = tokenizer(line).to(device)
                    with torch.no_grad():
                        concept_embedding = clip_backbone.encode_text(tokens)
                    concepts.append(concept_embedding)
            
            concepts = torch.nn.functional.normalize(torch.stack(concepts).squeeze(), dim=1)
            concepts = torch.nn.functional.normalize(concepts-torch.mean(concepts, dim=0), dim=1)
            torch.save(concepts, concept_path)

    else:
        raise RuntimeError(f"Vocabulary {vocabulary} not supported.")
    
    
    model_path = model_name.replace("/","-")
    mean_path = _download(os.path.join(GITHUB_HOST_LINK, "means", f"{library}_{model_path}_image.pt"), download_root or os.path.expanduser("~/.cache/splice/"), "means")
    image_mean = torch.load(mean_path)

    splice = SPLICE(
        image_mean=image_mean,
        dictionary=concepts,
        clip=clip_backbone,
        device=device,
        **kwargs
    )

    return splice

def get_vocabulary(name: str, vocabulary_size: int, download_root = None):
    """get_vocabulary: Gets a list of vocabulary for use in mapping sparse weight vectors to text.

    Parameters
    ----------
    name : str
        Supported vocabulary type. Either 'mscoco' or 'laion'.
    vocabulary_size : int
        Number of concepts to consider. Will consider highest frequency concepts.
    download_root : str, optional
        If specified, where to access vocab txt file from, otherwise will use default "~/.cache/splice/vocab", by default None

    Returns
    -------
    _type_
        _description_
    """
    if name in SUPPORTED_VOCAB:
        vocab_path = _download(os.path.join(GITHUB_HOST_LINK, "vocab", name + ".txt"), download_root or os.path.expanduser("~/.cache/splice/"), "vocab")

        vocab = []
        with open(vocab_path, "r") as f:
            lines = f.readlines()[-vocabulary_size:]
            vocab += [line.strip().split(", ")[0] for line in lines]
        return vocab
    else:
        raise RuntimeError(f"Vocabulary {name} not supported.")

def get_tokenizer(name: str):
    """get_tokenizer Gets tokenizer for SpLiCE model

    Parameters
    ----------
    name : str
        SpLiCE model

    Returns
    -------
    _type_
        CLIP backbone tokenizer
    """
    if ":" not in name:
        raise RuntimeError("Please define your CLIP backbone with the syntax \'[library]:[model]\'")

    library, model_name = name.split(":")
    if library in SUPPORTED_MODELS.keys():
        if model_name in SUPPORTED_MODELS[library]:
            if library == "clip":
                import clip
                return clip.tokenize
            elif library == "open_clip":
                import open_clip
                return open_clip.get_tokenizer(model_name)
            else:
                raise RuntimeError("Only CLIP and Open CLIP supported at this time. Try manual construction instead.")
        else:
            raise RuntimeError(f"Model type {model_name} not supported. Try manual construction instead.")
    else:
        raise RuntimeError(f"Library {name} not supported. Try manual construction instead.")
    
def get_preprocess(name: str):
    """get_preprocess Gets image preprocessing transform

    Parameters
    ----------
    name : str
        SpLiCE model

    Returns
    -------
    _type_
        CLIP backbone preprocessing transform.
    """
    if ":" not in name:
        raise RuntimeError("Please define your CLIP backbone with the syntax \'[library]:[model]\'")

    library, model_name = name.split(":")
    if library in SUPPORTED_MODELS.keys():
        if model_name in SUPPORTED_MODELS[library]:
            if library == "clip":
                import clip
                return clip.load(model_name)[1]
            elif library == "open_clip":
                import open_clip
                return open_clip.create_model_and_transforms(model_name)[2]
            else:
                raise RuntimeError("Only CLIP and Open CLIP supported at this time. Try manual construction instead.")
        else:
            raise RuntimeError(f"Model type {model_name} not supported. Try manual construction instead.")
    else:
        raise RuntimeError(f"Library {name} not supported. Try manual construction instead.")
    
def decompose_dataset(dataloader, splicemodel=None, device="cpu"):
    """decompose_dataset decomposes a full dataset and returns the mean weights of the sparse decomposition.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader
        Dataloader that returns (image, label) tuples for decomposition. 
    splicemodel : SPLICE
        A splicemodel instance
    device : str optional
        Torch device.
    Returns
    -------
    weights : torch.tensor
        A vector of the mean value of sparse weights over the dataset.
    """
    if splicemodel is None:
        splicemodel = load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000, l1_penalty=0.15, return_weights=True,device=device)
    splicemodel.eval()

    splicemodel.return_weights = True
    splicemodel.return_cosine = True

    ret_weights = None
    l0 = 0
    cosine = 0
    total = 0

    for data in dataloader:
        try: ## Handle dataloaders of just images or images and labels
            image, _ = data
        except:
            image = data
        image = image.to(device)

        with torch.no_grad():

            (batch_weights, batch_cosine) = splicemodel.encode_image(image)
            if ret_weights is None:
                ret_weights = torch.sum(batch_weights, dim=0)
            else:
                ret_weights += torch.sum(batch_weights, dim=0)
            
            l0 += torch.linalg.vector_norm(batch_weights, dim=1, ord=0).sum().item()
            cosine += batch_cosine.item()
            total += image.shape[0]
        
    return ret_weights/total, l0/total, cosine/total

def decompose_classes(dataloader, target_label, splicemodel=None, device="cpu"):
    """decompose_dataset decomposes a full dataset and returns the mean weights of the sparse decomposition per class.

    Parameters
    ----------
    dataloader : torch Dataloader
        Dataloader that returns (image, label) tuples for decomposition
    target_label : int optional
        Specific class label to decompose
    splicemodel : SPLICE
        A splicemodel instance
    device : str optional
        Torch device
    

    Returns
    -------
    class_weights : dict
        A dictionary of elements {label : mean sparse weight vector}
    """
    if splicemodel is None:
        splicemodel = load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000, l1_penalty=0.15, return_weights=True, return_cosine=True, device=device)
    splicemodel.eval()

    class_weights={}
    class_totals={}

    splicemodel.return_weights = True
    splicemodel.return_cosine = True

    l0 = 0
    cosine = 0
    total = 0

    for idx, (image, label) in enumerate(dataloader):

        if target_label != None:
            idx = torch.argwhere(label == target_label).squeeze()
            if idx.nelement() == 0:
                continue

            image = image[idx]
            label = label[idx]

            if idx.nelement() == 1:
                image, label = image.unsqueeze(0), label.unsqueeze(0)
            
            

        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)

            (weights, batch_cosine) = splicemodel.encode_image(image)

            for i in range(image.shape[0]):
                weights_i, label_i = weights[i], label[i].item()
                if label_i in class_weights:
                    class_weights[label_i] += weights_i
                    class_totals[label_i] += 1
                else:
                    class_weights[label_i] = weights_i
                    class_totals[label_i] = 1

            l0 += torch.linalg.vector_norm(weights, dim=1, ord=0).sum().item()
            cosine += batch_cosine.item()
            total += image.shape[0]
    
    for label in class_weights.keys():
        class_weights[label] /= class_totals[label]

    return class_weights, l0/total, cosine/total
    

def decompose_image(image, splicemodel=None, device="cpu"):
    """decompose_image _summary_

    Parameters
    ----------
    image : torch tensor
        A preprocessed image to decompose
    splicemodel : SPLICE
        A splicemodel instance
    device : str optional
        Torch device.
    """
    if splicemodel is None:
        splicemodel = load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000, l1_penalty=0.15, return_weights=True, device=device)
    splicemodel.eval()

    splicemodel.return_weights = True
    splicemodel.return_cosine = True

    (weights, cosine) = splicemodel.encode_image(image.to(device))
    l0_norm = torch.linalg.vector_norm(weights.squeeze(), ord=0).item()

    return weights, l0_norm, cosine.item()