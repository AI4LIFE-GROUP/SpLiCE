import torch
from typing import Union
from .model import SPLICE
import os
import urllib
from tqdm import tqdm

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
    root_subfolder = os.path.join(root, subfolder)
    os.makedirs(root_subfolder, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root_subfolder, filename)
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break

            output.write(buffer)
                # loop.update(len(buffer))

    return download_target

def load(name: str, vocabulary: str, vocabulary_size: int = 10000, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, **kwargs):
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

def get_tokenizer(name: str):
    """
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