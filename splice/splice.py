import torch
from typing import Union
from .model import SPLICE
import os

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

def load(name: str, vocabulary: str, vocabulary_size: int = 10000, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
    """load SpLiCE

    Parameters
    ----------
    name : str
        the name of the CLIP backbone used
    vocabulary : str
        the vocabulary set used for dictionary learning
    device : Union[str, torch.device], optional
        torch device
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
                clip_backbone = open_clip.create_model(model_name, device=device) ##FIXME maybe allow specifying pretrained? probs not though
                tokenizer = open_clip.get_tokenizer(model_name)
            else:
                raise RuntimeError("Only CLIP and Open CLIP supported at this time. Try manual construction instead.")
        else:
            raise RuntimeError(f"Model type {model_name} not supported. Try manual construction instead.")
    else:
        raise RuntimeError(f"Library {name} not supported. Try manual construction instead.")
    
    dirname = os.path.dirname(__file__)

    if vocabulary in SUPPORTED_VOCAB:
        concepts = []
        vocab = []

        with open(os.path.join(dirname, f"../data/vocab/{vocabulary}.txt")) as f:
            lines = f.readlines()
            for line in lines[:vocabulary_size]:
                line = line.split(", ")[0].strip()
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
    image_mean = torch.load(os.path.join(dirname, f"../data/weights/{library}_{model_path}_image.pt"))
    # image_mean = torch.load(f"../data/weights/{library}_{model_name}_image.pt")
    
    splice = SPLICE(
        image_mean=image_mean,
        dictionary=concepts,
        clip=clip_backbone,
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