# SpLiCE
### Sparse Linear Concept Embeddings
---
[paper](https://alexoesterling.com)

To install SpLiCE, run the following

```
git clone git@github.com:AI4LIFE-GROUP/SpLiCE.git
cd SpLiCE
pip install .
```

Then, to load an existing SpLiCE model, use the `splice.load` function. In the following example, we load a SpLiCE model built on [open clip]'s ViT-B-32 implementation, and construct a concept dictionary out of the top 10000 tokens present in [LAION] captions. We use an l1 penalty of 0.15 when conducting sparse dictionary learning, and return the dense reconstructed embedding.

```
import splice

splicemodel = splice.load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000, l1_penalty=0.15,device="cuda")
```

A splice model has four primary functions. `encode_image`, `encode_text`, `decompose`, and `recompose`.
 * `encode_image` and `encode_text` function the same as a normal CLIP model, but will return sparse weight vectors if the flag `return_weights` is set to `True`. For `encode_text`, it will only decompose and return a sparse weight vector or reconstruction if the flag `decomp_text` is set to `True` when constructing a SpLiCE model and a `text_mean` is provided.
 * `decompose` takes in a dense CLIP embedding and outputs a sparse weight vector representation
 * `recompose` takes in a sparse weight vector and outputs a dense reconstruction. Use this in combination with `return_weights=True` to get both sparse representations and dense reconstructions. For example,

```
import splice

splicemodel = splice.load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000, l1_penalty=0.15, return_weights=True, device="cuda")

embedding = torch.randn(512)
sparse_weights = splicemodel.encode_image(embedding)
reconstruction = splicemodel.recompose(sparse_weights)
```

The sparse weights can be mapped back to text using the `splice.get_vocabulary` function. For example, the following lines will take a random embedding, get its sparse representation, and then print the sorted vocabulary and their corresponding weights.

```
import splice

splicemodel = splice.load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000, l1_penalty=0.15, return_weights=True, device="cuda")

embedding = torch.randn(512)
sparse_weights = splicemodel.encode_image(embedding)
vocabulary = splice.get_vocabulary("laion", 10000)

for weight_idx in torch.sort(sparse_weights, descending=True)[1]:
    print(f"{vocabulary[weight_idx]}: {sparse_weights[weight_idx]}")
```

If you want to construct your own splice model built on your own VLM backbone or vocabulary, you can directly construct your own splice model. All you need to provide is a module that implements `encode_image`, and `encode_text`, an estimated `image_mean` over that module, and a `dictionary` of text embeddings.

```
import splice

vlm_backbone = VLMBackbone()
image_mean = torch.load('path/to/image_mean.pt')
vocab_path = 'path/to/vocab.txt'

concepts = []
with open(vocab_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        concepts.append(vlm_backbone.encode_text(line))
concepts = torch.nn.functional.normalize(torch.stack(concepts), dim=1)
concepts = torch.nn.functional.normalize(concepts-torch.mean(concepts, dim=0), dim=1)

splicemodel = SPLICE(image_mean, concepts, clip=vlm_backbone, device="cuda)
```






python decompose_image.py -path "/n/home04/aoesterling/quantizing/000000308175.jpg" --out_path "aa.txt" --l1_penalty 0.25 -device "cuda" --verbose
python concept_histogram.py -dataset "CIFAR10" --out_path "b.txt" --verbose --l1_penalty 0.2 --class_label 2 -device "cuda"