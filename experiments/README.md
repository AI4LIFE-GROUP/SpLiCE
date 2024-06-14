# Experiment Details:
---
## Zero Shot Accuracy and Cosine Similarity

To compute zero shot accuracy and cosine similarity, run the following code.
```
python experiments/zero_shot.py -dataset CIFAR100 -l1_penalty 0.25 -vocab laion -vocab_size 10000
```

Our experiment code supports `CIFAR10`, `CIFAR100`, [`ImageNetVal`](https://imagenetv2.org/), and [`MITStates`](https://web.mit.edu/phillipi/Public/states_and_transformations/index.html) (downloads linked) for zero-shot evaluation, with easy extension to additional datasets via the `experients/datasets.py` package.

## Spurious Correlation Discovery

To discover spurious correlations, we can compare the prevalence of a certain concept (that should be independent of class labels) between two concepts. To do so, run

```
python experiments/concept_histogram.py -l1_penalty 0.25 -dataset CIFAR100 -out_path "."
```

You can change the concepts and classes plotted in `experiments/concept_histogram.py` on lines 97 and 98. The existing code will replicate Fig. 4 of the paper out of the box.

## Intervention

To perform intervention, we explore two datasets, [`CelebA`](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [`Waterbirds`](https://github.com/kohpangwei/group_DRO?tab=readme-ov-file#waterbirds). To replicate Table 1 of the paper, run

```
python experiments/intervention.py -l1_penalty 0.2 -dataset CelebA -data_path "/my/data/path" -probe_regularization 10
```

with the proper path to CelebA in place of `/my/data/path`. This experiment considers suppressing the weights on concepts ['eyewear', 'sunglasses', 'glasses', 'visor'] and measuring classification performance on two tasks: gender and whether or not a celebrity is wearing glasses. We evaluate this task for both zero-shot classification and probing.

Next, we consider a spuriously correlated dataset, Waterbirds, where the task is to classify land and waterbirds. However, the backgrounds are spuriously correlated, so there are many pictures of waterbirds on land backgrounds (specifically forests and bamboo) and vice-versa in the training dataset. By suppressing the concepts relating to the background (various forests), we are able to improve test-time performance by removing a probe's reliance on a spurious correlation. To conduct this experiment, run

```
python experiments/intervention.py -l1_penalty 0.3 -dataset Waterbirds -data_path "/my/data/path" -probe_regularization 4 -vocab "laion_bigrams" -vocab_size 15000
```

## Retrieval

We also benchmark CLIP performance in terms of retrieval over the [`MSCOCO`](https://cocodataset.org/#download) dataset (2017 version). Due to computational complexity, we compute precision and recall statistics over `-batch_size'-sized subsets of the dataset at a time. To replicate this experiment, first embed the MSCOCO dataset using the specific clip model and library you want with the following code:

```
python experiments/embed_mscoco.py -data_path "/my/data/path/to/mscoco" -out_path "/my/data/path/to/mscoco/embeddings" -model "open_clip:ViT-B-32"
```

In addition to embedding MSCOCO, we scraped the JSON of the dataset for the image ids to make loading faster. The `mscoco_ids.txt` file is included in `/data`. After embedding MSCOCO, run the retrieval experiment with the following command.

```
python experiments/retrieval.py -data_path "/my/data/path/to/mscoco/embeddings" -l1_penalty 0.25 -model "open_clip:ViT-B-32"
```