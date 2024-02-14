import torch
import torch.nn as nn
from sklearn import linear_model
from .admm import ADMM

class SPLICE(nn.Module):
    """Decomposes images into a sparse nonnegative linear combination of concept embeddings

    Parameters
    ----------
    image_mean : torch.tensor
        A {CLIP dimensionality} sized tensor measuring the average offset of all image embeddings for the provided CLIP backbone.
    dictionary : torch.tensor
        A {num_concepts x CLIP dimensionality} matrix used as the dictionary for the sparse nonnegative linear solver.
    clip : torch.nn.module, optional
        A CLIP backbone that implements encode_image() and encode_text(). If none, assumed that inputs to model are already CLIP embeddings (useful when working on large datasets where you don't want to forward pass through CLIP each time).
    solver : str, optional
        Either 'admm' or 'skl', by default 'skl'
    l1_penalty : float, optional
        The l1 penalty applied to the solver. Increase this for sparser solutions.
    return_weights : bool, optional
        Whether the model returns a sparse vector in {num_concepts} or the dense reconstructed embeddings, by default False
    decomp_text : bool, optional
        Whether the text encoder should also run decomposition, by default False
    text_mean : _type_, optional
        If decomposing text, a {CLIP dimensionality} sized tensor measuring the average offset of all text embeddings for the provided CLIP backbone. Only useful if decomp_text is True, by default None
    device : str, optional
        Torch device, "cuda", "cpu", etc. by default "cpu"
    """
    def __init__(self, image_mean, dictionary, clip=None, solver='skl', l1_penalty=0.01, return_weights=False, return_cosine=False, decomp_text=False, text_mean=None, device="cpu"):
        super().__init__()
        self.device = device
        self.clip = clip.to(self.device) if clip else None
        self.image_mean = image_mean.to(self.device)
        self.text_mean = text_mean.to(self.device) if text_mean else None
        self.dictionary = dictionary.to(self.device)
        self.l1_penalty = l1_penalty
        self.return_weights = return_weights
        self.return_cosine = return_cosine
        self.decomp_text = decomp_text

        if solver not in ['skl', 'admm']:
            return RuntimeError(f"Solver {solver} not supported, only \'skl\' or \'admm\'")
        self.solver = solver

        if self.solver == 'skl':
            self.l1_penalty = l1_penalty/(2*self.image_mean.shape[0]) ## skl regularization is off by a factor of 2 times the dimensionality of the CLIP embedding. See SKL docs.
        if self.solver == 'admm': 
            self.rho = 5
            self.tol = 1e-6
            self.admm = ADMM(rho=self.rho, l1_penalty=self.l1_penalty, tol=self.tol, max_iter=2000, device="cuda", verbose=False)

    def decompose(self, embedding):
        """decompose Decomposes a dense CLIP embedding into a sparse weight vector

        Parameters
        ----------
        embedding : torch.tensor
            A {batch x CLIP dimensionality} vector or batch of vectors.

        Returns
        -------
        weights : torch.tensor
            A {batch x num_concepts} sparse vector over concepts.
        """
        if self.solver == 'skl':
            clf = linear_model.Lasso(alpha=self.l1_penalty, fit_intercept=False, positive=True, max_iter=10000, tol=1e-6)
            skl_weights = []
            for i in range(embedding.shape[0]):
                clf.fit(self.dictionary.T.cpu().numpy(), embedding[i,:].cpu().numpy())
                skl_weights.append(torch.tensor(clf.coef_))
            weights = torch.stack(skl_weights, dim=0).to(self.device)
        elif self.solver == 'admm':
            weights = self.admm.fit(self.dictionary, embedding).to(self.device)
        return weights
    
    def recompose_text(self, weights):
        """recompose Converts a set of weights into a reconstructed dense embedding

        Parameters
        ----------
        weights : torch.tensor
            A {batch x num_concepts} tensor of sparse weights.

        Returns
        -------
        recon_text : torch.tensor
            A {batch x CLIP dimensionality} tensor of dense reconstructions.
        """
        recon_text = weights@self.dictionary
        recon_text = torch.nn.functional.normalize(recon_text, dim=1)
        recon_text = torch.nn.functional.normalize(recon_text + self.text_mean, dim=1)
        return recon_text

    def recompose_image(self, weights):
        """recompose Converts a set of weights into a reconstructed dense embedding

        Parameters
        ----------
        weights : torch.tensor
            A {batch x num_concepts} tensor of sparse weights.

        Returns
        -------
        recon_image : torch.tensor
            A {batch x CLIP dimensionality} tensor of dense reconstructions.
        """
        recon_image = weights@self.dictionary
        recon_image = torch.nn.functional.normalize(recon_image, dim=1)
        recon_image = torch.nn.functional.normalize(recon_image + self.image_mean, dim=1)
        return recon_image

    def forward(self, image, text):
        """forward pass through both image and text encoders.

        Parameters
        ----------
        image : torch.tensor
        text : torch.tensor

        Returns
        -------
        image : torch.tensor
            A SpLiCE embedding of the image
        text : torch.tensor
            A SpLiCE embedding of the text
        """
        image = self.encode_image(image)
        text = self.encode_text(text)
        return image, text
    
    def encode_image(self, image):
        """encode_image Encodes an image with SpLiCE.

        Parameters
        ----------
        image : torch.tensor
            A batch preprocessed images (if self.clip is not None) or CLIP embeddings to be encoded

        Returns
        -------
            If self.return_weights is True, returns the sparse weights of the images. If False, returns the dense reconstructions.
        """
        if self.clip != None:
            self.clip.eval()
            with torch.no_grad():
                image = self.clip.encode_image(image)

        image = torch.nn.functional.normalize(image, dim=1)
        centered_image = torch.nn.functional.normalize(image-self.image_mean, dim=1)

        weights = self.decompose(centered_image)

        if self.return_weights and not self.return_cosine:
            return weights

        recon_image = self.recompose_image(weights)

        if self.return_weights and self.return_cosine:
            return (weights, torch.diag(recon_image @ image.T).sum())
        
        if self.return_cosine:
            return (recon_image, torch.diag(recon_image @ image.T).sum())
        
        return recon_image
    
    def encode_text(self, text):
        """encode_text Encodes text with SpLiCE. Only runs SpLiCE decomposition if self.decomp_text is True

        Parameters
        ----------
        text : torch.tensor
            A batch tokenized text (if self.clip is not None) or CLIP embeddings to be encoded

        Returns
        -------
            If self.return_weights is True, returns the sparse weights of the text. If False, returns the dense reconstructions.
        """
        if self.clip != None:
            self.clip.eval()
            with torch.no_grad():
                text = self.clip.encode_text(text)
        
        text  = torch.nn.functional.normalize(text, dim=1)
        if self.decomp_text:
            centered_text  = torch.nn.functional.normalize(text-self.text_mean, dim=1)
            
            weights = self.decompose(centered_text)

            if self.return_weights:
                return weights

            recon_text = self.recompose_text(weights)

            return recon_text
        return text
    
    def intervene_image(self, image, intervention_indices):
        """intervene_image Encodes an image with SpLiCE and suppresses specific weights.

        Parameters
        ----------
        image : torch.tensor
            A batch preprocessed images (if self.clip is not None) or CLIP embeddings to be encoded
        intervention_indices : list
            A list of indices to set to zero when conducting the decomposition. Useful for intervention.

        Returns
        -------
            If self.return_weights is True, returns the sparse weights of the images. If False, returns the dense reconstructions.
        """
        if self.clip != None:
            self.clip.eval()
            with torch.no_grad():
                image = self.clip.encode_image(image)
        
        image = torch.nn.functional.normalize(image, dim=1)
        centered_image = torch.nn.functional.normalize(image-self.image_mean, dim=1)

        weights = self.decompose(centered_image)

        for w in intervention_indices:
            weights[:, w] *= 0
        if self.return_weights:
            return weights

        recon_image = self.recompose_image(weights)

        return recon_image