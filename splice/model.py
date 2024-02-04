import torch
import torch.nn as nn
from sklearn import linear_model
from .admm import ADMM

class SPLICE(nn.Module):
    def __init__(self, image_mean, dictionary, clip=None, solver='skl', l1_penalty=0.01, return_weights=False, decomp_text=False, text_mean=None, device="cpu"):
        super().__init__()
        self.device = device
        self.clip = clip
        self.image_mean = image_mean.to(self.device)
        self.text_mean = text_mean.to(self.device) if text_mean else None
        self.dictionary = dictionary.to(self.device)
        self.l1_penalty = l1_penalty
        self.return_weights = return_weights
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

    def forward(self, image, text):
        image = self.encode_image(image)
        text = self.encode_text(text)
        return image, text
    
    def encode_image(self, image):
        if self.clip != None:
            self.clip.eval()
            with torch.no_grad():
                image = self.clip.encode_image(image)

        image = torch.nn.functional.normalize(image, dim=1)
        centered_image = torch.nn.functional.normalize(image-self.image_mean, dim=1)

        weights = self.decompose(centered_image)
        if self.return_weights:
            return weights

        recon_image = weights@self.dictionary
        recon_image = torch.nn.functional.normalize(recon_image, dim=1)
        recon_image = torch.nn.functional.normalize(recon_image + self.image_mean, dim=1)
        
        return recon_image
    
    def encode_text(self, text):
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

            recon_text = weights@self.dictionary
            
            recon_text = torch.nn.functional.normalize(recon_text, dim=1)
            recon_text = torch.nn.functional.normalize(recon_text + self.text_mean, dim=1)

            return recon_text
        return text
    
    def intervene_image(self, image, intervention_indices):
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

        recon_image = weights@self.dictionary
        recon_image = torch.nn.functional.normalize(recon_image, dim=1)
        recon_image = torch.nn.functional.normalize(recon_image + self.image_mean, dim=1)

        return recon_image