import torch
    
class ADMM():
    def __init__(self, rho=1., l1_penalty=0.2, tol=1e-6, max_iter=10000, device="cuda", verbose=False):
        self.rho = rho
        self.l1_penalty = l1_penalty
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        self.verbose = verbose

    def step(self, Cb, Q_cho, z, u):
        xn = torch.cholesky_solve(2*Cb + self.rho*(z - u), Q_cho)
        zn = torch.where((xn + u - self.l1_penalty/self.rho) > 0, xn + u - self.l1_penalty/self.rho, 0)
        un = u + xn - zn
        return xn, zn, un
    
    def fit(self, C, v):
        ## iterates are in c, the number of concepts
        c = C.shape[0]

        ## size: c x c
        Q = 2*C @ C.T + (torch.eye(c)*self.rho).to(self.device)

        # factor Q for quicker solve -- this is critical.
        Q_cho = torch.linalg.cholesky(Q)

        # iterates, size: c x batch
        x = torch.randn((c, v.shape[0])).to(self.device)
        z = torch.randn((c, v.shape[0])).to(self.device)
        u = torch.randn((c, v.shape[0])).to(self.device)

        for ix in range(self.max_iter):
            z_old = z
            x, z, u = self.step(C@v.T, Q_cho, z, u)

            res_prim = torch.linalg.norm(x-z, dim=0)
            res_dual = torch.linalg.norm(self.rho*(z-z_old), dim=0)

            if (res_prim.max() < self.tol) and (res_dual.max() < self.tol):
                break
        if self.verbose:
            print("Stopping at iteration {}".format(ix))
            print("Prime Residual, r_k: {}".format(res_prim.mean()))
            print("Dual Residual, s_k: {}".format(res_dual.mean()))
        return z.T