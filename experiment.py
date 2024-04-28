import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import normalize

class Task():
    def __init__(self, is_source):
        self.task_id = None
        self.X = None
        self.y = None
        self.w = None
        self.is_source = is_source

class Experiment():
    def __init__(
            self, 
            input_d, 
            feature_d, 
            num_source_tasks,
            source_sample_size, 
            target_sample_size, 
            rho=1.0,
            sigma=1.0,
            assumptions=[True, True, True, True]):
        self.d = input_d
        self.k = feature_d
        self.T = num_source_tasks
        self.n1 = source_sample_size
        self.n2 = target_sample_size
        self.rho = rho
        self.sigma = sigma
        self.A1, self.A2, self.A3, self.A4 = assumptions

        self.B = None

        self.source_tasks = []
        self.target_task = None

        self.source_distribution = None
        self.target_distribution = None

    def instantiate(self):
        """
        instantiate the simulation
        """
        self.B = self.representation()
        self.source_distribution = self.specialization(target_task=False)
        self.target_distribution = self.specialization(target_task=True)
        
        for i in range(self.T):
            task = Task(is_source=True)
            task.task_id = "source_"+str(i)
            task.X = self.generate_X(self.n1, self.rho)
            w_t = self.source_distribution.sample()
            task.w = normalize(w_t, p=2.0, dim=0)
            task.y = self.prediction(task.X, task.w)
            self.source_tasks.append(task)
        
        task = Task(is_source=False)
        task.task_id = "T+1"
        task.X = self.generate_X(self.n2, self.rho)
        w_t = self.target_distribution.sample()
        task.w = normalize(w_t, p=2.0, dim=0)
        task.y = self.prediction(task.X, task.w)
        self.target_task = task

    def generate_X(self, n, rho):
        """
        generate n samples of data
        rho^2-subgaussian
        following assumption 4.1
        """
        mean = torch.zeros(self.d)
        cov = torch.diag(torch.ones(self.d) * rho**2)
        p = MultivariateNormal(mean, covariance_matrix=cov)
        X = p.sample((n,))
        return X

    def representation(self):
        """
        ground truth representation function
        """
        return torch.rand(self.d, self.k)
    
    def prediction(self, X, w):
        """
        ground truth prediction function
        """
        return X @ self.B @ w + self.noise(self.sigma, X.shape[0])
    
    def specialization(self, target_task=False, epsilon=10e-5):
        """
        generate a specialization function for task t

        returns gaussian distribution for w_t
        """
        k = self.k
        if self.A3:
            ev = np.linspace(1.0, 1.0, k)
        else:
            ev = np.linspace(1.0, 1.0*k, k)

        if not self.A2 and not target_task:
            ev[-1] = epsilon

        if self.A4 and target_task:
            ev /= k
        
        Q, _ = torch.linalg.qr(torch.randn(k, k))

        mean = torch.zeros(k)
        cov = Q @ torch.diag(torch.tensor(ev.astype(np.float32))) @ Q.T

        p_t = MultivariateNormal(mean, covariance_matrix=cov)
        return p_t

    def noise(self, sigma, n):
        """
        simulate gaussian noise vector of length n
        """
        return torch.normal(0, sigma, size=(n,))