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
            configs):
        self.configs = configs
        self.d = configs.get("d")
        self.k = configs.get("k")
        self.T = configs.get("nT")
        self.n1 = configs.get("n1")
        self.n2 = configs.get("n2")
        self.epsilon = configs.get("c")
        self.rho = configs.get("rho")
        self.sigma = configs.get("sigma")
        self.A1, self.A2, self.A3, self.A4 = configs.get("A1"), configs.get("A2"), configs.get("A3"), configs.get("A4")

        self.B = None

        self.source_tasks = []
        self.target_tasks = []

        self.source_distribution = None
        self.target_distribution = None

    def instantiate(self, num_experiments, initial=True, new_experiment:dict=None):
        """
        instantiate the simulation
        """
        self.assumptions_check()
        print(f"Experiment: A4.1 {self.A1}, A4.2 {self.A2}, A4.3 {self.A3}, A4.4 {self.A4}")
        
        self.B = self.representation()
        source_p = self.specialization(target_task=False)
        target_p = self.specialization(target_task=True)

        if not initial:
            self.B = new_experiment.get("B", self.B)
            source_p = new_experiment.get("source_p", source_p)
            target_p = new_experiment.get("target_p", target_p)

        self.generate_source_tasks(source_p)
        print(f"Source tasks generated: {len(self.source_tasks)}")
        self.generate_target_tasks(target_p, num_experiments)
        print(f"Target tasks generated: {len(self.target_tasks)}")

    def assumptions_check(self):
        configs = self.configs
        k, d, n1, n2, nT, c, delta, rho = configs.get("k"), configs.get("d"), configs.get("n1"), configs.get("n2"), configs.get("nT"), configs.get("c"), configs.get("delta"), configs.get("rho")
        C = 1.0
        n1_bound = C * rho**4 * (d + np.log(nT / delta))
        n2_bound = C * rho**4 * (k + np.log(1 / delta))
        assert 2*k <= min(d, nT), "Assumption not met: 2k <= min {d, T}"
        assert n1 > n1_bound, f"Assumption not met: n1 = {n1} >> {n1_bound: .0f} = rho^4 (d + log(T/delta))"
        assert n2 > n2_bound, f"Assumption not met: n2 = {n2} >> {n2_bound: .0f} = rho^4 (k + log(1/delta))"
        assert c * n1 >= n2, f"Assumption not met: c * n1 = {c*n1}>= {n2} = n2"
        print("Parameter assumptions checked")
        
    def generate_source_tasks(self, p):
        p = self.specialization(target_task=False)
        for i in range(self.T):
            task = Task(is_source=True)
            task.X = self.generate_X(is_source=True)
            if self.A3 or i < 10:
                task.w = p.sample()
            else:
                task.w = self.source_tasks[0].w
            task.w = normalize(task.w, p=2.0, dim=0)
            task.y = self.prediction(task.X, task.w)
            task.task_id = "source_"+str(i)
            self.source_tasks.append(task)

    def generate_target_tasks(self, p, num_experiments):
        for i in range(num_experiments):
            task = Task(is_source=False)
            task.X = self.generate_X(is_source=False)
            task.w = p.sample()
            task.w = normalize(task.w, p=2.0, dim=0)
            task.y = self.prediction(task.X, task.w)
            task.task_id = "target_"+str(i)
            self.target_tasks.append(task)

    def generate_X(self, is_source):
        """
        generate n samples of data
        rho^2-subgaussian
        following assumption 4.1 
        """
        rho = self.rho

        mean = torch.zeros(self.d)
        cov = torch.diag(torch.ones(self.d) * rho**2)
        if is_source:
            n = self.n1
            if not self.A2:
                cov[-1][-1] *= self.epsilon
        else:
            n = self.n2
            
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
    
    def specialization(self, target_task=False):
        """
        generate a specialization function for task t

        returns gaussian distribution for w_t
        """
        k = self.k
        ev = torch.ones(k)
        # if self.A3 and not target_task:
        #     ev = torch.ones(k)
        # elif not self.A3 and not target_task:
        #     ev = torch.linspace(1.0, 1.0*k, k)

        if self.A4 and target_task:
            ev /= k
        mean = torch.zeros(k)
        cov = torch.diag(ev)
        p_t = MultivariateNormal(mean, covariance_matrix=cov)
        return p_t

    def noise(self, sigma, n):
        """
        simulate gaussian noise vector of length n
        """
        return torch.normal(0, sigma, size=(n,))
    

