import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import hydra
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import wandb
import os
from scipy.special import logsumexp
os.environ['HYDRA_FULL_ERROR']='1'

#TODO: add vanilla classifier to compare with
#TODO: fix plot for class prob 
#TODO: add regularizer (KL) to loss 

class QuadraticHead(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, device, use_linear_term, use_alpha_gamma=False):
        super(QuadraticHead, self).__init__()

        self.input_dim = input_dim + 1
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_linear_term = use_linear_term
        self.use_alpha_gamma = use_alpha_gamma
        
        if self.use_alpha_gamma:
            self.input_dim += 2

        self.quadratic = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.hidden_dim//2),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim//2, self.hidden_dim//4),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim//4, 1, bias=False))
        
        if self.use_linear_term:
            self.linear = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.hidden_dim*2),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim*2, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.hidden_dim//2),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim//2, self.hidden_dim//4),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim//4, 1, bias=False))
        
        self.bias = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.hidden_dim//4),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim//4, 1, bias=False))

        self.quadratic.train(True)
        self.bias.train(True)

        if self.use_linear_term:
            self.linear.train(True)

    def forward(self, t, alpha=None, gamma=None):
        t_sq = t**2
        if self.use_alpha_gamma:
            x = torch.cat([t.unsqueeze(-1), t_sq.unsqueeze(-1), alpha.unsqueeze(-1), gamma.unsqueeze(-1)], dim=-1).to(self.device)
        else:
            x = torch.cat([t.unsqueeze(-1), t_sq.unsqueeze(-1)], dim=-1).to(self.device)

        A2 = self.quadratic(x)
        if self.use_linear_term:
            A1 = self.linear(x)
        b = self.bias(x)
        #1d for now 
        if self.use_linear_term:
            return A2, A1, b
        else:
            return A2, b
        

class Classifier(torch.nn.Module):
    def __init__(self, input_dim, device, use_linear_term):
        super(Classifier, self).__init__()

        self.input_dim = input_dim
        self.device = device
        self.use_linear_term = use_linear_term

        self.quadratic = nn.Parameter(torch.randn(self.input_dim, 1), requires_grad=True)
        if self.use_linear_term:
            self.linear = nn.Parameter(torch.randn(self.input_dim, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1),requires_grad=True)
    
    def forward(self):
        if self.use_linear_term:
            return self.quadratic, self.linear, self.bias
        else:
            return self.quadratic, self.bias



class SDE:
    def __init__(self):
        self.beta_0 = 0.1
        self.beta_1 = 20

    def int_beta_fn(self, t):      
        return self.beta_0 * t + (
                self.beta_1 - self.beta_0
            ) * (t**2 / 2)
    

    def alpha(self, t):
        a = torch.exp(-.5*self.int_beta_fn(t))
        return a
    
    def gamma(self, t):
        b = 1-torch.exp(-self.int_beta_fn(t))
        return b
    
    def transition_mean_coefficient(self, t):
        beta_max = self.beta_1
        beta_min = self.beta_0
        int_beta_t = self.int_beta_fn(t)
        coef = torch.exp(-0.5 * int_beta_t)
        return coef

    def transition_mean_coefficient_like(self, x, t):
        coef = self.transition_mean_coefficient(t)
        coef = torch.mul(torch.ones_like(x), coef)
        assert x.shape==coef.shape
        return coef

    def transition_mean(self, x, t):
        coef = self.transition_mean_coefficient_like(x, t)
        mu = x * coef
        return mu

    def transition_var(self, t):
        int_beta_t = self.int_beta_fn(t)
        assert int_beta_t.shape==t.shape
        var = 1 - torch.exp(-int_beta_t)
        return var

    def sample_from_transition_kernel(self, x, t, eval=False):
        mean = self.transition_mean(x, t)
        var = self.transition_var(t)
        std = torch.sqrt(var)
        std = torch.ones_like(x)*std

        assert x.shape==std.shape
        eps = torch.randn_like(x)
        xt = mean + eps * std
        return xt


class Workspace:
    def __init__(self, cfg):

        if cfg.use_wandb:
            exp_name = 'dre'
            wandb.init(project="dre", group='1', name=exp_name)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.global_step = 0
        self.p = torch.distributions.Normal(loc=cfg.mu_p, scale=cfg.sigma_p)
        self.q = torch.distributions.Normal(loc=cfg.mu_q, scale=cfg.sigma_q)
        self.samples_p = self.p.sample(sample_shape=torch.Size([cfg.data_size//2]))
        self.samples_q = self.q.sample(sample_shape=torch.Size([cfg.data_size//2]))
        y_p = torch.ones(cfg.data_size//2)
        y_q = torch.zeros(cfg.data_size//2)
        train_dataset1 = torch.utils.data.TensorDataset(self.samples_p , y_p)
        train_dataset2 = torch.utils.data.TensorDataset(self.samples_q , y_q)

        # class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
        # print('class_sample_count', class_sample_count)
        # weight = 1. / class_sample_count
        # print('weight', weight)
        
        # samples_weight = np.array([weight[t] for t in y])
        # samples_weight = torch.from_numpy(samples_weight)
        # sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        print('mu_p', cfg.mu_p)
        self.loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=cfg.batch_size, shuffle=True)
        self.loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=cfg.batch_size, shuffle=True)

        self.data_iter1 = iter(self.loader1)
        self.data_iter2 = iter(self.loader2)
        self.eps = cfg.eps
        self.sde = SDE()

        if (self.cfg.mu_p!=0. or self.cfg.mu_q!=0.) and (self.cfg.sigma_p != self.cfg.sigma_q) :
            print('mu_p is not 0, using linear term', self.cfg.mu_p)
            print('mu_q is not 0, using linear term', self.cfg.mu_q)
            linear = True
        else:
            linear = False

        if self.cfg.single_classifier is False:
            self.model = QuadraticHead(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, device=self.device, use_linear_term=linear, use_alpha_gamma=cfg.use_alpha_gamma).to(cfg.device)
        else:
            self.model = Classifier(input_dim=cfg.input_dim, device=self.device, use_linear_term=linear).to(cfg.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.error = np.zeros((4, 1000,))
        self.loss_t = np.empty((1,))
        self.count = 0
        self.log_nu = 0 

    def log_density_ratio_gaussians(self, x, mu_p, sigma_p, mu_q, sigma_q):

        r_p = (x - mu_p) / sigma_p
        r_q = (x - mu_q) / sigma_q

        return np.log(sigma_q) - np.log(sigma_p) + .5 * (r_q**2 - r_p**2)

    # def plot(self, global_step, t, tre, loss):

        # samples = torch.cat([x_p, x_q])[:,None]
        # print('samples', samples)
        # y = torch.cat([torch.zeros(x_p.shape[0]), torch.ones(x_q.shape[0])]).to(self.cfg.device)
        # print('y', y)
        # samples_y = torch.cat([samples, y[:,None]], dim=-1).to(self.cfg.device)
        # print('samples_y', samples_y)
        # samples_y = samples_y[torch.randperm(samples_y.size()[0])]
        # samples = samples_y[:,0]
        # y = samples_y[:,1]
        # print('samples', samples)
        # r_p = (samples.detach().cpu().numpy() - mu_p) / sigma_p
        # r_q = (samples.detach().cpu().numpy() - mu_q) / sigma_q
        #1/2 * (log(2)) - log(sigma_p) + 1/2*log(sigma_q**2 - sigma_p**2) - log(sigma_q)
        # theta_true = 1/(2*sigma_q**2)-1/(2*sigma_p**2)
        
        # log_ratios = self.log_density_ratio_gaussians(samples.detach().cpu().numpy(), mu_p, sigma_p, mu_q, sigma_q)
        

        # if self.global_step % 1000 == 0:
        #     plt.clf()
        #     fig, ax = plt.subplots(figsize=(8, 5))
        #     ax.set_title('density ratio')
        #     ax.plot(samples, log_ratios, label='analytical')
        #     #ax.plot(mesh.detach().cpu().numpy(), log_ratios.cpu().numpy(), label='analytical')
        #     ax.plot(samples, log_odds, label='Estimated')
        #     ax.set_xlim(-5.5, 5.5)
        #     ax.set_xlabel('$x$')
        #     #ax.set_ylim(0., 1.)
        #     ax.set_ylabel('$log r(x)$')
        #     ax.legend()
        #     plt.savefig(f"./{global_step}_log_dr.png")
        #     wandb.save(f"./{global_step}_log_dr.png")


    def log_sigmoid(self, x):
        return torch.clamp(x, max=0) - torch.log(torch.exp(-torch.abs(x)) + 1) + 0.5 * torch.clamp(x, min=0, max=0)

    def loss(self, h, y):
        term1 = h[y==1.]
        term2 = h[y==0.]
        return -torch.mean(torch.log(term1))- torch.mean(torch.log(1-term2))
    

    def eval(self, global_step):
        self.model.eval()
        if self.cfg.single_classifier is False:
            t = torch.linspace(0.,1.,1000).to(self.cfg.device)
        else:
            t = torch.tensor([0]).to(self.cfg.device)
        

        for i, it in enumerate(t):
            sum_p=0
            sum_q=0
            sq_sum_p=0
            sq_sum_q=0
            for ix in range(10):
                try:
                    x, y = next(self.data_iter1)
                    x1, y1 = next(self.data_iter2)
                except StopIteration:
                    self.data_iter1 = iter(self.loader1)
                    self.data_iter2 = iter(self.loader2)
                    x, y = next(self.data_iter1)
                    x1, y1 = next(self.data_iter2)

                x = torch.cat((x,x1),0)
                y = torch.cat((y,y1),0)
                rand = torch.randperm(x.shape[0])
                x = x[rand]
                y = y[rand]
                x = x.reshape((x.shape[0],1)).to(self.cfg.device)
                y = y.reshape((y.shape[0],1)).to(self.cfg.device)

                xt= self.sde.sample_from_transition_kernel(x.squeeze(1), it, eval=True).unsqueeze(-1)
                x_p = xt[y==1.]
                x_q = xt[y==0.]
                sum_p += torch.mean(x_p).detach().cpu().numpy()
                sum_q += torch.mean(x_q).detach().cpu().numpy()
                sq_sum_p += torch.mean(x_p**2).detach().cpu().numpy()
                sq_sum_q += torch.mean(x_q**2).detach().cpu().numpy()

            mu_p = sum_p / 10
            mu_q = sum_q / 10
            sigma_p = (sq_sum_p/10 - mu_p**2)**.5
            sigma_q = (sq_sum_q/10 - mu_q**2)**.5
            r = (sigma_p**2)/(sigma_q**2)
            # print('mu_p', mu_p)
            # print('mu_q', mu_q)
            # print('sigma_p', sigma_p)
            # print('sigma_q', sigma_q)


            theta_true2 = -1/(2*(self.sde.gamma(it)+(self.sde.alpha(it)**2)*(sigma_p**2))) + 1/(2*(self.sde.gamma(it)+(self.sde.alpha(it)**2)*(sigma_q**2)))
            theta_true1 = 2*(self.sde.alpha(it)*self.cfg.mu_p)/(2*(self.sde.gamma(it)+(self.sde.alpha(it)**2)*(sigma_p**2))) - (self.sde.alpha(it)*self.cfg.mu_q)/(2*(self.sde.gamma(it)+(self.sde.alpha(it)**2)*(sigma_q**2)))
            theta_true0 = np.log(sigma_q) - np.log(sigma_p) -(self.sde.alpha(it)*self.cfg.mu_p)**2/(2*(self.sde.gamma(it)+(self.sde.alpha(it)**2)*(sigma_p**2))) + (self.sde.alpha(it)*self.cfg.mu_q)/(2*(self.sde.gamma(it)+(self.sde.alpha(it)**2)*(sigma_q**2)))
            log_ratios = theta_true0 + theta_true1*xt + theta_true2*(xt**2)

            with torch.no_grad():
                if self.cfg.mu_p == 0. and self.cfg.mu_q == 0.:

                    if self.cfg.use_alpha_gamma:
                        theta_est2, theta_est0 = self.model(it, alpha = self.sde.alpha(it), gamma = self.sde.gamma(it))
                    else:
                        theta_est2, theta_est0 = self.model(it)

                    log_odds = theta_est0 - torch.exp(theta_est2)*(xt**2)
                    print('theta2 true', theta_true2)
                    print('theta2 est', -torch.exp(theta_est2))
                    print('theta0 true', theta_true0)
                    print('theta0 est', theta_est0)
                elif self.cfg.sigma_p == self.cfg.sigma_q:

                    if self.cfg.use_alpha_gamma:
                        theta_est1, theta_est0 = self.model(it, alpha = self.sde.alpha(it), gamma = self.sde.gamma(it))
                    else:
                        theta_est1, theta_est0 = self.model(it)

                    if self.cfg.theta1_use_exp:
                        log_odds = theta_est0 + torch.exp(theta_est1)*xt
                    else:
                        log_odds = theta_est0 + theta_est1*xt
                    print('theta1 true', theta_true1)
                    if self.cfg.theta1_use_exp:
                        print('theta1 est', torch.exp(theta_est1))
                    else:
                        print('theta1 est', theta_est1)
                    print('theta0 true', theta_true0)
                    print('theta0 est', theta_est0)


                else:
                    if self.cfg.use_alpha_gamma:
                        theta_est2, theta_est1, theta_est0 = self.model(it, alpha = self.sde.alpha(it), gamma = self.sde.gamma(it))
                    else:
                        theta_est2, theta_est1, theta_est0 = self.model(it)
                    if self.cfg.theta1_use_exp:
                        log_odds = theta_est0 + torch.exp(theta_est1)*xt -torch.exp(theta_est2)*(xt**2)
                    else:
                        log_odds = theta_est0 + theta_est1*xt -torch.exp(theta_est2)*(xt**2)
            
                    print('theta2 true', theta_true2)
                    print('theta2 est', -torch.exp(theta_est2))
                    print('theta1 true', theta_true1)
                    if self.cfg.theta1_use_exp:
                        print('theta1 est', torch.exp(theta_est1))
                    else:
                        print('theta1 est', theta_est1)
                    print('theta0 true', theta_true0)
                    print('theta0 est', theta_est0)
        
            evaluated = np.concatenate([xt.detach().cpu().numpy(), log_ratios.detach().cpu().numpy(), log_odds.detach().cpu().numpy()], axis=-1)
            evaluated = evaluated[evaluated[:,0].argsort()]
            x = evaluated[:,0]
            log_ratios = evaluated[:,1]
            log_odds = evaluated[:,2]

            err = np.log(sum(abs(log_odds - log_ratios)))
            self.error[0,i] = err

            if self.cfg.sigma_p != self.cfg.sigma_q:
                err_theta2 = abs(torch.log(-theta_true2) - theta_est2)
                self.error[1,i] = err_theta2

            if self.cfg.mu_p != 0. or self.cfg.mu_q != 0.:
                if self.cfg.theta1_use_exp:
                    err_theta1 = abs(torch.log(theta_true1) - theta_est1)
                else:
                    err_theta1 = abs(theta_true1 - theta_est1)
                print('err theta1', err_theta1)
                self.error[2,i] = err_theta1

            err_theta0 = abs(theta_true0 - theta_est0)
            self.error[3,i] = err_theta0

            if i%50==0:
                plt.clf()
                fig, ax = plt.subplots(figsize=(8, 5))
                plt.plot(np.linspace(-5,8, log_ratios.shape[0]), log_ratios, label='true ratio')
                plt.plot(np.linspace(-5,8, log_odds.shape[0]), log_odds, label='estimated ratio')
                ax.set_xlabel('$x$')

                #ax.set_ylim(0., 1.)
                ax.set_ylabel('$log ratio$')
                if self.cfg.mu_p == 0. and self.cfg.mu_q == 0.:
                    ax.set_title(f"err_theta2{err_theta2.detach().cpu().numpy().round(3)}, err_theta0{err_theta0.detach().cpu().numpy().round(3)}")
                elif self.cfg.sigma_p == self.cfg.sigma_q:
                    ax.set_title(f"err_theta1{err_theta1.detach().cpu().numpy().round(3)}, err_theta0{err_theta0.detach().cpu().numpy().round(3)}")
                else:
                    ax.set_title(f"err_theta2{err_theta2.detach().cpu().numpy().round(3)}, err_theta1{err_theta1.detach().cpu().numpy().round(3)}, err_theta0{err_theta0.detach().cpu().numpy().round(3)}")


                ax.legend()

                plt.savefig(f"./{global_step}_it{it}_log_ratio.png")
                wandb.save(f"./{global_step}_it{it}_log_ratio.png")

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 5))
        prob = torch.sigmoid(torch.tensor(log_odds))
        # print('x', x)
        # print('prob', prob)
        plt.scatter(x, prob.detach().cpu().numpy(), c=y.detach().cpu().numpy(), s=500)
        ax.set_xlabel('$x$')

        #ax.set_ylim(0., 1.)
        ax.set_ylabel('$p(y=1|x)$')

        ax.legend()

        plt.savefig(f"./{global_step}_class_prob.png")
        wandb.save(f"./{global_step}_class_prob.png")

        

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.set_title('error over time')
        # print('err', err)
        ax.plot(np.linspace(0,1,self.error.shape[1]), self.error[0], label='log error')
        ax.legend(np.linspace(0,1,self.error.shape[1])[::-1], title='$t$')
        ax.set_xlabel('$t$')

        #ax.set_ylim(0., 1.)
        ax.set_ylabel('$log_error$')

        # ax.legend()

        plt.savefig(f"./{global_step}_error_log_dr.png")
        wandb.save(f"./{global_step}_error_log_dr.png")

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('error over time')
        # print('err', err)
        if self.cfg.sigma_p != self.cfg.sigma_q:
            ax.plot(np.linspace(0,1,self.error.shape[1]), self.error[1], label='theta2 error')
        if self.cfg.mu_p != 0. or self.cfg.mu_q != 0.:
            ax.plot(np.linspace(0,1,self.error.shape[1]), self.error[2], label='theta1 error')

        ax.plot(np.linspace(0,1,self.error.shape[1]), self.error[3], label='thetha0 error')
        ax.legend(np.linspace(0,1,self.error.shape[1])[::-1], title='$t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$abs_error$')
        # ax.legend()
        plt.savefig(f"./{global_step}_coeff_errors.png")
        wandb.save(f"./{global_step}_coeff_errors.png")

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('loss over time')
        tmp_array = np.array([[np.linspace(0,global_step,self.loss_t.shape[0]), self.loss_t]]).squeeze(0)
        print('tmp_array', tmp_array.shape)
        print('self.loss_t', self.loss_t)
        tmp_array = tmp_array[:,tmp_array[0].argsort()[::-1]]
        ax.plot(tmp_array[0], tmp_array[1], label='loss')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$log loss$')
        ax.legend()
        plt.savefig(f"./{global_step}_loss.png")
        wandb.save(f"./{global_step}_loss.png")

        self.model.train()
    
    def eval_single(self):
        self.model.eval()

        sigma_p = self.cfg.sigma_p
        sigma_q = self.cfg.sigma_q
        mu_p = self.cfg.mu_p
        mu_q = self.cfg.mu_q
        r = (sigma_p**2)/(sigma_q**2)
        print('mu_p', mu_p)
        print('mu_q', mu_q)
        print('sigma_p', sigma_p)
        print('sigma_q', sigma_q)

        try:
            x, y = next(self.data_iter1)
            x1, y1 = next(self.data_iter2)
        except StopIteration:
            self.data_iter1 = iter(self.loader1)
            self.data_iter2 = iter(self.loader2)
            x, y = next(self.data_iter1)
            x1, y1 = next(self.data_iter2)

        x = torch.cat((x,x1),0)
        y = torch.cat((y,y1),0)
        rand = torch.randperm(x.shape[0])
        x = x[rand]
        y = y[rand]
        x = x.reshape((x.shape[0],1)).to(self.cfg.device)
        y = y.reshape((y.shape[0],1)).to(self.cfg.device)

        theta_true2 = 1/(2*sigma_q**2)- 1/(2*sigma_p**2)
        theta_true1 = 2*(-mu_q/(2*sigma_q**2)+ mu_p/(2*sigma_p**2))
        theta_true0 = np.log(sigma_q) - np.log(sigma_p) + mu_q**2/(2*sigma_q**2)- mu_p**2/(2*sigma_p**2)
        log_ratios = theta_true0 + theta_true1*x + theta_true2*(x**2)

        with torch.no_grad():
            if self.cfg.mu_p == 0. and self.cfg.mu_q == 0.:
                theta_est2, theta_est0 = self.model()
                log_odds = theta_est0 - torch.exp(theta_est2)*(x**2)
                print('theta2 true single', theta_true2)
                print('theta2 est single', -torch.exp(theta_est2))
                print('theta0 true single', theta_true0)
                print('theta0 est single', theta_est0)
            
            elif self.cfg.sigma_p == self.cfg.sigma_q:
                theta_est1, theta_est0 = self.model()
                if self.cfg.theta1_use_exp:
                    log_odds = theta_est0 + torch.exp(theta_est1)*x
                else:
                    log_odds = theta_est0 +theta_est1*(x)
                print('theta1 true single', theta_true1)
                print('theta1 est single', theta_est1)
                print('theta0 true single', theta_true0)
                print('theta0 est single', theta_est0)

            else:
                theta_est2, theta_est1, theta_est0 = self.model()

                if self.cfg.theta1_use_exp:
                    
                    log_odds = theta_est0 + torch.exp(theta_est1)*x - torch.exp(theta_est2)*(x**2)
                else:
                    log_odds = theta_est0 + theta_est1*x - torch.exp(theta_est2)*(x**2)
        
                print('theta2 true single', theta_true2)
                print('theta2 est single', -torch.exp(theta_est2))
                print('theta1 true single', theta_true1)
                if self.cfg.theta1_use_exp:
                    print('theta1 est single', torch.exp(theta_est1))
                else:
                    print('theta1 est single', theta_est1)
                print('theta0 true single', theta_true0)
                print('theta0 est single', theta_est0)
    
        evaluated = np.concatenate([x.detach().cpu().numpy(), log_ratios.detach().cpu().numpy(), log_odds.detach().cpu().numpy()], axis=-1)
        evaluated = evaluated[evaluated[:,0].argsort()]
        x = evaluated[:,0]
        log_ratios = evaluated[:,1]
        log_odds = evaluated[:,2]

        err = np.log(sum(abs(log_odds - log_ratios)))
        self.error[0,self.count] = err

        err_theta2 = abs(torch.log(theta_true2) - theta_est2)
        self.error[1,self.count] = err_theta2

        if self.cfg.mu_p != 0. or self.cfg.mu_q != 0.:
            err_theta1 = abs(torch.log(theta_true1) - theta_est1)
            self.error[2,self.count] = err_theta1

        err_theta0 = abs(theta_true0 - theta_est0)
        self.error[3,self.count] = err_theta0
        self.count+=1


        if self.global_step>=self.cfg.total_steps-1:
            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.plot(np.linspace(-5,8, log_ratios.shape[0]), log_ratios, label='true ratio')
            plt.plot(np.linspace(-5,8, log_odds.shape[0]), log_odds, label='estimated ratio')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$log ratio$')
            if self.cfg.mu_p == 0. and self.cfg.mu_q == 0.:
                ax.set_title(f"err_theta2{err_theta2.detach().cpu().numpy().round(3)}, err_theta0{err_theta0.detach().cpu().numpy().round(3)}")
            else:
                ax.set_title(f"err_theta2{err_theta2.detach().cpu().numpy().round(3)}, err_theta1{err_theta1.detach().cpu().numpy().round(3)}, err_theta0{err_theta0.detach().cpu().numpy().round(3)}")
            ax.legend()
            plt.savefig(f"./{self.global_step}_log_ratio.png")
            wandb.save(f"./{self.global_step}_log_ratio.png")


            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 5))
            prob = torch.sigmoid(torch.tensor(log_odds))
            print('x', x)
            print('prob', prob)
            plt.scatter(x, prob.detach().cpu().numpy(), c=y.detach().cpu().numpy(), s=500)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$p(y=1|x)$')
            ax.legend()
            plt.savefig(f"./{self.global_step}_class_prob.png")
            wandb.save(f"./{self.global_step}_class_prob.png")

            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title('error over time')
            print('err', err)
            ax.plot(np.linspace(0,1,self.error.shape[1]), self.error[0], label='log error')
            ax.legend(np.linspace(0,1,self.error.shape[1])[::-1], title='$t$')
            ax.set_xlabel('$t$')
            ax.set_ylabel('$log_error$')
            ax.legend()
            plt.savefig(f"./{self.global_step}_error_log_dr.png")
            wandb.save(f"./{self.global_step}_error_log_dr.png")

            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title('error over time')
            print('err', err)
            ax.plot(np.linspace(0,1,self.error.shape[1]), self.error[1], label='theta2 error')
            if self.cfg.mu_p != 0. or self.cfg.mu_q != 0.:
                ax.plot(np.linspace(0,1,self.error.shape[1]), self.error[2], label='theta1 error')
            ax.plot(np.linspace(0,1,self.error.shape[1]), self.error[3], label='thetha0 error')
            ax.legend(np.linspace(0,1,self.error.shape[1])[::-1], title='$t$')
            ax.set_xlabel('$t$')
            ax.set_ylabel('$abs_error$')
            ax.legend()
            plt.savefig(f"./{self.global_step}_coeff_errors.png")
            wandb.save(f"./{self.global_step}_coeff_errors.png")


            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title('loss over time')


            ax.plot(np.linspace(0,1,self.loss_t.shape[0]), self.loss_t, label='loss')
            # ax.legend(np.linspace(0,1,self.error.shape[1])[::-1], title='$t$')
            ax.set_xlabel('$step$')
            ax.set_ylabel('$log loss$')
            ax.legend()
            plt.savefig(f"./{self.global_step}_loss.png")
            wandb.save(f"./{self.global_step}_loss.png")

        self.model.train()

    def train(self):

        while self.global_step < self.cfg.total_steps:
            try:
                x, y = next(self.data_iter1)
                x1, y1 = next(self.data_iter2)
            except StopIteration:
                self.data_iter1 = iter(self.loader1)
                self.data_iter2 = iter(self.loader2)
                x, y = next(self.data_iter1)
                x1, y1 = next(self.data_iter2)
            x = torch.cat((x,x1),0)
            y = torch.cat((y,y1),0)
            rand = torch.randperm(x.shape[0])
            x = x[rand]
            y = y[rand]
            x = x.reshape((x.shape[0],1)).to(self.cfg.device)
            y = y.reshape((y.shape[0],1)).to(self.cfg.device)

            if self.global_step < 1:
                plt.clf()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.set_title('samples')
                ax.scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label='samples', marker="o")
                plt.savefig(f"./samples.png")
                wandb.save(f"./samples.png")
            

            if self.cfg.single_classifier is False:
                if self.cfg.random_sample:
                    #TODO: different t's in each batch sample (not just one t for the whole batch)
                    t = torch.rand(size=(x.shape[0],)).to(self.cfg.device)
                    t1 = torch.rand(size=(x.shape[0],)).to(self.cfg.device)
                elif self.cfg.curriculum:
                    t = torch.clip(torch.pow(torch.tensor([(1 - self.global_step / self.cfg.total_steps)*(1-self.eps)]), self.cfg.power_scale).to(self.cfg.device), 0, .99)

                xt= self.sde.sample_from_transition_kernel(x.squeeze(1), t)
                if self.cfg.random_sample:
                    xt1= self.sde.sample_from_transition_kernel(x.squeeze(1), t1)
            else:
                t = None
                xt = x

            if self.cfg.mu_p == 0. and self.cfg.mu_q == 0.:

                if self.cfg.use_alpha_gamma:
                    theta_est2, theta_est0 = self.model(t, alpha = self.sde.alpha(t), gamma = self.sde.gamma(t))
                    if self.cfg.regularize:
                        theta_est2_1, theta_est0_1 = self.model(t1, alpha = self.sde.alpha(t1), gamma = self.sde.gamma(t1))

                elif self.cfg.single_classifier is False:
                    theta_est2, theta_est0 = self.model(t)
                    if self.cfg.regularize:
                        theta_est2_1, theta_est0_1 = self.model(t1)
                else:
                    theta_est2, theta_est0 = self.model()
                
                log_odds = theta_est0 - torch.exp(theta_est2)*(xt.unsqueeze(-1)**2)
                if self.cfg.regularize:
                    log_odds_1 = theta_est0_1 - torch.exp(theta_est2_1)*(xt1.unsqueeze(1)**2)

            elif self.cfg.sigma_p == self.cfg.sigma_q:

                if self.cfg.use_alpha_gamma:
                    theta_est1, theta_est0 = self.model(t, alpha = self.sde.alpha(t), gamma = self.sde.gamma(t))
                    if self.cfg.regularize:
                        theta_est1_1, theta_est0_1 = self.model(t1, alpha = self.sde.alpha(t1), gamma = self.sde.gamma(t1))

                elif self.cfg.single_classifier is False:
                    theta_est1, theta_est0 = self.model(t)
                    if self.cfg.regularize:
                        theta_est1_1, theta_est0_1 = self.model(t1)
                else:
                    theta_est1, theta_est0 = self.model()
                
                if self.cfg.theta1_use_exp:
                    log_odds = theta_est0 + torch.exp(theta_est1)*(xt.unsqueeze(1))
                else:
                    log_odds = theta_est0 + theta_est1*(xt.unsqueeze(1))

                if self.cfg.regularize:
                    if self.cfg.theta1_use_exp:
                        log_odds_1 = theta_est0_1 + torch.exp(theta_est1)*(xt.unsqueeze(1))
                    else:
                        log_odds_1 = theta_est0_1 + theta_est1*(xt.unsqueeze(1))
            else:
                if self.cfg.use_alpha_gamma:
                    theta_est2, theta_est1, theta_est0 = self.model(t, alpha = self.sde.alpha(t), gamma = self.sde.gamma(t))
                    if self.cfg.regularize:
                        theta_est2_1, theta_est1_1, theta_est0_1 = self.model(t1, alpha = self.sde.alpha(t1), gamma = self.sde.gamma(t1))
                elif self.cfg.single_classifier is False:
                    theta_est2, theta_est1, theta_est0 = self.model(t)
                    if self.cfg.regularize:
                        theta_est2_1, theta_est1_1, theta_est0_1 = self.model(t1)
                else:
                    theta_est2, theta_est1, theta_est0 = self.model()

                if self.cfg.theta1_use_exp:
                    log_odds = theta_est0 + torch.exp(theta_est1)*xt.unsqueeze(1) - torch.exp(theta_est2)*(xt.unsqueeze(1)**2)
                else:
                    log_odds = theta_est0 + theta_est1*xt.unsqueeze(1) - theta_est2*(xt.unsqueeze(1)**2)

                if self.cfg.regularize:
                    if self.cfg.theta1_use_exp:
                        log_odds_1 = theta_est0_1 + torch.exp(theta_est1_1)*xt.unsqueeze(1) - torch.exp(theta_est2_1)*(xt.unsqueeze(1)**2)
                    else:
                        log_odds_1 = theta_est0_1 + theta_est1_1*xt.unsqueeze(1) - theta_est2_1*(xt.unsqueeze(1)**2)

            log_odds_p = log_odds[y==1.]
            log_odds_q = log_odds[y==0.]

            if self.cfg.regularize:
                log_odds_p_1 = log_odds_1[y==1.]
                log_odds_q_1 = log_odds_1[y==0.]

            #TODO: calculate log_odds_t for picking a t1 larger than current t and add regularization coefficinet
            term1 = self.log_sigmoid(log_odds_p)
            term2 = self.log_sigmoid(-log_odds_q)
            if self.cfg.regularize:
                term1_1 = self.log_sigmoid(log_odds_p_1)
                term2_1 = self.log_sigmoid(-log_odds_q_1)

            if self.cfg.regularize is False or self.cfg.single_classifier:
                loss = -torch.mean(term1) - torch.mean(term2)
            else:
                kl = -torch.sum(torch.sigmoid(log_odds)*torch.log(torch.sigmoid(log_odds)/(torch.sigmoid(log_odds_1))))
                
                loss = -torch.mean(term1) - torch.mean(term2) + torch.mean(abs(self.cfg.regularize_coef/(t-t1)*(kl)))


            if self.global_step % (self.cfg.total_steps//100) == 0:
                # print('sigmoid log odds', torch.sigmoid(log_odds))
                # print('sigmoid log odds_1', torch.sigmoid(log_odds_1))
                # print('kl', kl)
                # print('reg', abs(self.cfg.regularize_coef/(t-t1)*(kl)))
                # print('loss', loss)
                self.loss_t = np.append(self.loss_t, loss.item())
                print('self.ls', self.loss_t.shape)
            metrics = {'loss': loss}
            
            if self.cfg.use_wandb:
                wandb.log(metrics)
    
            self.optimizer.zero_grad()
            loss.backward()
            # for name, p in self.model.named_parameters():
            #     print(name, 'param', p)
            #     print('grad', p.grad)
            # print('loss', loss.grad)
            self.optimizer.step()

            if self.global_step % (self.cfg.total_steps//100) == 0:
                print('step: ', self.global_step, 'loss: ', metrics['loss'])
                print('t', t)

            self.global_step += 1

            if self.cfg.single_classifier:
                if self.global_step % 100 == 0:
                    self.eval_single()
            else:
                if self.global_step % 10000 == 0:
                    self.eval(self.global_step)
        if self.cfg.single_classifier is False:
            self.eval(self.global_step)


@hydra.main(config_path='.', config_name='scratch')
def main(cfg):
    from scratch import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
         
