
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from .vqvae import VQEmbedding


def create_nn(input_size, output_size, hidden_size, num_layers, activation_fn=nn.ReLU, input_normalizer=None,
              final_activation_fn=None, hidden_init_fn=None, b_init_value=None, last_fc_init_w=None):
    # Optionally add a normalizer as the first layer
    if input_normalizer is None:
        input_normalizer = nn.Sequential()
    layers = [input_normalizer]

    # Create and initialize all layers except the last one
    for layer_idx in range(num_layers - 1):
        fc = nn.Linear(input_size if layer_idx == 0 else hidden_size, hidden_size)
        if hidden_init_fn is not None:
            hidden_init_fn(fc.weight)
        if b_init_value is not None:
            fc.bias.data.fill_(b_init_value)
        layers += [fc, activation_fn()]

    # Create and initialize  the last layer
    last_fc = nn.Linear(hidden_size, output_size)
    if last_fc_init_w is not None:
        last_fc.weight.data.uniform_(-last_fc_init_w, last_fc_init_w)
        last_fc.bias.data.uniform_(-last_fc_init_w, last_fc_init_w)
    layers += [last_fc]

    # Optionally add a final activation function
    if final_activation_fn is not None:
        layers += [final_activation_fn()]
    return nn.Sequential(*layers)


class DensityModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def novelty(self, *args, **kwargs):
        return torch.zeros(10)


class simple_path(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class BaseVAEDensity(nn.Module, DensityModule):
    def __init__(self, num_skills, state_size, hidden_size, code_size,
                 num_layers=4, normalize_inputs=False, skill_preprocessing_fn=simple_path,
                 input_key='next_state', input_size=None):
        super().__init__()

        self.num_skills = int(num_skills)
        self.state_size = int(state_size) if input_size is None else int(input_size)
        self.code_size = int(code_size)
        self.normalize_inputs = bool(normalize_inputs)
        self.skill_preprocessing_fn = skill_preprocessing_fn
        self.input_key = str(input_key)

        self._make_normalizer_module()

        assert num_layers >= 2
        self.num_layers = int(num_layers)

        self.encoder = create_nn(input_size=self.input_size, output_size=self.encoder_output_size,
                                 hidden_size=hidden_size, num_layers=self.num_layers,
                                 input_normalizer=self.normalizer if self.normalizes_inputs else nn.Sequential())

        self.decoder = create_nn(input_size=self.code_size, output_size=self.input_size,
                                 hidden_size=hidden_size, num_layers=self.num_layers)

        self.mse_loss = nn.MSELoss(reduction='none')

    @property
    def input_size(self):
        return self.state_size + self.num_skills

    @property
    def encoder_output_size(self):
        return NotImplementedError

    @property
    def normalizes_inputs(self):
        return self.normalizer is not None

    def _make_normalizer_module(self):
        raise NotImplementedError

    def compute_logprob(self, batch, **kwargs):
        raise NotImplementedError

    def novelty(self, batch, **kwargs):
        with torch.no_grad():
            return -self.compute_logprob(batch, **kwargs).detach()

    def update_normalizer(self, **kwargs):
        if self.normalizes_inputs:
            self.normalizer.update(**kwargs)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint)

    def forward(self, batch):
        raise NotImplementedError


class VQVAEDensity(BaseVAEDensity):
    def __init__(self, num_skills, state_size, hidden_size, codebook_size, code_size, beta=0.25, **kwargs):
        super().__init__(num_skills=num_skills, state_size=state_size, hidden_size=hidden_size, code_size=code_size,
                         **kwargs)
        self.codebook_size = int(codebook_size)
        self.beta = float(beta)

        self.apply(self.weights_init)

        self.vq = VQEmbedding(self.codebook_size, self.code_size, self.beta)

    @property
    def encoder_output_size(self):
        return self.code_size

    def _make_normalizer_module(self):
        self.normalizer = Normalizer(self.input_size) if self.normalize_inputs else None

    @classmethod
    def weights_init(cls, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            try:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            except AttributeError:
                print("Skipping initialization of ", classname)

    def compute_logprob(self, batch, with_codes=False):
        s, z = batch[self.input_key], self.skill_preprocessing_fn(batch['skill'])
        x = torch.cat([s, z], dim=1)
        z_e_x = self.encoder(x)
        z_q_x, selected_codes = self.vq.straight_through(z_e_x)
        x_ = self.decoder(z_q_x)
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)
        if with_codes:
            return logprob, z_e_x, selected_codes
        else:
            return logprob

    def get_centroids(self, z_idx):
        z_q_x = torch.index_select(self.vq.embedding.weight.detach(), dim=0, index=z_idx)
        centroids = self.decoder(z_q_x)
        if self.normalizes_inputs:
            centroids = self.normalizer.denormalize(centroids)
        return centroids

    def novelty(self, batch, **kwargs):
        with torch.no_grad():
            return -self.compute_logprob(batch, with_codes=False).detach()

    def forward(self, batch):
        logprob, z_e_x, selected_codes = self.compute_logprob(batch, with_codes=True)
        loss = self.vq(z_e_x, selected_codes) - logprob
        return loss.mean()


class VQVAEDiscriminator(VQVAEDensity):
    def __init__(self, state_size, hidden_size, codebook_size, code_size, beta=0.25, **kwargs):
        super().__init__(num_skills=0, state_size=state_size, hidden_size=hidden_size, codebook_size=codebook_size,
                         code_size=code_size, beta=beta, **kwargs)
        self.softmax = nn.Softmax(dim=1)

    def _make_normalizer_module(self):
        # self.normalizer = DatasetNormalizer(self.input_size) if self.normalize_inputs else None
        self.normalizer = None

    def compute_logprob(self, batch, with_codes=False):
        x = batch[self.input_key]
        z_e_x = self.encoder(x)
        z_q_x, selected_codes = self.vq.straight_through(z_e_x)
        x_ = self.decoder(z_q_x)
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)
        if with_codes:
            return logprob, z_e_x, selected_codes
        else:
            return logprob

    def compute_logprob_under_latent(self, batch, z=None):
        x = batch[self.input_key]
        if z is None:
            z = batch['skill']
        z_q_x = self.vq.embedding(z).detach()
        x_ = self.decoder(z_q_x).detach()
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)
        return logprob

    def log_approx_posterior(self, batch):
        x, z = batch[self.input_key], batch['skill']
        z_e_x = self.encoder(x)
        codebook_distances = self.vq.compute_distances(z_e_x)
        p = self.softmax(codebook_distances)
        p_z = p[torch.arange(0, p.shape[0]), z]
        return torch.log(p_z)

    def surprisal(self, batch):
        with torch.no_grad():
            return self.compute_logprob_under_latent(batch).detach()
        
    def compute_intr_reward(self, skill, next_obs):
        z_q_x = self.vq.embedding(skill).detach()
        x_ = self.decoder(z_q_x).detach()
        log_q_s_z = -1. * self.mse_loss(next_obs, x_).sum(dim=1)
        return torch.clamp(log_q_s_z, max=10.0)
        '''        
        sum_q_s_z_i = torch.zeros_like(log_q_s_z)
        for z_i in range(self.codebook_size):
            z_q_x = self.vq.embedding(torch.tensor(z_i).to(skill.device)).detach()
            x_ = self.decoder(z_q_x).detach()
            logprob = -1. * self.mse_loss(next_obs, x_).sum(dim=1)
            sum_q_s_z_i += torch.exp(logprob)
        r = log_q_s_z + torch.log(torch.tensor(self.codebook_size)) - torch.log(sum_q_s_z_i)
        
        return torch.clip(r, max=10.0)
        '''
