from tqdm import tqdm

from pgmpy.factors import factor_product
from pgmpy.inference import Inference
from pgmpy.models import BayesianModel, MarkovChain, MarkovModel
from pgmpy.utils.mathext import sample_discrete
from pgmpy.sampling import _return_samples
from pgmpy.global_vars import SHOW_PROGRESS


class ProbabilityInference(BayesianModelSampling):
    """
    Class to calculate probability (pmf) values specific to Bayesian Models
    Parameters
    ----------
    model: instance of BayesianModel
        model on which inference queries will be computed
    """

    def __init__(self, model):
        super(ProbabilityInference, self).__init__(model)

    def _log_probability_x(self, x, columns):
        """
        """
        log_probability = np.array([self._log_probability_x_node(x, columns, node) for node in self.topological_order])
        return np.sum(log_probability)

    def _log_probability_x_node(self, x, columns, node):
        """
        """
        cpd = self.model.get_cpds(node)

        # current node in graph
        current = cpd.variables[0]
        current_idx = columns.index(current)
        current_val = x[current_idx]
        current_no = cpd.name_to_no[current][current_val]

        # dependent nodes (i.e. conditional probability)
        evidence = cpd.variables[:0:-1]
        evidence_idx = [columns.index(ev) for ev in evidence]
        evidence_val = x[evidence_idx]
        evidence_no = [cpd.name_to_no[e][v] for e, v in zip(evidence, evidence_val)]
        evidence_no = tuple(evidence_no)

        if evidence:
            cached_values = self.pre_compute_reduce(variable=node)
            weights = cached_values[evidence_no]
        else:
            weights = cpd.values

        log_probability = np.log(weights[current_no])
        return log_probability

    def log_probability(self, X, columns=None, show_progress=True):
        """
        """
        if columns is None:
            columns = list(range(X.shape[0]))
        else:
            columns = list(columns)

        if show_progress and SHOW_PROGRESS:
            pbar = tqdm(X)
        else:
            pbar = X

        logp = np.array([self._log_probability_x(x, columns) for x in pbar])
        return logp if len(X) > 1 else logp[0]
