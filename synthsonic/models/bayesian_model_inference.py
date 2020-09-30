from pgmpy.inference import Inference
from pgmpy.models import BayesianModel
import networkx as nx


class BayesianModelInference(Inference):
    """Class to calculate probability (pmf) values specific to Bayesian Models
    """

    def __init__(self, model):
        """Class to calculate probability (pmf) values specific to Bayesian Models

        :param model: instance of BayesianModel model on which inference queries will be computed
        """
        if not isinstance(model, BayesianModel):
            raise TypeError("Model expected type: BayesianModel, got type: ", type(model))
        super(BayesianModelInference, self).__init__(model)

        self.topological_order = list(nx.topological_sort(model))

    def _log_probability_node(self, X, columns, node):
        """Evaluate the log probability of each datapoint for a specific node.

        Internal function used by log_probability().

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :param list columns: list of column names used by the Bayesian network.
        :param node: node from the Bayesian network.
        :return: ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        def vec_translate(a, my_dict):
            return np.vectorize(my_dict.__getitem__)(a)

        cpd = self.model.get_cpds(node)

        # direct variable: x[n], where n is the node
        current = cpd.variables[0]
        current_idx = columns.index(current)
        current_val = X[:, current_idx]
        current_no = vec_translate(current_val, cpd.name_to_no[current])

        # conditional dependencies E of the direct variable
        evidence = cpd.variables[:0:-1]
        evidence_idx = [columns.index(ev) for ev in evidence]
        evidence_val = X[:, evidence_idx]
        evidence_no = np.empty_like(evidence_val)
        for i, ev in enumerate(evidence):
            evidence_no[:, i] = vec_translate(evidence_val[:, i], cpd.name_to_no[ev])

        if evidence:
            # there are conditional dependencies E
            # Here we retrieve array: p(x[n]|E). We do this for each x in X.
            # We pick the specific node value below of the arrays below.
            cached_values = self.pre_compute_reduce(variable=node)
            weights = np.array([cached_values[tuple(en)] for en in evidence_no])
        else:
            # there are NO conditional dependencies E
            # retrieve array: p(x[n]).  We do this for each x in X.
            # We pick the specific node value below of the array below.
            weights = np.array([cpd.values] * len(X))

        # pick the specific node value x[n] from the array p(x[n]|E) or p(x[n]), for each x in X.
        probability_node = np.take_along_axis(weights, current_no, axis=None)

        return np.log(probability_node)

    def log_probability(self, X, columns):
        """Evaluate the logarithmic probability of each point in a data set.

        :param X: pandas dataframe OR array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :param list columns: list of column names used by the Bayesian network model.
        :return: ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        if isinstance(X, pd.DataFrame):
            # use numpy array from now on.
            columns = df.columns.to_list()
            X = X.values

        if columns is None:
            columns = list(range(X.shape[0]))
        else:
            columns = list(columns)

        logp = np.array([self._log_probability_node(X, columns, node) for node in self.topological_order])
        return np.sum(logp, axis=0)

    def score(self, X, columns):
        """Compute the total log probability density under the model.

        :param X: pandas dataframe OR array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :param list columns: list of column names used by the Bayesian model.
        :return: float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return np.sum(self.log_probability(X, columns))
