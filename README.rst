Synthsonic
==========

Synthsonic: fast, probabilistic data modelling and synthesis

Installation
------------

Users can install with :code:`pip install .`

Developers should install with :code:`pip install -e '.[dev,test]'`

Requirements
************

Python 3.6 -- 3.8 are supported

Project Organization
--------------------

* The `Synthsonic` model is found at: `synthsonic/models/kde_copula_nn_pdf.py`
* As an example how to run it with the SDGym package, have a look at the notebook: `notebooks/leaderboard/adult_all_variables.ipynb`
* Figures of the neurips paper can be reproduced by running the notebooks in: `notebooks/plots/`
* Experiments (including leaderboard, ablation studies) can be rerun with the instructions found in: `notebooks/experiments/`


Quick run
---------

As a quick example, you can do:

.. code-block:: python

  from synthsonic.models.kde_copula_nn_pdf import KDECopulaNNPdf
  import pandas as pd

  # open fake car insurance data
  df = pd.read_csv('notebooks/plots/fake_insurance_data.csv.gz')
  df.head()

  # model the data
  model= KDECopulaNNPdf()
  model = model.fit(df.values)
