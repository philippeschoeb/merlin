:github_url: https://github.com/merlinquantum/merlin

============================================================================
Computational Advantage in Hybrid Quantum Neural  Networks: Myth or Reality?
============================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Computational Advantage in Hybrid Quantum Neural  Networks: Myth or Reality?

   **Authors**: Muhammad Kashif, Alberto Marchisio, Muhammad Shafique

   **Published**: arXiv Preprint (2024)

   **DOI**: `https://doi.org/10.48550/arXiv.2412.04991 <https://doi.org/10.48550/arXiv.2412.04991>`_

   **Reproduction Status**: 🚧 In Progress

   **Reproducer**: Cassandre Notton (cassandre.notton@quandela.com)

Abstract
========

In this work, the authors propose a benchmark methodology to compare Hybrid Quantum Neural Networks (HQNN) and Multi Layer Perceptron.
First, they define a spiral dataset with a fixed number of classes and an increasing number of features.
They compare the number of FLOPs and parameters to reach at least 90% of accuracy on a dataset on which they can increase the complexity.

Significance
============

This paper is highly relevant because of the benchmark methodology proposed and its conclusion:
*In summary, HQNNs offer a scalable, resource-efficient alternative to classical models, positioning them as a promising solution for complex tasks in machine learning.*

MerLin Implementation
=====================

First, for each number of modes ``m`` and for specific number of photons from 1 to ``m//2``, we generate a classifier made of 2 generic interferometers and an encoding layer in between.
Then, we sort all these models by the number of trainable parameters.
We do the same for the classical MLP where we generate models of different thicknesses and depths and sort them by their parameter count.

Key Contributions Reproduced
============================

**Comparison of the number of trainable parameters for features from 5 to 50**
  * We observe that the HQNN requires fewer parameters than the MLP as the number of features increases.
  * The HQNN demonstrates superior efficiency in parameter usage, leading to reduced computational complexity.
  * Specific achievement: The HQNN maintains high accuracy even with a lower number of parameters, showcasing its effectiveness.
  * Specific achievement: The HQNN's performance scales better with an increasing number of features, making it more suitable for high-dimensional data.



Implementation Details
======================

Here, we use the encoding strategy from `Gan et al. <https://arxiv.org/abs/2107.05224>`_ and build our quantum circuit accordingly (see function ``create_quantum_circuit`` in the hqnn_spiral notebook: :doc:`../../notebooks/hqnn_spiral`).

.. code-block:: python

   import merlin as ml

   boson_layer = ML.QuantumLayer(
                    input_size=NB_FEATURES,
                    output_size=NB_CLASSES,
                    circuit=circuit,
                    trainable_parameters=["ps", "bs"],
                    input_parameters=["px"],
                    input_state=input_state,
                    no_bunching = True,
                    output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
                )


Experimental Results
====================

Below, we present the minimum number of parameters required to reach at least 90% of accuracy (averaged on 5 experiments), for the spiral dataset with the number of features increasing from 5 to 50:

+------------+-----------+-----------+
| Features   | MLP       | HQNN      |
+============+===========+===========+
| 5          | 75        | 62        |
+------------+-----------+-----------+
| 10         | 59        | 67        |
+------------+-----------+-----------+
| 20         | 99        | 77        |
+------------+-----------+-----------+
| 30         | 275       | 87        |
+------------+-----------+-----------+
| 40         | 355       | 178       |
+------------+-----------+-----------+
| 50         | 435       | 203       |
+------------+-----------+-----------+


Interactive Exploration
=======================

**Jupyter Notebook**: doc:`../../notebooks/hqnn_spiral`

Extensions and Future Work
==========================

This is still a work in progress. **Future experiments** will include
  * FLOP analysis
  * deeper analysis of the hyperparameters for both HQNN and MLP



Code Access and Documentation
=============================

**GitHub Repository**: `merlin/reproductions/[folder_name] <https://github.com/merlinquantum/merlin/tree/main/reproductions/hqnn_spiral>`_


Citation
========

.. code-block:: bibtex

   @article{kashif2024computational,
     title={Computational Advantage in Hybrid Quantum Neural Networks: Myth or Reality?},
     author={Kashif, Muhammad and Marchisio, Alberto and Shafique, Muhammad},
     journal={arXiv preprint arXiv:2412.04991},
     year={2024}
   }


.. note:: End of document.

