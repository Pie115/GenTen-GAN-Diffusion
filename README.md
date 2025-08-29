# Efficiently Generating Multidimensional Calorimeter Data with Tensor Decomposition Parameterization

Code used in our experiments.

Abstract: Producing large complex simulation datasets can often be a time and resource consuming task. Especially when these experiments are very expensive, it is becoming more reasonable to generate synthetic data for downstream tasks. Recently, these methods may include using generative machine learning models such as Generative Adversarial Networks or diffusion models. As these generative models improve efficiency in producing useful data, we introduce an internal tensor decomposition to these generative models to even further reduce costs. More specifically, for multidimensional data, or tensors, we generate the smaller tensor factors instead of the full tensor, in order to significantly reduce the model's output and overall parameters. This reduces the costs of generating complex simulation data, and our experiments show the generated data remains useful. As a result, tensor decomposition has the potential to improve efficiency in generative models, especially when generating multidimensional data, or tensors.

Research was supported in part by the National Science
Foundation under CAREER grant no. IIS 2046086 and
grant no. IIS 1901379, by the Agriculture and Food Research Initiative Competitive Grant no. 2020-69012-31914
from the USDA National Institute of Food and Agriculture,
and by the Army Research Office and was accomplished
under Grant Number W911NF-24-1-0397. The views and
conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Office or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for
Government purposes notwithstanding any copyright notation herein
