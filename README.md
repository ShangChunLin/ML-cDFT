# ML-cDFT
Machine learning classical density functional theory (ML-cDFT)

One need Jupyter notebook with python3 for training, and CUDA (also Nvidia GPU, of course) if want to generate own data.  

A machine learning method to approximate the classical free energy density functional. The detail could be found in SciPost, https://scipost.org/SciPostPhys.6.2.025

Usaully ML is a black box to theoretical physicists, here we try to discover the insight and connect ML and cDFT. 
This is a project to answer, "could ML find a free energy functional?". The short answer is, sort of yes, as least it could be approximated by ML. 
A good free energy density require hard works and incredible physical insight, such as fundmental measure theory for hard sphere colloid. However, even we have good approximation for that (hard sphere system), by adding mean field for attractive potential, it usually got spoiled. Thus, we propose a ML method to improve such situation. It turns out that the idea could atually improve the attractive part that better than mean field treatment with simple convolution and multiplication network.    
In the paper we apply the idea to the 1D LJ fluid to prove the concept. The training data are grand canonical MC simulation and speeded up by GPU (CUDA), and training by python (in Jupyter notebook). The training should be able to imposed by Tensorflow, but somehow not working . It will be very appreciated if someone could rewrite the training part by Tensorflow.

description of folders:

HR_test - Compare GMC result with exact 1D hard rod functional

LJ_data - density profile with vary tempature and chemical potential 

LJ_data_fix_mu_T - density profile with fixed tempature and chemical potential

LJ_data_fuzzy - density profile with vary tempature and chemical potential, higher resolution but less accuracy

LJ_python_prototype - GMC simulation in jupyternotebook. To compare with GPU result. 

ML_data - save weighting parameters.

LJ_function_exam_XX.ipynb - Exam the functional generated by ML training  

LJ_ML_training_XX.ipynb - ML training part

Usage:

The comment mainly in LJ_function_exam_sq.ipynb and LJ_ML_training_sq.ipynb. The cubic version are similar. 

I have some pre-trianed weights for both LJ_function_exam_XX.ipynb in the folder "ML_data", just open LJ_function_exam_xx.ipynb and press run.

If one want to train self, open LJ_ML_training_XX.ipynb, choose parameter, training, and wait.

For training data, edit LJ_data.cu in the folder "LJ_data", then compile and run it on your favorite way. GMC simulation and CUDA is not the major points here and I am not an expert in MC simulation. 

For any question/suggestion/bug/help/whatever, please mail me to 
shang-chun.lin@uni-tuebingen.de
