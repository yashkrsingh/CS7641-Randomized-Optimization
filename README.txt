------------------------------------------------------------------------
CS7641 Randomized Optimization
------------------------------------------------------------------------
------------------------------------------------------------------------
Project Link - https://github.com/yashkrsingh/CS7641-Randomized-Optimization
------------------------------------------------------------------------

------------------
Project Structure
------------------

1. data
    - winequality-white.csv: Dataset taken from UCI ML Repository, containing 11 attributes of different white wines and their quality label as judges by experts on the scale of 1-10
    - winequality.names: Feature headers for winequality dataset

2. scripts
    - processing.py: Contains functions for data preprocessing and result consolidation, including validation and learning curve creation.
    - learner.py: Contains code for fitting a model and performing grid search over a given set of hyperparameters.
    - main.py: Runner or driver for the code containing all the experiments for the project.

3. Requirement.txt
    - contains the required packages to create running environment within the IDE

-----------
How to Run
-----------

Clone the repository using the command `git clone https://github.com/yashkrsingh/CS7641-Randomized-Optimization.git`

Once the files are available in the working directory, running the command `python main.py` from within scripts directory would run all the experiments and generate the figures in the scripts directory.

Separately, `python hyperparametertuning.py` can be run to gather best parameters via a manual gridsearch. The same parameters are already used for all experiments in main.py
