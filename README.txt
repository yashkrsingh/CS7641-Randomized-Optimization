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
    - fetal-health.csv: Dataset taken from Kaggle with 10+ attributes of fetus health during pregnancy with their subsequent class labels.
    - fetal-health.names: Feature headers for breast cancer dataset

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

