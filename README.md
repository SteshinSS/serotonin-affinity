# Solution to the test task
This is repo of solution of my test task. The task is [here](https://drive.google.com/file/d/1dNMYiFuu5nFWBw4zGyKAw9sqFra4mVUv/view?usp=sharing).

## Installation
```bash
# Clone repo
git clone https://github.com/SteshinSS/serotonin-affinity.git serotonin_affinity
cd serotonin_affinity

# Install packages
conda create --name steshin_solution python=3.9
conda activate steshin_solution
pip install -r requirements.txt

# Download data
dvc pull
```

Note: it is possible you are reading this instruction after I shut down my S3 bucket. In this case, download raw data from my gdrive and run the pipeline. It should be done in ten minutes:

```bash
python scripts/download_data.py
dvc repro
```

## Result
To see final metrics, run:
```
dvc metrics show
```
Here is my output:
```
| Path                                 | Accuracy | F1-Score | ROC_AUC |
|--------------------------------------|----------|----------|---------|
| results/gb_topological_fp/train.json | 0.95509  | 0.86962  | 0.96166 |
| results/gb_topological_fp/val.json   | 0.89493  | 0.66667  | 0.82903 |
| results/gb_morgan_fp/train.json      | 0.93585  | 0.82291  | 0.9486  |
| results/gb_morgan_fp/val.json        | 0.88743  | 0.64286  | 0.81369 |
| results/gb_maccs_fp/train.json       | 0.90239  | 0.74712  | 0.91591 |
| results/gb_maccs_fp/val.json         | 0.84428  | 0.59113  | 0.82143 |
| results/cnn/train.json               | 0.58148  | 0.37874  | 0.68182 |
| results/cnn/val.json                 | 0.57223  | 0.34483  | 0.66279 |
| results/dummy/train.json             | 0.84586  | 0.0      | 0.5     |
| results/dummy/val.json               | 0.85741  | 0.0      | 0.5     |
```

## Details
### Highlights
- Homology-based splitting
- Weights for inbalanced dataset
- Gradient Boosting on fingerprints as a baseline
- 3D-CNN on voxel grid as a complicated model
- AWS S3 for storage
- DVC for data management
- Docker for reproducibility (TODO)
- GitAction for continuous integration (TODO)
- The Workflow allows working with scripts as well as notebooks.

### How to use it
`jupyter-notebook notebooks/train/train_gb.ipynb` if you prefer notebooks.

`dvc repro` and see `params.yaml` if you prefer command line.

### What is here

#### Pipeline
I've prepared a whole pipeline only to show my engineering skills. By no means is the pipeline appropriate because it's too complicated for the toy task. In real cases, however, choosing tools would depend on our goals, resources, and a team's preference. We can discuss some details of it if you are interested. 

Run `dvc dag` to see pipeline's scheme. Here is part of it for evaluating CNN on the validation dataset.

```
          +--------------+
          | data/raw.dvc |      
          +--------------+      
                  *
                  *
                  *
             +--------+
             | filter |
             +--------+
                  *
                  *
                  *
              +-------+
              | split |
              +-------+
                  *
                  *
                  *
       +---------------------+
       | generate_conformers |
       +---------------------+
                  *
                  *
                  *
         +-----------------+
         | generate_voxels |
         +-----------------+
           ***         ***
          *               *
        **                 ***
+-----------+                 *
| train_cnn |              ***
+-----------+             *
           ***         ***
              *       *
               **   **
        +------------------+
        | evaluate_cnn@val |
        +------------------+
```

#### Splitting
There are many ways of splitting our data. A simple way is a random split. The problem is that we can have very similar molecules in both train and test datasets. That will affect our decision in a way that our model became overfitted to the current dataset. That is why we prefer a homology-based split generally. We will cluster similar molecules and select whole clusters for different splits. See `notebooks/EDA/01_filtered.ipynb` for details.

#### Model Selection
I chose sklearn's implementation of Gradient Boosting trees because it's a powerful and fast algorithm. I almost did no hyperparameter optimization, but I used weighted loss to compensate for the unbalanced dataset.

#### Improved Model
I'm applying for the DL researcher position, so for an advanced model, I chose to train 3D-CNN. I generated conformers by rdkit, prepared a voxel grid by MoleculeKit, and train 3D-CNN by Pytorch-Lightning. It allows us to inject information about the molecular force field and geometry. You can find this approach, for example, [here](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00740). 

The resulting network is only marginally better than the dummy baseline. I don't have time to tune it, but I would augment the voxels dataset by random rotation of conformers.

#### Results / Conclusion
Note: I didn't do any hyperparameter optimization, because it's easy but time-consuming. I personally love it and find it way funnier than creating infrastructure. Tell me, if you want me to do HPO, anyway. In this case, I would use ad-hoc scripts, but I also know how to with RayTune and KerasTuner.

Back to the task, now we can see, that metrics for the Gradient Boosting models are pretty the same. We can make a weak conclusion that topological fingerprints are the best choice, but that is probably is due to the lack of proper tuning.

#### What to do next
- Try GNN. (Or both CNN and GNN multimodal network! See [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01306)
- We probably want to score potential ligands instead of making 0-1 predictions. That's why we want to know how confident our model is. There are many ways to include uncertainty and Monte-Carlo Dropout is a computationally cheap example.
- We only have 6k objects in a dataset. Let's try some transfer learning. For example, we can use pre-trained embeddings, like Smiles2Vec or some self-supervised neural embeddings.
- There are physics-based affinity prediction models. Can we use their predictions to augment our dataset? (Something like Physical-Based DL)
