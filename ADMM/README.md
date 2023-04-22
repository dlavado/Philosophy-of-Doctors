
# SCENE-Net: Interpretable Semantic Segmentation of 3D Point Clouds via Signature Shape  Identification Based on Group Equivariant Non-Expansive Operators

This project allows for the reproducility of the results achieved with SCENE-Net.



##
## Installation

Install the requiremnts for this project

```bash
    pip install -r requirements.txt
```
    
## Deployment

SCENE-Net can be run with the following command:

```bash
  python scenenet_pipeline.py [--vis] [--val] [--no_train] [--tuning] [--load_model LOAD_MODEL] [--model_tag MODEL_TAG] [--model_date MODEL_DATE]
                              [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [-a ALPHA] [-e EPSILON]
                              [-k K_SIZE [K_SIZE ...]] [--cy CY] [--cone CONE] [--neg NEG]
```

- Visualization Mode `--vis` allows us to visualize the performance of SCENE-Net in the testing set. This mode should be used if you have a trained SCENE-Net model
  This mode requires flags: `--load_model` and `--model_date`;  

- Validation Mode `--val` performs validation during SCENE-Net's training.

- No Train Mode `no_train` allows the user to skip training of a SCENE-Net model.
  - Usually paired with `--load_model` and `--model_date` flags.

- Fine-Tuning Mode `--tuning` allows the user to continue training a pre-existing model.
  - This mode requires flags: `--load_model` and `--model_date`;  This can also be used to test the current model if paired with flag`--no_train`

- Model Directory `--model_date` this is the grandparent directory where the `gnet.pt` file resides, which contains the desired SCENE-Net model.
  - This directory MUST be a date in YEAR-MONTH-Day format (all ints)

- Select Model `--load_model`: this selects the parent directory inside the `--model_date` directory.
  - This can be an `int`: with the index of the parent directory containing the model (imagine that your `--model_date` is a python list)
  - Or it can be a `string`: with the relative path to said intented folder

- Model Tag `--model_tag` let us choose what model to use. During training, we save the best model according to each metric, the latest model and the best in terms of loss.
  - --model tag {JaccardIndex, Precision, F1Score, FBetaScore, latest, loss}

- kernel size `-k | --k_size` takes three ints as input as the kernel size (z, x, y)

- All of the remaining flags set up the respective parameters in the pipeline 


### Running  Example 
```bash
  python scenenet_pipeline.py --load_model 2 --model_date 2022-08-04 --model_tag FBetaScore 
                         --vis --k_size 9 5 5
```