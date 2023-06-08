Implementation of the model in the *Data-driven score-based models for generating
stable structures with adaptive crystal cells* paper.

The cuda version used is V10.1.243

### Data

Extract the `data.zip` archive in the root folder to gain access to the data.

### Training

Use the following command to train the model:

    python3 main.py --config config.yml --doc choose\_a\_name
    
To continue training:

    python3 main.py --config config.yml --doc name --resume_training
    
### Sampling

To sample using a trained model model, prepare a file following the example at sampling\_orders/example.yml and execute:

    python3 main.py --config config.yml --doc name --sample --sampling\_order\_path name\_of\_the\_file

Or use a pretrained model:

    python3 main.py --config config.yml --doc pretrained --sample --sampling\_order\_path name\_of\_the\_file
    
The samples are found in `exp/cif_samples`. You can run the script `postprocessing/uniques.py` to remove the redundant structures, if necessary, for example:

    python3 postprocessing/uniques.py --dir exp/cif_samples/pretrained/name

