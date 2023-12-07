# sara-ie

This GitHub repo contains the code for the experiments of the paper

**Connecting Symbolic Statutory Reasoning with Legal Information Extraction** \[[paper](https://aclanthology.org/2023.nllp-1.12.pdf)\]
Nils Holzenberger and Benjamin Van Durme
*Natural Legal Language Processing Workshop*, 2023

First, install the required packages listed in `requirements.txt`.

Then, download the data, unzip it and place it in the right directory, by running `bash get_sara_v3.sh`. You can also find the data on its [landing page](https://nlp.jhu.edu/law/sara_v3/).

You can now try the model used in the paper. The script `eval-sara-ie.sh` will train and evaluate a model, given a config file, so you can use this as a starting point. This script typically runs with a GPU.

Other scripts contain comments on how they are to be used.
