## Step 1
create the environment using 

```bash
conda env create -f env.yml
```

## Step 2
download the package for text tokenization

```bash
python -m spacy download en_core_web_lg
```

## Step 3
check demo.ipynb to see how to tokenize the text and detect the usage of LLM