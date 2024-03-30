# RAG IMPLEMENTATION USING AWS BEDROCK API & RAG FUSION TECHNIQUE  IMPLEMENTATION

## How to run the Project?

```bash
conda create -n bedrock python=3.11.7 -y
```

# NOTE:  

1) Initialize the API Keys in a  .env file inside  the source  code  folder  if you are using some other LLM Models or OpenAI API instead of AWS Bedrock API for Integration purpose.
2) Kindly create an IAM User and  attach the policies for full access to  AWS Bedrock service to this user.
3) Kindly  request for the specific model access in the aws Bedrock API landing page  before getting  started with this project. 

```bash
conda activate bedrock 
```

```bash
pip install -r requirements.txt
```

### Install aws cli from the following link:
```bash
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
```

### Add credentials by running the following command
```bash
aws configure
```

### To run streamlit app

```bash
streamlit run bedrock_test.py
```

```bash
streamlit run rag_demo.py
```
