
import boto3
import json

data_source_id = ""
kb_client_runtime = boto3.client('bedrock-agent-runtime')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')

model_id = "anthropic.claude-v2"
q_model_id = "anthropic.claude-instant-v1"
region = "us-east-1"

original_query = ''

queries = generate_queries(original_query)
queries.insert(0, original_query)

all_results = {}
for q in queries:
    all_results[q] = kb_search(q)

reranked_results = list(reciprocal_rank_fusion(all_results).keys())[:5]

information = ''
for i, r in enumerate(reranked_results):
    information += f'{i+1}. {r}\n'

prompt = """

<Information>
{information}
</Information>

<UserQuery>
{query}
</UserQuery>
"""
prompt = prompt.format(information=information, query=original_query)

print(invoke_claude(prompt, model_id))

def get_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=500)

    docs = text_splitter.split_documents(documents)
    return docs


def invoke_claude(text, model_id, max_tokens_to_sample=1000):
    body = json.dumps({
        "prompt": f"\n\nHuman:{text}\n\nAssistant: ",
        "max_tokens_to_sample": max_tokens_to_sample,
        "temperature": 0.1,
        "top_p": 0.9,
    })
    accept = 'application/json'
    content_type = 'application/json'

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    response_body = json.loads(response.get('body').read())
    return response_body.get('completion')[1:]


def generate_queries(original_query, n=4):
    f = ''
    for i in range(n):
        f += f'{n}: \n'

    prompt = f'''{n}

<Query>
{original_query}
</Query>

<Format>
{f}
</Format>
'''
    result = invoke_claude(prompt, q_model_id)
    result = result.split('<Format>')[1].split('</Format>')[0]
    # print(result)
    generated_queries = []
    for q in result.split('\n'):
        if q == '':
            continue
        generated_queries.append(q.split(' ')[1])
    return generated_queries[1:-1]

def kb_search(query, n=5):
    res = kb_client_runtime.retrieve(
        retrievalQuery= {
            'text': query
        },
        knowledgeBaseId=knowledge_base_id,
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': n
            }
        }
    )
    # {doc: score}の形に整形します
    return_dict = {}
    for r in res['retrievalResults']:
        return_dict[r['content']['text']] = r['score']

    return return_dict

#### RRF  Calculation Logic

def reciprocal_rank_fusion(all_results, k=50):
    fused_scores = {}
    for query, doc_scores in all_results.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

