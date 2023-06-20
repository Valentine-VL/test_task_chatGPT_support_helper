import ast

import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding
from scipy import spatial

openai.api_key = ""
openai.organization = ""

# --------------------------
# embedding model parameters which were used to create *embeddings.csv

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
embedding_encoding = "cl100k_base"
max_tokens = 8000


def make_embedding():
    df = pd.read_json("FAQ.json")
    df = df[["Question_original", "Answer_plain_text", "Notes"]]
    df = df.dropna()
    df["combined"] = (
            "Question: " + df.Question_original.str.strip() +
            "; Answer: " + df.Answer_plain_text.str.strip() +
            "; Notes: " + df.Notes.str.strip()
    )

    top_n = 1000
    df = df.sort_values("Question_original").tail(top_n * 2)
    df.drop("Question_original", axis=1, inplace=True)

    encoding = tiktoken.get_encoding(embedding_encoding)

    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens].tail(top_n)

    df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    df.to_csv("task_FAQ_embeddings.csv")


# ----------------------------------


embeddings_path = "task_FAQ_embeddings.csv"
df = pd.read_csv(embeddings_path)

df['embedding'] = df['embedding'].apply(ast.literal_eval)


def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 30
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["Answer_plain_text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below FAQ info to answer the subsequent question. If the answer cannot be found in the info, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nFAQ info:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def gpt_answer(
        query: str,
        df: pd.DataFrame = df,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message
