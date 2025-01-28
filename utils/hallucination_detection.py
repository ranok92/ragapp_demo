from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import spacy
import torch 
import numpy as np
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from sklearn.metrics.pairwise import cosine_similarity
import os

with open('../assets/openai_api_key.txt', 'r') as f:
    key = f.read()
os.environ["OPENAI_API_KEY"]=key
nlp = spacy.load("en_core_web_sm")



class CosineDetector:
    def __init__(self, embdedding_model):
        self.embedding_model = embdedding_model

    def check_hallucination(self, query, response, context):

        context_page_content = [doc.page_content for doc in context]
        context_page_content.append("I don't know the answer.")

        resp_sentences = [sent.text.strip() for sent in nlp(response).sents] # spacy sentence tokenization
        sent_embeddings = self.embedding_model.embed_documents(resp_sentences)
        context_embeddings = self.embedding_model.embed_documents(context_page_content)
        query_embeddings = self.embedding_model.embed_documents([query])
        sentence_cosine_scores = cosine_similarity(sent_embeddings, context_embeddings)
        sentence_cosine_scores_original_query = cosine_similarity(sent_embeddings, query_embeddings)
        import ipdb; ipdb.set_trace()
        sentence_cosine_scores = np.concatenate(sentence_cosine_scores, 
                                                sentence_cosine_scores_original_query, axis=1)
        return resp_sentences, np.abs(np.max(sentence_cosine_scores, axis=1))


class SelfcheckNLIDetector:
    def __init__(self, sample_size, response_gen_chain):
        self.sample_size = sample_size
        self.response_sampler_chain = response_gen_chain 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selfcheck_nli = SelfCheckNLI(device=device) # set device to 'cuda' if GPU is available

    def check_hallucination(self, query, response, context):
        '''
        Given the query and context used to generate the original respose
        and the response, check for hallucination on a sentence level
        '''

        sample_responses = []
        #generate sample answers
        for i in range(self.sample_size):
            samp_resp = self.response_sampler_chain.invoke({'input':query, 
                                                    'context':context})
            sample_responses.append(samp_resp)

        #break into sentence
        resp_sentences = [sent.text.strip() for sent in nlp(response).sents] # spacy sentence tokenization

        sent_scores_nli = self.selfcheck_nli.predict(
            sentences = resp_sentences,                          # list of sentences
            sampled_passages = sample_responses, # list of sampled passages
        )
        return resp_sentences, 1-np.array(sent_scores_nli) # score 0: hallucination, 1: no hallucination


class DeepEvalDetector:
    def __init__(self, metric, model: str, threshold: float = 0.5, include_reason: bool= False, async_mode: bool = True):
        if 'faithfulness' in metric:
            self.metric = FaithfulnessMetric(
                                    threshold=threshold,
                                    model=model,
                                    include_reason=include_reason,
                                    async_mode=async_mode
                                )
        else:
            raise NotImplementedError 
    
    def check_hallucination(self, input, output, context):

        context_page_content = [doc.page_content for doc in context]
        resp_sentences = [sent.text.strip() for sent in nlp(output).sents] # spacy sentence tokenization
        scores = []
        for sent in resp_sentences:
            test_case = LLMTestCase(
                    input=input,
                    actual_output=sent,
                    retrieval_context=context_page_content
                )
            self.metric.measure(test_case)
            scores.append(self.metric.score)

        return resp_sentences, scores
    

    