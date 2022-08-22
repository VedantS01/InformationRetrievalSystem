from unittest import result
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from LSA import LSA
from Glasgow import Glasgow
from GLSA import GLSA
from NormalizedWeights import NormalizedWeights
from evaluation import Evaluation
from bigramIR import BigramIR
from hybridIR import HybridIR
from GLSA_HYBRID import GLSAH
from Glasgow_HYBRID import GlasgowH

from sys import version_info
import os
import argparse
import json
import matplotlib.pyplot as plt
import pickle

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")


class SearchEngine:

    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()
        if(self.args.term_weighting == 'naive'):
            if(self.args.n_gram == 'hybrid'):
                self.informationRetriever = HybridIR()
            elif(self.args.n_gram):
                self.informationRetriever = BigramIR()
            else:
                self.informationRetriever = InformationRetrieval()
        elif(self.args.term_weighting == 'lsa'):
            self.informationRetriever = LSA(k=int(self.args.dim))
        elif(self.args.term_weighting == 'glasgow'):
            if(self.args.n_gram == 'hybrid'):
                self.informationRetriever = GlasgowH()
            else:
                self.informationRetriever = Glasgow()
        elif(self.args.term_weighting == 'glasgow_lsa'):
            if(self.args.n_gram == 'hybrid'):
                self.informationRetriever = GLSAH(k=int(self.args.dim))
            else:
                self.informationRetriever = GLSA(k=int(self.args.dim))
        elif(self.args.term_weighting == 'normalize'):
            self.informationRetriever = NormalizedWeights()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(
            self.args.out_folder + "segmented_queries.txt", 'w'))
        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(
            self.args.out_folder + "tokenized_queries.txt", 'w'))
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(
            self.args.out_folder + "reduced_queries.txt", 'w'))
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(
            self.args.out_folder + "stopword_removed_queries.txt", 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(
            self.args.out_folder + "segmented_docs.txt", 'w'))
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(
            self.args.out_folder + "tokenized_docs.txt", 'w'))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(
            self.args.out_folder + "reduced_docs.txt", 'w'))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(
            self.args.out_folder + "stopword_removed_docs.txt", 'w'))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP 
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """
        result = ''
        if(not os.path.exists(self.args.out_folder+"/")):
            os.mkdir(self.args.out_folder)
        # Read queries
        queries_json = json.load(
            open(args.dataset + "cran_queries.json", 'r'))[:]
        query_ids, queries = [item["query number"] for item in queries_json], \
            [item["query"] for item in queries_json]
        # Process queries
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
            [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for each query
        doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

        # Read relevance judements
        qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)
            recall = self.evaluator.meanRecall(
                doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print("Precision, Recall and F-score @ " +
                  str(k) + " : " + str(precision) + ", " + str(recall) +
                  ", " + str(fscore))
            result += "Precision, Recall and F-score @ " +\
                  str(k) + " : " + str(precision) + ", " + str(recall) +\
                  ", " + str(fscore) + '\n'
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +
                  str(k) + " : " + str(MAP) + ", " + str(nDCG))
            result += "MAP, nDCG @ " +\
                  str(k) + " : " + str(MAP) + ", " + str(nDCG) + '\n'

        # Plot the metrics and save plot
        print("Precisions are = "+str(precisions))
        result += "Precisions are = "+str(precisions) + '\n'
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(self.args.out_folder + "eval_plot.png")
            
        with open(self.args.out_folder + "result.txt","w") as f:
            f.write(result)

    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        # Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
            [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        ### UPDATE
        # SAVE PROCESSED DOCS TO PLAY WITH
        json.dump(processedDocs, open('play/processed_docs.json', 'w'))

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='main.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset', default="cranfield/",
                        help="Path to the dataset folder")
    parser.add_argument('-out_folder', default="output/",
                        help="Path to output folder")
    parser.add_argument('-segmenter', default="punkt",
                        help="Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer',  default="ptb",
                        help="Tokenizer Type [naive|ptb]")
    parser.add_argument('-term_weighting',  default="glasgow_lsa",
                        help="Use LSI indexing [naive|lsa|glasgow|glasgow_lsa|normalize]")
    parser.add_argument('-n_gram',  default="unigram",
                        help="Use LSI indexing [hybrid|bigram|unigram]")
    parser.add_argument('-dim',  default=400,
                        help="The 'k' for LSA in simple LSA or GLASGOW LSA")
    parser.add_argument('-custom', action="store_true",
                        help="Take custom query as input")

    # Parse the input arguments
    args = parser.parse_args()

    # Create an instance of the Search Engine
    searchEngine = SearchEngine(args)

    # Either handle query from user or evaluate on the complete dataset
    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
