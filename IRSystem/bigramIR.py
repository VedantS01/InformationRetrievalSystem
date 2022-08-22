from util import *

# Add your import statements here
from math import log10, sqrt


class BigramIR():

    def __init__(self):
        self.index = None
        self.doc_IDs = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
                A list of lists of lists where each sub-list is
                a document and each sub-sub-list is a sentence of the document
        arg2 : list
                A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = None

        # Fill in code here
        index = {}
        for i in range(len(docIDs)):
            doc = docs[i]
            id = docIDs[i]
            for sentence in doc:
                sen_len = len(sentence)
                j = 0
                while j < sen_len - 1:
                    token1 = sentence[j]
                    token2 = sentence[j+1]
                    type = (token1, token2)
                    idVals = index.get(type)
                    if idVals == None:
                        index[type] = {id: 1}
                    else:
                        freqVal = idVals.get(id, 0)
                        idVals[id] = freqVal + 1
                    j += 1

        self.doc_IDs = docIDs
        self.index = index

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
                A list of lists of lists where each sub-list is a query and
                each sub-sub-list is a sentence of the query


        Returns
        -------
        list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []

        # Fill in code here
        N = len(self.doc_IDs)
        t = len(self.index)

        doc_vectors = {doc: {token: 0 for token in self.index}
                       for doc in self.doc_IDs}

        for type in self.index:
            idf = log10(N/len(self.index[type]))
            for doc_id in self.index[type]:
                doc_vectors[doc_id][type] = self.index[type][doc_id]*idf

        doc_magnitudes = {}
        for doc_id in doc_vectors:
            doc_magnitudes[doc_id] = sqrt(
                sum([doc_vectors[doc_id][type]**2 for type in doc_vectors[doc_id]]))

        for query in queries:
            sim = {}
            dot_prods = {doc_id: 0 for doc_id in self.doc_IDs}
            query_magnitude = 0
            query_types = {}
            for sentence in query:
                sen_len = len(sentence)
                j = 0
                while j < sen_len - 1:
                    token1 = sentence[j]
                    token2 = sentence[j+1]
                    type = (token1, token2)
                    query_types[type] = query_types.get(type, 0) + 1
                    j += 1
            for type in query_types:
                query_magnitude += (query_types[type]*idf)**2
                if self.index.__contains__(type):
                    id_vals = self.index[type]
                else:
                    id_vals = None
                if id_vals != None:
                    idf = log10(N/len(self.index[type]))
                    for doc_id in id_vals:
                        dot_prods[doc_id] += query_types[type] * \
                            idf*doc_vectors[doc_id][type]
            query_magnitude = sqrt(query_magnitude)
            for doc_id in dot_prods:
                if (doc_magnitudes[doc_id]*query_magnitude) > 0:
                    sim[doc_id] = dot_prods[doc_id] / \
                        (doc_magnitudes[doc_id]*query_magnitude)
                else:
                    sim[doc_id] = 0

            doc_IDs_ordered.append(
                sorted([doc_id for doc_id in self.doc_IDs], key=sim.get, reverse=True))

        return doc_IDs_ordered
