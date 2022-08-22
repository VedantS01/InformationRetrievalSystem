from pickletools import float8
from util import *

# Add your import statements here
from math import log10, sqrt
import pickle
import numpy as np
from sklearn.utils.extmath import randomized_svd


class GLSAH():

    def __init__(self, k: int = 400):
        self.index = None
        self.doc_IDs = None
        self.lsi = None
        self.k = k

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
        N_DOCS = len(docIDs)
        types = set([])
        length = {}
        bigram_index = {}
        for i in range(N_DOCS):
            doc = docs[i]
            id = docIDs[i]
            l = 0
            for sentence in doc:
                for token in sentence:
                    types.add(token)
                    idVals = index.get(token)
                    if idVals == None:
                        index[token] = {id: 1}
                        l += 1
                    else:
                        freqVal = idVals.get(id, 0)
                        if(freqVal == 0):
                            l += 1
                        idVals[id] = freqVal + 1

                    # HYBRID
                sen_len = len(sentence)
                j = 0
                while j < sen_len - 1:
                    token1 = sentence[j]
                    token2 = sentence[j+1]
                    type = (token1, token2)
                    idVals = bigram_index.get(type)
                    if idVals == None:
                        bigram_index[type] = {id: 1}
                    else:
                        freqVal = idVals.get(id, 0)
                        idVals[id] = freqVal + 1
                    j += 1
                    # END

            length[id] = l

        # MYCODE
        sum_dict = {}
        for key in bigram_index:
            values = bigram_index[key]
            sum = 0
            for doc in values:
                sum += values[doc]
            sum_dict[key] = sum

        sorted_sumdict = sorted(sum_dict, key=sum_dict.get, reverse=True)[
            :(len(sum_dict)*(25))//100]

        for key in sorted_sumdict:
            index[key] = bigram_index[key]
            types.add(key)
            for doc in bigram_index[key]:
                length[doc] += 1

        self.doc_IDs = docIDs
        self.index = index
        # ENDMYCODE

        for type in index:
            item0 = index[type]
            for doc_id in item0:
                index[type][doc_id] = log10(
                    index[type][doc_id] + 1) / log10(length[doc_id])
        self.doc_IDs = docIDs
        self.index = index
        N_TERMS = len(index)
        tdmat = np.zeros((N_TERMS, N_DOCS), dtype=np.float16)
        get_term_at_i = list(types)
        for i in range(N_TERMS):
            term = get_term_at_i[i]
            idf = log10(N_DOCS/len(self.index[term]))
            for j in range(N_DOCS):
                try:
                    tdmat[i][j] = np.float16(index[term][j+1])
                except KeyError:
                    tdmat[i][j] = np.float16(0)

        # print("Doing")
        U, S, Vh = randomized_svd(tdmat,
                                  n_components=self.k,
                                  n_iter=5,
                                  random_state=None)
        # print("Done")
        # U, S, Vh = svd(tdmat)
        k = self.k
        U1 = U[:, :k]
        S1 = S[:k]
        Vh1 = Vh[:k]
        D1 = np.diag(S1)
        recon = U1 @ D1 @ Vh1
        del(U)
        del(S)
        del(Vh)
        del(U1)
        del(S1)
        del(Vh1)
        del(D1)
        import gc
        gc.collect()
        lsi = {}
        for t in range(N_TERMS):
            lsi[get_term_at_i[t]] = {}
            for d in range(N_DOCS):
                lsi[get_term_at_i[t]][d+1] = recon[t][d]
        self.lsi = lsi

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
        t = len(self.lsi)

        doc_vectors = {doc: {token: 0 for token in self.lsi}
                       for doc in self.doc_IDs}

        for token in self.lsi:
            idf = log10(N/len(self.index[token]))
            for doc_id in self.lsi[token]:
                # try:
                doc_vectors[doc_id][token] = self.lsi[token][doc_id]*idf
                # doc_vectors[doc_id][token] = self.lsi[token][doc_id]
                # except:
                #     print("issue in", token)

        doc_magnitudes = {}
        for doc_id in doc_vectors:
            doc_magnitudes[doc_id] = sqrt(
                sum([doc_vectors[doc_id][token]**2 for token in doc_vectors[doc_id]]))

        for query in queries:
            sim = {}
            dot_prods = {doc_id: 0 for doc_id in self.doc_IDs}
            query_magnitude = 0
            query_tokens = {}
            for sentence in query:
                for token in sentence:
                    query_tokens[token] = query_tokens.get(token, 0) + 1

                sen_len = len(sentence)
                j = 0
                while j < sen_len - 1:
                    token1 = sentence[j]
                    token2 = sentence[j+1]
                    type = (token1, token2)
                    query_tokens[type] = query_tokens.get(type, 0) + 1
                    j += 1
            for token in query_tokens:
                query_magnitude += (query_tokens[token])**2
                if self.lsi.__contains__(token):
                    id_vals = self.lsi[token]
                else:
                    id_vals = None
                if id_vals != None:
                    idf = log10(N/len(self.index[token]))
                    for doc_id in id_vals:
                        dot_prods[doc_id] += query_tokens[token] * \
                            idf*doc_vectors[doc_id][token]
                    # for doc_id in id_vals:
                    #     dot_prods[doc_id] += query_tokens[token] * \
                    #         doc_vectors[doc_id][token]
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
