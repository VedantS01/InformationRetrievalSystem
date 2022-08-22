from util import *

# Add your import statements here
from math import log2


class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The precision value as a number between 0 and 1
        """
        precision = -1
        true_positives = 0
        false_positives = 0
        if len(query_doc_IDs_ordered) == 0:
            return -1
        for i in range(min(k, len(query_doc_IDs_ordered))):
            flag = 0
            id = query_doc_IDs_ordered[i]
            for relevantId in true_doc_IDs:
                if str(id) == str(relevantId):
                    flag = 1
                    true_positives += 1
                    break
            if flag == 0:
                false_positives += 1
        # Fill in code here
        precision = true_positives/k
        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean precision value as a number between 0 and 1
        """
        meanPrecision = -1
        sumPrecision = 0
        totalCount = 0
        if len(doc_IDs_ordered) != len(query_ids):
            return -1
        for i in range(len(query_ids)):
            relevant_docs = []
            for j in range(len(qrels)):
                if qrels[j]['query_num'] == str(query_ids[i]):
                    relevant_docs.append(qrels[j]['id'])
            sumPrecision += self.queryPrecision(
                doc_IDs_ordered[i], query_ids[i], relevant_docs, k)
            totalCount += 1
        # Fill in code here
        meanPrecision = sumPrecision/totalCount
        return meanPrecision

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The recall value as a number between 0 and 1
        """

        recall = -1
        true_positives = 0
        false_positives = 0
        if len(true_doc_IDs) == 0:
            return -1
        for i in range(min(k, len(query_doc_IDs_ordered))):
            flag = 0
            id = query_doc_IDs_ordered[i]
            for relevantId in true_doc_IDs:
                if str(id) == str(relevantId):
                    flag = 1
                    true_positives += 1
                    break
            if flag == 0:
                false_positives += 1
        # Fill in code here
        recall = true_positives/(len(true_doc_IDs))
        # Fill in code here

        return recall

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean recall value as a number between 0 and 1
        """

        meanRecall = -1
        sumRecall = 0
        totalCount = 0
        if len(doc_IDs_ordered) != len(query_ids):
            return -1
        for i in range(len(query_ids)):
            relevant_docs = []
            for j in range(len(qrels)):
                if qrels[j]['query_num'] == str(query_ids[i]):
                    relevant_docs.append(qrels[j]['id'])
            sumRecall += self.queryRecall(
                doc_IDs_ordered[i], query_ids[i], relevant_docs, k)
            totalCount += 1
        # Fill in code here
        meanRecall = sumRecall/totalCount
        return meanRecall

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The fscore value as a number between 0 and 1
        """

        fscore = -1

        p = self.queryPrecision(query_doc_IDs_ordered,
                                query_id, true_doc_IDs, k)
        r = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

        if p == 0 or r == 0:
            fscore = 0
        else:
            fscore = 2*p*r/(p+r)

        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean fscore value as a number between 0 and 1
        """

        meanFscore = -1
        sum = 0
        N = len(doc_IDs_ordered)

        for i in range(N):
            query_doc_IDs_ordered = doc_IDs_ordered[i]
            query_id = query_ids[i]
            res = []
            for item in qrels:
                if(str(item['query_num']) == str(query_id)):
                    res.append(item)
            res = sorted(res, key=lambda x: x['position'])
            true_doc_IDs = [int(item['id']) for item in res]

            sum += self.queryFscore(query_doc_IDs_ordered,
                                    query_id, true_doc_IDs, k)
        meanFscore = sum / N
        return meanFscore

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of dictionaries (qrels)
        arg4 : int
                The k value

        Returns
        -------
        float
                The nDCG value as a number between 0 and 1
        """

        nDCG = -1

        num_docs = len(query_doc_IDs_ordered)
        if(num_docs < k):
            print("Insufficient docs retrieved, failing evaluation")
            return -1
        rel_vals = {}
        rel_docs = []
        for doc in true_doc_IDs:
            doc_id = int(doc["id"])
            rel_value = 5 - doc["position"]
            rel_vals[int(doc_id)] = rel_value
            rel_docs.append(int(doc_id))
        DCG = 0
        for i in range(1, k+1):
            doc_ID = int(query_doc_IDs_ordered[i-1])
            if doc_ID in rel_docs:
                rel_value = rel_vals[doc_ID]
                DCG += rel_value / log2(i+1)
        ordered_vals = sorted(rel_vals.values(), reverse=True)
        num_docs = len(ordered_vals)
        IDCG = 0
        for i in range(1, min(num_docs, k)+1):
            rel_value = ordered_vals[i-1]
            IDCG += rel_value / log2(i+1)
        if(IDCG == 0):
            print("IDCG is 0, failing evaluation")
            return -1
        nDCG = DCG/IDCG

        return nDCG

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean nDCG value as a number between 0 and 1
        """

        meanNDCG = -1

        sum = 0
        N = len(doc_IDs_ordered)

        for i in range(N):
            query_doc_IDs_ordered = doc_IDs_ordered[i]
            query_id = query_ids[i]
            res = []
            for item in qrels:
                if(str(item['query_num']) == str(query_id)):
                    res.append(item)
            res = sorted(res, key=lambda x: x['position'])
            true_doc_IDs = [
                {"id": item['id'], "position":item['position']} for item in res]

            sum += self.queryNDCG(query_doc_IDs_ordered,
                                  query_id, true_doc_IDs, k)
        meanNDCG = sum / N

        return meanNDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The average precision value as a number between 0 and 1
        """

        avgPrecision = -1

        # Fill in code here
        relPrecisions = []
        if len(query_doc_IDs_ordered) == 0 or k < 1:
            return -1
        for i in range(min(k, len(query_doc_IDs_ordered))):
            if str(query_doc_IDs_ordered[i]) in true_doc_IDs:
                relPrecisions.append(self.queryPrecision(
                    query_doc_IDs_ordered, query_id, true_doc_IDs, i+1))

        if len(relPrecisions) == 0:
            avgPrecision = 0
        else:
            avgPrecision = sum(relPrecisions)/len(relPrecisions)

        return avgPrecision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The MAP value as a number between 0 and 1
        """

        meanAveragePrecision = -1

        # Fill in code here
        qap = []
        for i in range(len(query_ids)):
            relevant_docs = []
            for j in range(len(q_rels)):
                if str(q_rels[j]['query_num']) == str(query_ids[i]):
                    relevant_docs.append(q_rels[j]['id'])
            qap.append(self.queryAveragePrecision(
                doc_IDs_ordered[i], query_ids[i], relevant_docs, k))
        meanAveragePrecision = sum(qap)/len(qap)

        return meanAveragePrecision
