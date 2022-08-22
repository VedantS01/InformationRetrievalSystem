from util import *

# Add your import statements here
from nltk.stem.porter import PorterStemmer


class InflectionReduction:

    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of
                stemmed/lemmatized tokens representing a sentence
        """

        reducedText = None

        # Fill in code here
        porter_stem = PorterStemmer()
        reducedText = []

        for sentence in text:
            reducedSentence = []
            for token in sentence:
                reducedSentence.append(porter_stem.stem(token))
            reducedText.append(reducedSentence)

        return reducedText
