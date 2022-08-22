from util import *

# Add your import statements here
from nltk.corpus import stopwords


class StopwordRemoval():

    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence with stopwords removed
        """

        stopwordRemovedText = None

        # Fill in code here
        # download_sw()
        try:
            sw_eng = stopwords.words('english')
        except LookupError:
            download_sw()
            sw_eng = stopwords.words('english')
        except:
            print("ISSUE HERE")
            raise RuntimeError
        stopwordRemovedText = []

        for sentence in text:
            stopwordRemovedSentence = []
            for token in sentence:
                if token.lower() not in sw_eng:
                    stopwordRemovedSentence.append(token)
            stopwordRemovedText.append(stopwordRemovedSentence)

        return stopwordRemovedText
