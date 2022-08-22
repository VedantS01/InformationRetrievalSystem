from util import *

from nltk.tokenize import TreebankWordTokenizer as tk
import re


class Tokenization():

    # SUBS for naive approach
    # replace obvious word-breakers with whitespace
    SUBS = [
        (re.compile(r'\.\.\.'), r' '),
        (re.compile(r"^\""), r" "),
        (re.compile(r"(``)"), r" "),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 "),
        (re.compile(r'[-?!;\(\)\[\]\{\}]'), r' '),
        (re.compile(r'[@#$%&]'), r' g<0> ')
    ]

    # periods in the end of sentence breaks words, others do not, unless they are at the ends of the word already. Since we are working on sentence segmented text, life is much easier.
    SUBS.append(
        (re.compile(r'\.\s'), r' ')
    )
    # for commas seperating numbers, word doesn't end, but it does when it is followed by a space.
    SUBS.append(
        (re.compile(r'[\,\:]\s'), r' ')
    )
    # differentiation in single quotes being used as quotes or apostrophe marks generally can't be quaranteed, but we will introduce a few rules to make things easier.
    SUBS.append(
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]) "), r"\1 \2 ")
    )  # note here that this is not the only way apostrophes are used, but a major case of possesives can be handled like this.

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []

        # Fill in code here
        for string in text:
            string = string.casefold()
            for regexp, substitution in self.SUBS:
                string = regexp.sub(substitution, string)
            tokenizedText.append(string.split())

        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []

        # Fill in code here
        for sentence in text:
            tokenizedText.append(tk().tokenize(sentence))

        return tokenizedText
