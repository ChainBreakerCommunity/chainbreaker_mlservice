import spacy
from typing import List

def generate_bigrams(bigram_one: List[str], bigram_two: List[str]) -> List[str]:
    """
    This function takes to lists of strings and generates all possible combination between both.
    Input: List[str], List[str]
    Output: List[str] with all possible combinations.
    """
    bigrams = list()
    for i in range(len(bigram_one)):
      for j in range(len(bigram_two)):
        bigrams.append(bigram_one[i] + " " + bigram_two[j])
    return bigrams

def get_special_bigrams(doc: spacy.tokens.doc.Doc, bigrams: List[str], bigram_window: int, verbose = False) -> bool: 
    """
    This function looks for the existance of the bigrams in the given document.
    Input: doc (nlp object), bigrams: List[str], bigram_window (int)
    Output: True if some bigram is found, False otherwise
    """
    for bigram in bigrams:
        bi0, bi1 = bigram.split(" ")
        for i in range(0, len(doc)): 
            token = doc[i]
            token_str = token.text.lower()

            if bi0 == token_str:
                for j in range(0, bigram_window):
                  try:
                    next_j_token = doc[i + j]
                    if next_j_token.text.lower() == bi1:
                      if verbose:
                        print(bigram)
                      return True
                  except:
                    return False
    return False

def search_words(doc :spacy.tokens.doc.Doc, words: List[str], verbose = False) -> bool:
    """
    This functions looks for given pronouns in a document.
    Input: document (nlp object), pronuons (list of pronuons)
    Output: True if some pronoun is found, False otherwise
    """
    # For every token in the document.
    for token in doc:
        token_str = token.text.lower()

        # Check if token is in the list.
        if token_str in words:
            if verbose: 
              print(token_str)
            return True
    # Return false otherwise.
    return False

def search_sequence_of_words(text: str, sequences: List[str], verbose = False) -> bool:
    """
    This functions looks for a sequence of words in a text
    """
    # For every sequuence in sequences.
    for sequence in sequences:
      if sequence in text.lower():
        if verbose: 
          print(sequence)
        return True
    return False