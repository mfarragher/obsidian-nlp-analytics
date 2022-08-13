import numpy as np
import scipy.sparse as ss
import corextopic.corextopic as ct
import corextopic.vis_topic as vt
from sklearn.feature_extraction.text import CountVectorizer


class VaultNLP:
    def __init__(self, vault):
        self._docs = list(vault.readable_text_index.values())
        self._note_names = list(vault.readable_text_index.keys())
        # allow these to be changed in case of further data cleaning:
        self.word_matrix = None
        self.words = []
        
        self.topic_model = None
        self.anchor_topic_model = None
        self.anchor_words = []

    def generate_sparse_word_matrix(self):
        # vectorise all docs in the vault:
        vectorizer = CountVectorizer(stop_words='english',
                                     max_features=20000, binary=True)
        doc_word = vectorizer.fit_transform(self._docs)
        doc_word = ss.csr_matrix(doc_word)

        # list of words to label matrix columns:
        words = list(np.asarray(vectorizer.get_feature_names_out()))
        # remove numeric strings:
        not_digit_inds = [ind for ind, word in enumerate(words)
                          if not word.isdigit()]
        doc_word = doc_word[:, not_digit_inds]
        words = [word for ind, word in enumerate(words)
                 if not word.isdigit()]
        self.words = words
        self.word_matrix = doc_word
        pass

    def fit_topic_model(self, n_hidden=18, max_iter=200, verbose=False, seed=1):
        topic_model = ct.Corex(n_hidden=n_hidden, words=self.words,
                               max_iter=max_iter, verbose=verbose, seed=seed)

        topic_model.fit(self.word_matrix, words=self.words,
                        # note name, rather than content:
                        docs=self._note_names)
        self._topic_model = topic_model
        pass

    def fit_anchored_topic_model(self, n_hidden=18, max_iter=200, verbose=False,
                                 anchors=None, anchor_strength=7, seed=1):
        if anchors is None:
            anchors = self.anchor_words

        anchored_topic_model = ct.Corex(
            n_hidden=n_hidden, words=self.words,
            max_iter=max_iter, verbose=verbose, seed=seed)

        anchored_topic_model.fit(self.word_matrix, words=self.words,
                                 # note name, rather than content:
                                 docs=self._note_names,
                                 anchors=anchors,
                                 anchor_strength=anchor_strength)

        self.anchor_topic_model = anchored_topic_model
        pass
