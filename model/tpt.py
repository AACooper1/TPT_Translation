import pyconll
import os
import random
from math import inf, exp
from collections import OrderedDict
import re
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from transformers import MT5Tokenizer
import itertools
import torch
from scipy.special import kl_div

import pyconll.tree

class TreePlantedHead:

    decoder = False
    next_word_attn_weights = None
    supervision_attn_weights = None

    # THIS IS JUST SOFTMAX IT TOOK ME THIS LONG TO REALIZE THIS IS JUST FUCKING SOFTMAX AAAAAAA I'M GOING TO KILL MYSELF
    def distance_matrix_to_supervision(self, dist: OrderedDict, decoder=False):
        s = np.zeros([len(dist), len(dist)])

        i = 0
        for from_word in dist:
            i+= 1
            j = 0

            for to_word in dist[from_word]:
                if j >= i and decoder:
                    s[i - 1][j] = 0
                else:
                    numer = (exp(-1 * dist[from_word][to_word]))                        # e^{D[attn_from][attn_to]}
                    # divis = sum([exp(-dist[from_word][k]) for k in dist[from_word]])     # sum(e^{D[attn_from][attn_to]})
                    denom = 0
                    for k in dist[from_word]:
                        denom += exp(-dist[from_word][k])
                        
                        if decoder:
                            if k == from_word:
                                break

                    s[i - 1][j] = numer/denom

                j+=1

        return torch.tensor(s)

    def tree_to_adjacency(self, tree: pyconll.tree=None, sentence=None, adjacency=None, debug=False):
        if adjacency is None:
            adjacency = OrderedDict()
            for token in sentence._tokens:
                if token.form in adjacency:
                    adjacency[token.form + token.id] = []
                else:
                    adjacency[token.form] = []
                    
                if debug:
                    print(f"{token} --> {[child._data.form for child in tree._children]}")

        if tree is None:
            tree = sentence.to_tree()

        token = tree._data.form + tree._data.id
        if token not in adjacency:
            token = tree._data.form

        for child_tree in tree._children:
            
            child = child_tree._data

            adjacency = self.tree_to_adjacency(tree=child_tree, sentence=None, adjacency=adjacency, debug=debug)

            if child.form + child.id in adjacency:
                adjacency[token].append(child.form + child.id)
            else:
                adjacency[token].append(child.form)

        return adjacency

    def pyconll_to_distance_matrix(self, sentence, debug=False):
        """Transforms the syntactic tree to a syntactic distance matrix using the Floyd-Warshall algorithm.

        Args:
            tree: The pyconll.Tree instance to transform.
            decoder: If True, the matrix will be lower triangular (all entries with j >= i will be set to 0.)
        
        Returns:
            A 2D matrix stored as a list of lists representing the distance between two words.
        """

        adjacency = self.tree_to_adjacency(sentence=sentence, debug=debug)

        tokens = []
        for token in sentence._tokens:
            if token.form + token.id in adjacency:
                tokens.append(token.form + token.id)
            elif token.form in adjacency:
                tokens.append(token.form)

        dist = OrderedDict()

        # Initialize the distance matrix
        for i in tokens:
            # no_id = token.form
            
            # if no_id not in dist:
            #     i = no_id

            dist[i] = {}
            
            for j in tokens:
                dist[i][j] = inf
            dist[i][i] = 0
        for node in dist:
            for neighbor in adjacency[node]:
                dist[node][neighbor] = 1
                dist[neighbor][node] = 1

        # Now the actual algorithm
        for k in adjacency:
            for i in adjacency:
                for j in adjacency:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Check for infs; if there are any disconnected nodes, it'll stay at inf and fuck up the supervision equation
        # (This happens for tokens with *-que*, for instance, as they're in sentences._tokens twice)
        for i in dist:
            for j in dist[i].copy():
                if dist[i][j] == inf:
                    del dist[i][j]

        for i in dist.copy():
            if sum(dist[i].values()) == 0 and len(dist) != 1:
                del dist[i]

        # That should do it
        return dist

    def group_subwords(self, input: str, tokenizer):
        subword_tokens = tokenizer.tokenize(text=input)
        word_tokens = []
        for i in range(len(subword_tokens)):
            if subword_tokens[i] == "▁":
                subword_tokens[i+1] = subword_tokens[i] + subword_tokens[i+1]
            elif not re.match(r"^[A-Za-z0-9]",subword_tokens[i]) or i == 0:
                word_tokens.append([tokenizer.convert_tokens_to_ids(subword_tokens[i])])
            else:
                word_tokens[-1].append(tokenizer.convert_tokens_to_ids(subword_tokens[i]))
        return word_tokens

    def subtoken_weights_to_word_weights(self, tokens_to_words, token_weights, batch_size=1):
        # change this later
        token_weights = token_weights[0]
        subword_sums = []
        l = 0
        for i in range(len(tokens_to_words)):
            sum = 0
            subword_sums.append([0 for z in tokens_to_words])
            for _l in range(len(tokens_to_words[i])):
                m = 0
                for j in range(len(tokens_to_words)):
                    for _m in range(len(tokens_to_words[j])):
                        subword_sums[i][j] += token_weights[l][m].item()
                        m+=1
                l+=1
        # subword_sums = np.array(subword_sums)
        del l, m
        
        word_level_weights = [[0 for i in j] for j in subword_sums]
        
        for i in range(len(subword_sums)):
            for j in range(len(subword_sums[i])):
                sum = 0
                word_level_weights[i][j] = 0
                for k in range(len(subword_sums[i])):
                    sum += subword_sums[i][k]
                if sum == 0:
                    sum = 1
                word_level_weights[i][j] = subword_sums[i][j] / sum
        
        for i in range(len(word_level_weights)):
            assert len(word_level_weights[i]) == len(word_level_weights[0]), f"List at index {i} has length {len(word_level_weights[i])}, expected {len(word_level_weights[0])}"

        word_level_weights = torch.tensor(word_level_weights)

        return word_level_weights

    def calculate_tree_loss(supervision_weights, word_weights):
        try:
            assert len(word_weights) == len(supervision_weights), f"The tree-planting weight matrix has length {len(supervision_weights)}, while the next-word prediction weight has length {len(word_weights)}. Are you sure you passed in the word-level weights?"
            for i in range(len(supervision_weights)):
                assert np.isclose(torch.sum(supervision_weights[i]).item(), 1.0), f"The sum of the supervision weights at index {i} is {torch.sum(supervision_weights[i]).item()}, when it should be approximately 1."
                assert np.isclose(torch.sum(word_weights[i]).item(), 1.0), f"The sum of the word weights at index {i} is {torch.sum(word_weights[i]).item()}, when it should be approximately 1."
        except Exception as e:
            return e
        for i in range(1, len(word_weights) - 1):
            sum = torch.sum(kl_div(supervision_weights[i], word_weights[i]))
        
        return sum / (len(word_weights) - 1)

    def calculate_epoch_loss(next_word_losses, tree_losses):
        return NotImplemented

if __name__ == "__main__":
    test_tokenizer = True

    treebank_filepath = 'UD/UD_Latin-Perseus/la_perseus-ud-train.conllu'
    treebank = pyconll.load_from_file(treebank_filepath)
    treebank_filepath = 'UD/UD_Latin-CIRCSE/la_circse-ud-test.conllu'
    treebank += pyconll.load_from_file(treebank_filepath)
    # sentence_example = treebank._sentences[1026]
    # sentence_example = random.choice(treebank)
    # tree_example = sentence_example.to_tree()

    # print(sentence_example.text)
    tph = TreePlantedHead()

    if test_tokenizer:
        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
        word_tokens = []
        errors = 0
        l_errors = []

        for i in tqdm(range(len(treebank))):
            try:
                word_tokens.append(tph.group_subwords(treebank[i].text, tokenizer))
                # for j in word_tokens[i]:
                    # the = ''.join(j).replace('▁', '') 
                    # assert the in [token.form for token in treebank[i]]
                tph.subtoken_weights_to_word_weights(word_tokens)
            except Exception as e:
                errors += 1
                l_errors.append((word_tokens, treebank[i].text))
        print("Num errors:", errors)        
        exit()

    errors = 0
    dist_matrices = []
    superv_matrices = []
    for sentence in tqdm(treebank):
        try:
            distance_matrix = tph.pyconll_to_distance_matrix(sentence, debug=False)
            superv_matrix = tph.distance_matrix_to_supervision(distance_matrix)

            dist_matrices.append(distance_matrix)
            superv_matrices.append(superv_matrix)

            # prettyprint_nested_dict(distance_matrix)
        except Exception as e:
            errors+= 1
            print("Error", errors)
            raise(e)
    
    print("errors:", errors, "out of", len(treebank))

    errors = 0
    dist_matrices = []
    superv_matrices = []
    for sentence in tqdm(treebank):
        try:
            distance_matrix = tph.pyconll_to_distance_matrix(sentence, debug=False)
            superv_matrix = tph.distance_matrix_to_supervision(distance_matrix, decoder=True)

            dist_matrices.append(distance_matrix)
            superv_matrices.append(superv_matrix)

            # prettyprint_nested_dict(distance_matrix)
        except Exception as e:
            errors+= 1
            print("Error", errors)
            raise(e)
    
    print("errors:", errors, "out of", len(treebank))

