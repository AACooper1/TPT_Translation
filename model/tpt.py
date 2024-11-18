import pyconll
import os
import random
from math import inf
from collections import OrderedDict
from tabulate import tabulate
import re

import pyconll.tree


    
def iter_over_tree(tree: pyconll.tree, adjacency=None, debug=False):
    if adjacency is None:
        adjacency = OrderedDict()

    token = tree._data.form + tree._data.id

    no_id = re.sub(r'[0-9]+', '', token)
    
    if no_id not in adjacency:
        token = no_id

    if debug:
        print(f"{token} --> {[child._data.form for child in tree._children]}")
    
    adjacency[token] = []

    for child_tree in tree._children:
        
        child = child_tree._data

        adjacency = iter_over_tree(child_tree, adjacency=adjacency, debug=debug)

        if child.form + child.id in adjacency:
            adjacency[token].append(child.form + child.id)
        else:
            adjacency[token].append(child.form)

    return adjacency



def pyconll_to_distance_matrix(sentence, decoder=False, debug=False):
    """Transforms the syntactic tree to a syntactic distance matrix using the Floyd-Warshall algorithm.

    Args:
        tree: The pyconll.Tree instance to transform.
        decoder: If True, the matrix will be lower triangular (all entries with j >= i will be set to 0.)
    
    Returns:
        A 2D matrix stored as a list of lists representing the distance between two words.
    """

    adjacency = iter_over_tree(sentence.to_tree(), debug=debug)

    tokens = []
    for token in sentence._tokens:
        if token.form + token.id in adjacency:
            tokens.append(token.form + token.id)
        elif token.form in adjacency:
            tokens.append(token.form)

    if decoder:
        indices = {}
        for i in range(len(tokens)):
            indices[tokens[i]] = i

    dist = OrderedDict()

    # Initialize the adjacency matrix
    for i in tokens:
        no_id = re.sub(r'[0-9]+', '', i)
        
        if no_id not in dist:
            i = no_id

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
                if decoder and indices[i] < indices[j]:
                    dist[i][j] = 0
                elif dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # That should do it
    return dist

# The following code was kindly provided by the good folks at chatgpt.com : )
"""
Remember to delete this later.
"""
def prettyprint_nested_dict(nested_dict):
    # Collect all unique row and column headers
    rows = list(nested_dict.keys())
    cols = list(nested_dict.keys())
    
    # Create the 2D array with an additional row and column for headers
    table = [[""] + cols]  # Add column headers
    for row in rows:
        # Create a row starting with the row header
        row_data = [nested_dict[row].get(col, 0) for col in cols]  # Fill with values or default (0)
        table.append([row] + row_data)

    max_cell_width = max(len(str(item)) for row in table for item in row)
    
    # Generate the header separator (a line of dashes)
    separator = "---".join("-" * max_cell_width for _ in table[0])

    # Pretty-print the table
    for i, row in enumerate(table):
        print(" | ".join(str(item).rjust(max_cell_width) for item in row))
        if i == 0:  # Print the separator after the headers
            print(separator)
"""
Remember to delete this later.
"""



if __name__ == "__main__":
    treebank_filepath = 'UD/UD_Latin-Perseus/la_perseus-ud-train.conllu'
    treebank = pyconll.load_from_file(treebank_filepath)
    treebank_filepath = 'UD/UD_Latin-CIRCSE/la_circse-ud-test.conllu'
    treebank += pyconll.load_from_file(treebank_filepath)
    # sentence_example = treebank._sentences[1026]
    # sentence_example = random.choice(treebank)
    # tree_example = sentence_example.to_tree()

    # print(sentence_example.text)

    errors = 0
    for sentence in treebank:
        try:
            distance_matrix = pyconll_to_distance_matrix(sentence, decoder=False, debug=True)

            prettyprint_nested_dict(distance_matrix)
        except:
            errors+= 1
    
    print("errors:", errors, "out of", len(treebank))

    pass

