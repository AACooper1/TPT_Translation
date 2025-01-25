import pyconll
from math import inf, exp, log
from torch.nn.functional import softmax, pad, kl_div
from torch import Tensor, stack, cumsum, arange, bucketize, zeros, tensor

from accelerate.utils import tqdm as acc_tqdm
from transformers import MarianTokenizer
from cupyx.scipy.stats import entropy

from pyconll.unit.sentence import Sentence

from preprocess import *

class TreePlantedHead:
    # On init, this class constructs the supervision matrices for the given data.
    def __init__(self, device, tokenizer: MarianTokenizer, treebank_path='data/parallel.conllu', λ=0.5, max_len=128):
        self.tokenizer = tokenizer
        self.λ = λ        
        self.treebank = None
        self.device = device
        
        # Load the ConLLU trees
        with open(treebank_path, "r") as in_file:
                treebank_raw = in_file.readlines()
        self.treebank = pyconll.load_from_resource(acc_tqdm(treebank_raw, leave=False))
        
        self.preprocessed = []

        for i in self.treebank:
            if "newpar" in i._meta:
                 self.preprocessed.append(' '.join(t.form for t in i))
            else:
                self.preprocessed[-1] += ' ' + ' '.join(t.form for t in i)
            
        
        self.treebank = self.treebank[:1000] #DEBUG

        # Convert the trees into adjacency matrix representations
        adjacency = {}
        for i in acc_tqdm(range(len(self.treebank)), desc="Adjacencies", leave=False, main_process_only=True):
            if "newpar" in self.treebank[i]._meta:
                 j=0
                 adjacency[len(adjacency)] = {}
            adjacency[len(adjacency) - 1].update(self.conll_to_adjacency(self.treebank[i], sentpos=j))
            j += 1

        del self.treebank
        
        # Now convert these into distance matrices
        distances = {}
        for i in acc_tqdm(list(adjacency.keys()), desc="Distances", leave=False, main_process_only=True):
            distance = self.adjacency_to_distance(adjacency[i])
            del adjacency[i]
            distances[i] = {j: distance[j] for j in distance if not (j.endswith('_root') and j[0].isnumeric())}

        self.supervision = {}
        self.token_steps = {}

        for i in acc_tqdm(distances.keys(), desc="Supervision", leave=False):
            self.supervision[i] = []
            n_tokens = [len(q) - (q.count(0) + q.count(4)) for q in [self.tokenizer.encode(p) for p in [q.split('_')[1] for q in distances[i].keys()]]]
            
            if sum(n_tokens) > 128: 
                b = 0
                s = 0
                while b < 128:
                    b += n_tokens[s]
                    s += 1
                n_tokens = n_tokens[:s - 1]
            self.token_steps[i] = n_tokens + [0] * (128 - len(n_tokens))

            for j in list(distances[i].keys()):
                if j.endswith('_root') and j[0].isnumeric():
                    del distances[i][j]
                else:
                    neg_dist = Tensor([-1 * i for i in list(distances[i][j].values())])
                    softmax_scores = softmax(neg_dist, dim=0).to(self.device)
                    self.supervision[i].append(softmax_scores)

            self.supervision[i] = stack(self.supervision[i])
            self.supervision[i] = pad(
                self.supervision[i],
                (0, 128 - self.supervision[i].shape[0], 0, 128 - self.supervision[i].shape[1]),
                value=0
            )
        pass
        self.distances = distances

        # '''

    def conll_to_adjacency(self, sentence: Sentence, sentpos=0):
        # Create a unidirectional adjacency matrix for the CoNLL tree.
        # Matrix is a mapping from (pointers to) tokens to the set of (pointers to) tokens dependent on them.
        # Special entry "root" points only to root node 
        adjacency = {}

        for token in sentence:
            form = token.id + "_" + token.form
            adjacency[form] = []
        
        for token in sentence:
            form = token.id + "_" + token.form
            if token.head == "0":
                # adjacency[str(sentpos) + "_root"] = [form]
                if not form in adjacency:
                    adjacency[form] = []
                # adjacency[form].append(str(sentpos) + "_root")
                continue
            elif token.head == None:
                continue
            head_tkn = sentence[sentence._ids_to_indexes[token.head]]
            head_form = head_tkn.id + "_" + head_tkn.form
            if not head_form in adjacency:
                adjacency[head_form] = []
            if not form in adjacency:
                adjacency[form] = []
            adjacency[head_form].append(form)
            adjacency[form].append(head_form)
                 

        return adjacency

    def adjacency_to_distance(self, sentence: dict):
        dist_matrix = {}
        # Floyd-Warshall Algorithm
        for u in sentence:
            dist_matrix[u] = {}
            for v in sentence: 
                # Initialize the entire array
                dist_matrix[u][v] = 1 if v in sentence[u] else len(sentence) # max out at the sentence length. Avoids fucky stuff with disparate clauses
            dist_matrix[u][u] = 1
        for k in dist_matrix:
            for i in dist_matrix:
                for j in dist_matrix:
                    if dist_matrix[i][j] > dist_matrix[i][k] + dist_matrix[k][j]:
                        dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
        return dist_matrix
    
    def token_weights_to_word_weights(self, sentence_id: int, token_weights: Tensor):
        token_steps = self.token_steps[sentence_id]
        token_steps = tensor(token_steps, device=token_weights.device)

        word_boundaries = cumsum(tensor(token_steps), dim=0)[:-1]

        token_indices = arange(token_weights.shape[0])
        word_indices = bucketize(token_indices, word_boundaries)

        word_weights = zeros((len(token_steps), len(token_steps)), device=token_weights.device)
        for i in range(len(token_steps)):
            for j in range(len(token_steps)):
                word_weights[i, j] = token_weights[word_indices == i][:, word_indices == j].sum()

        word_weights = word_weights / (word_weights.sum(dim=-1, keepdim=True) + 1e-12)

        return word_weights


    def calculate_tree_loss(self, nwp_attn: Tensor, ids: Tensor, batch_size: int):
        supervision_batch = stack([self.supervision[id.item()] for id in ids])

        # Mask out padding, etc.
        valid_mask = (nwp_attn.sum(dim=-1) > 0).unsqueeze(-1)

        # mask/normalize
        nwp_attn = nwp_attn * valid_mask
        supervision_batch = supervision_batch * valid_mask

        nwp_attn = nwp_attn / (nwp_attn.sum(dim=-1, keepdim=True) + 1e-12)
        supervision_batch = supervision_batch / (supervision_batch.sum(dim=-1, keepdim=True) + 1e-12)

        tp_loss = kl_div(
            nwp_attn.log(),
            supervision_batch,
            reduction="batchmean"
        )

        return tp_loss

