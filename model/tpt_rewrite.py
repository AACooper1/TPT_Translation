import pyconll
from math import inf, exp, log
from torch.nn.functional import softmax, pad
from torch import Tensor, stack, count_nonzero, nan_to_num

from accelerate.utils import tqdm as acc_tqdm
from transformers import MarianTokenizer
from cupyx.scipy.stats import entropy

from pyconll.unit.sentence import Sentence

from preprocess import *

class TreePlantedHead:
    # On init, this class constructs the supervision matrices for the given data.
    def __init__(self, device, tokenizer: MarianTokenizer, treebank_path='data/parallel.conllu', λ=0.5):
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
            
        
        # self.treebank = self.treebank[:10000] #DEBUG

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

        for i in acc_tqdm(distances.keys(), leave=False):
            self.supervision[i] = []
            n_tokens = [len(q) - (q.count(0) + q.count(4)) for q in [self.tokenizer.encode(p) for p in [q.split('_')[1] for q in distances[i].keys()]]]
            
            if sum(n_tokens) > 64: 
                b = 0
                s = 0
                while b < 64:
                    b += n_tokens[s]
                    s += 1
                n_tokens = n_tokens[:s - 1]
            self.token_steps[i] = n_tokens + [0] * (64 - len(n_tokens))

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
                (0, 64 - self.supervision[i].shape[0], 0, 64 - self.supervision[i].shape[1]),
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
                dist_matrix[u][v] = 1 if v in sentence[u] else inf
            dist_matrix[u][u] = 1
        for k in dist_matrix:
            for i in dist_matrix:
                for j in dist_matrix:
                    if dist_matrix[i][j] > dist_matrix[i][k] + dist_matrix[k][j]:
                        dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
        return dist_matrix
    
    def token_weights_to_word_weights(self, sentence_id: int, token_weights: Tensor):
        token_steps = self.token_steps[sentence_id]
        word_weights = []

        tkn_l = 0
        tkn_m = 0

        # Weight from each word...
        for wd_i in range(len(token_steps)):
            tkn_m = 0
            word_weights.append([])
            # ...to each word
            for wd_j in range(len(token_steps)):
                i_length = token_steps[wd_i]
                j_length = token_steps[wd_j]
                sum_i_j = 0
                word_weights[wd_i].append([])

                # From each token in i...
                for l_offset in range(i_length):
                    # ...to each token in j
                    for m_offset in range(j_length):
                        sum_i_j += token_weights[tkn_l + l_offset][tkn_m + m_offset].item()
                tkn_m += j_length
            
                word_weights[wd_i][wd_j] = sum_i_j

            sum_i = sum(word_weights[wd_i])
            # Softmax except without the exponent I guess
            weight_i_j = []
            for j in word_weights[wd_i]:
                if not sum_i == 0:
                    weight_i_j.append(j / sum_i)
                else:
                    weight_i_j.append(0)
            word_weights[wd_i] = weight_i_j

            tkn_l += i_length

        return Tensor(word_weights).to(self.device)
                
    def calculate_tree_loss(self, nwp_attn: Tensor, ids: Tensor, batch_size: int):
        tp_loss = 0
        samples = acc_tqdm(range(len(ids)), desc='Samples', leave=False)
        for p in samples:
            id = ids[p].item()
            nwp_sample = nwp_attn[p]
            supervision_sample = self.supervision[id]
            kl_sum = 0

            for i in range(len(nwp_sample)):
                kl_sum += sum([nan_to_num(supervision_sample[i][j] * log(supervision_sample[i][j] / nwp_sample[i][j]), nan=0, posinf=0.1, neginf=0.1) for j in range(len(supervision_sample)) if not nwp_sample[i][j].item() == 0 and not supervision_sample[i][j].item() == 0])                
            
            divisor = count_nonzero(nwp_sample[0]).item()

            if divisor == 0:
                divisor = 1

            kl_sum /= divisor

            tp_loss += kl_sum
            
            samples.set_postfix({"Tree Loss": f"{(tp_loss/(p + 1)):.3f}"})

        return tp_loss / batch_size