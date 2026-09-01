########################################################
# Version V3 
# date : 25/08/2026
# author : Stephane MEILLIEZ
########################################################
"""ADG generative linguistic steganography (Zhang et al., 2021).

Implements the Adaptive Dynamic Grouping algorithm.

Public API:
    ADG_encode          embed a bit payload into a stego sequence
    ADG_decode          recover the payload from a stego sequence
    ADG_generate_cover  sample a message-free sequence  
    ADG_replay          rescore a fixed token sequence under a given context

"""

# Imports

import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer 
from dataclasses import dataclass, field
from typing import Any

# ============================================================
#                          Dataclasses
# ============================================================


@dataclass
class ADGTreeNode:
    eta_tilde: float
    children: list["ADGTreeNode"] = field(default_factory=list)


@dataclass
class ADGConfig:
    model: Any
    tokenizer: Any
    device: str
    temperature: float = 1.0
    top_k: int = 50


@dataclass
class ADGGroup:
    recursion_level: int  # le niveau de recursion
    n_groups: int  # le nombre de groupes à cette étape (puissance de deux)
    group_idx: int  # le numéro du groupe entre 1 et n_groups
    eta_tilde: float  # le poids relatif


@dataclass
class TokenInfo:
    token_id: int  # ex : 125                                          # pas sur que ce sera utile...
    position: int  # ex : 23
    log_p_LM: float  # ordre de grandeur et directement utilisable
    bits_encoded: list[int]  # [0, 1, 0, 0] ou [] si vide
    tree_path: list[ADGGroup]  # l'historique du routage

    top_k_logits: Any | None = None  # pas terrible et à refactorer...
    top_k_ids: Any | None = None


@dataclass
class GeneratedText:
    text: str  # "there is a cat on the "
    tokens: list[int]  # [12, 123, 35 ...]
    token_infos: list[TokenInfo] = field(default_factory=list)

    @property
    def embedding_rate(self) -> float:
        """Calcule l'embedding_rate."""
        if not self.tokens or not self.token_infos:
            return 0.0
        total_bits = sum(
            len(individual_token_info.bits_encoded)
            for individual_token_info in self.token_infos
        )
        return total_bits / len(self.tokens)


@dataclass
class GroupSelectionResult:
    bits_encoded: list[int]
    tree_path: list[ADGGroup]
    candidate_ids: torch.Tensor  # tokens du groupe terminal (où échantillonner)
    probs: torch.Tensor  # leurs probas normalisées
    bit_index: int  # nouvelle position dans le bit_list
    p_LM: (
        torch.Tensor
    )  # distribution complète (pour pouvoir calculer log_p_LM)       <-----------topk ou pas
    past: Any  # past_key_values mis à jour
    raw_logits: torch.Tensor


# ===================================================================================
#                                 Utilitaires bits.
# ===================================================================================


def int2bits(inp: int, num_bits: int) -> list[int]:
    """Convert an integer to its binary representation as a list of bits (LSB first)."""
    if num_bits == 0:
        return []
    strlist = ("{0:0%db}" % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def bits2int(bits: list[int]) -> int:
    """Convert a list of bits (LSB first) to an integer."""
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2**i)
    return res


# ===================================================================================
#                                 ALgorithme ADG.
# ===================================================================================


def _find_nearest_prob_idx(sorted_probs: list[float], epsilon: float) -> int:
    """
    Returns the index of the token with probability closest to epsilon.
    Assumes sorted_probs is sorted in descending order.
    Uses binary search.
    """
    n = len(sorted_probs)

    if epsilon >= sorted_probs[0]:
        return 0
    if epsilon <= sorted_probs[-1]:
        return n - 1

    low_idx = 0
    high_idx = n - 1
    while high_idx - low_idx > 1:
        mid_idx = (low_idx + high_idx) // 2
        if sorted_probs[mid_idx] == epsilon:
            return mid_idx
        if sorted_probs[mid_idx] > epsilon:
            low_idx = mid_idx
        else:
            high_idx = mid_idx
    if abs(sorted_probs[low_idx] - epsilon) <= abs(sorted_probs[high_idx] - epsilon):
        return low_idx
    return high_idx


def sub_optimal_grouping(
    token_ids: torch.Tensor, probs: torch.Tensor, top_k: int
) -> list[list[int]]:
    """
    Implements Algorithm 1 from Zhang et al. (2021).

    Groups tokens into u = 2^floor(-log2(p_max)) groups of approximately equal probability.

    Args:
        token_ids: Token indices
        probs: Probability distribution (must sum to 1)
        top_k: Keep only top_k tokens before grouping

    Returns:
        List of groups, each group is a list of token IDs
    """
    assert len(token_ids) == len(probs)

    # Top-k filtering + sort
    if top_k is not None and top_k < len(probs):
        top_probs, top_indices = torch.topk(probs, top_k)
        token_ids = token_ids[top_indices]
        p_topk = top_probs / top_probs.sum()
        # p_topk = (top_probs / top_probs.sum()).float()
    else:
        p_topk = probs

    sorted_probs, sort_order = torch.sort(p_topk, descending=True)
    sorted_probs = sorted_probs.tolist()
    sorted_indices = token_ids[sort_order].tolist()

    # Compute number of groups : u
    p_max = sorted_probs[0]
    #print(f"p_max : {p_max} - {p_max:10.9f}")
     
    u = 2 ** math.floor(-math.log2(p_max))
    mean = 1.0 / u

    if len(sorted_probs) < u:
        raise ValueError(f"Not enough tokens ({len(sorted_probs)}) for {u} groups")

    G = [[] for _ in range(u)]
    remaining_sum = sum(sorted_probs)  # should be 1, but I kept exact value

    # Build first u-1 groups with equal probability
    for i in range(u - 1):
        if not sorted_probs:
            # Plus de tokens disponibles : les groupes restants seront vides.
            break

        current_first_index = sorted_indices.pop(0)
        current_max_prob = sorted_probs.pop(0)

        G[i].append(current_first_index)
        group_sum = current_max_prob
        remaining_sum -= current_max_prob

        while group_sum < mean and sorted_probs:
            epsilon = mean - group_sum
            idx_token = _find_nearest_prob_idx(sorted_probs, epsilon)
            prob_token = sorted_probs[idx_token]

            if prob_token - epsilon < epsilon:
                G[i].append(sorted_indices.pop(idx_token))
                group_sum += sorted_probs.pop(idx_token)
                remaining_sum -= prob_token
            else:
                break
        mean = remaining_sum / (u - (i + 1))

    # Last group gets remaining tokens
    G[-1].extend(sorted_indices)
    return G


def build_full_tree(p_LM: torch.Tensor, top_k: int) -> list[ADGTreeNode]:
    """
    Unfolds the complete ADG recursion tree.
    Returns a list of ADGTreeNode with eta_tilde RELATIVE to parent.
    At level 0, eta_tilde = mass(group) / total_mass.
    """
    token_ids = torch.arange(len(p_LM))  # c'est un tenseur
    groups_level_0 = sub_optimal_grouping(
        token_ids, p_LM, top_k
    )  # c'est une liste de listes d'index, pas un tenseur

    # total mass c'est la mass des topk... 
    total_mass = sum(p_LM[t].item() for g in groups_level_0 for t in g)

    def build_node(current_group, parent_mass):
        current_group_ids = torch.tensor(
            current_group
        )  # c'est un tenseur/ conversions de tensor a list à améliorer
        current_mass = p_LM[current_group_ids].sum().item()  # c'est un tenseur

        current_eta = current_mass / parent_mass if parent_mass > 0 else 0.0

        group_probs = p_LM[current_group_ids]
        group_probs = group_probs / group_probs.sum()  # group_probs est renomralisé

        sub_groups = sub_optimal_grouping(current_group_ids, group_probs, top_k=None)

        if len(sub_groups) == 1:
            return ADGTreeNode(
                eta_tilde=current_eta
            )  #  field(default_factory=list) donc [] est ajouté auto
        else:
            children = [build_node(g, current_mass) for g in sub_groups]
            return ADGTreeNode(eta_tilde=current_eta, children=children)

    return [build_node(g, total_mass) for g in groups_level_0]


def compute_D_KL(tree: list[ADGTreeNode]) -> float:
    """
    D_KL(p_LM || q_ADG) via leaf decomposition.
    D_KL = sum_f eta_f * log(eta_f / U_f)
    """
    total_DKL = 0

    def go_to_leaf(node, eta_path, u_path):
        current_eta = node.eta_tilde * eta_path
        if len(node.children) == 0:
            if current_eta > 0 and u_path > 0:
                return current_eta * math.log(current_eta / u_path)
            return 0.0
        else:
            n = len(node.children)
            return sum(
                go_to_leaf(child, current_eta, u_path / n) for child in node.children
            )

    n0 = len(tree)
    for node in tree:
        total_DKL += go_to_leaf(node, 1.0, 1.0 / n0)
    return total_DKL


# ===================================================================================
#                                 Interface LLM
# ===================================================================================


def load_model(model_name: str, device: torch.device):
    """Load a causal LM and its tokenizer from HuggingFace.

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, pad_token_id=tokenizer.eos_token_id
    )
    model.to(device)
    model.eval()
    print(f"Loaded {model_name} on {device} (vocab size: {tokenizer.vocab_size})")
    return model, tokenizer


def get_next_token_probs(
    context_ids: list[int],
    config: ADGConfig,
    past: Any = None,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    """
    Computes p_LM over vocabulary for next token.
    If past is None: forward sur tout context_ids, retourne (p_LM, new_past).
    Sinon: forward sur context_ids[-1] uniquement, en utilisant past, retourne (p_LM, new_past).
    """

    assert config.temperature > 0, "Temperature must be strictly positive"

    if past is None:

        tensor_input = torch.tensor([context_ids]).to(config.device)
        with torch.no_grad():
            out = config.model(tensor_input, use_cache=True)
        logits = out.logits
        new_past = out.past_key_values

    else:

        tensor_input = torch.tensor([[context_ids[-1]]]).to(config.device)
        with torch.no_grad():
            out = config.model(tensor_input, past_key_values=past, use_cache=True)
        logits = out.logits
        new_past = out.past_key_values

    # p_LM = F.softmax(logits[0, -1, :] / config.temperature, dim=-1)
    raw_logits = logits[0, -1, :]
    p_LM = F.softmax(raw_logits / config.temperature, dim=-1)
    return p_LM, new_past, raw_logits


def encode_prompt(prompt: str, config: ADGConfig) -> list[int]:
    """Unique point de conversion texte -> ids. Convention : pas de tokens spéciaux."""

    return config.tokenizer.encode(prompt, add_special_tokens=False)


# ===================================================================================
# --                                  Pipeline ADG
# ===================================================================================


def recursive_group_selection(
    bit_list: list[int],
    bit_index: int,
    context_ids: list[int],
    config: ADGConfig,
    past=None,
) -> GroupSelectionResult:
    """
    Recursively selects token groups based on bits to encode.
    Return selected token IDs, their probabilities, and updated bit_index.   
    """
    # Initial grouping
    p_LM, new_past, raw_logits = get_next_token_probs(context_ids, config, past=past)

    candidate_ids = torch.arange(len(p_LM)).to(
        config.device
    )  # à la premiere passe tous les tokens sont condidats

    bits_encoded = []
    current_tree_path = []

    G = sub_optimal_grouping(candidate_ids, p_LM, config.top_k)
    nb_bits = int(math.log2(len(G)))

    # on calcule la masse des top_k tokens dans G (rappel G est une liste de sous groupes)
    all_ids = [idx for group in G for idx in group]
    mass_tot = p_LM[all_ids].sum().item()

    candidate_ids = torch.tensor(all_ids).to(config.device)
    probs = p_LM[candidate_ids]
    probs = probs / probs.sum()

    # Recursive selection until group is indivisible
    while nb_bits > 0:  # tant qu'il arrive à faire des sous groupes...
        bits_needed = bit_list[bit_index : bit_index + nb_bits]

        # On met à jour avant le padding éventuel  
        bits_encoded.extend(bits_needed)

        # On gère le padding en fin de message
        if len(bits_needed) < nb_bits:
            bits_needed += [0] * (nb_bits - len(bits_needed))

        # le numéro de groupe correspond aux bits à encoder
        group_idx = bits2int(bits_needed)
        active_subgroup = G[group_idx]

        # on calcule eta_tilde la masse relative du sous groupe selectionné.
        mass_subgroup = p_LM[active_subgroup].sum().item()
        eta_tilde = mass_subgroup / mass_tot

        current_tree_path.append(
            ADGGroup(
                recursion_level=len(current_tree_path),
                n_groups=len(G),
                group_idx=group_idx,
                eta_tilde=eta_tilde,
            )
        )

        mass_tot = mass_subgroup
        candidate_ids = torch.tensor(active_subgroup).to(config.device)
        bit_index += nb_bits

        # on recalcule les probabilités locales normalisées.
        probs = p_LM[candidate_ids]
        probs = probs / probs.sum()

        G = sub_optimal_grouping(
            candidate_ids, probs, top_k=None
        )  # top_k déjà appliqué
        nb_bits = int(math.log2(len(G)))

    return GroupSelectionResult(
        bits_encoded=bits_encoded,
        tree_path=current_tree_path,
        candidate_ids=candidate_ids,
        probs=probs,
        bit_index=bit_index,
        p_LM=p_LM,
        past=new_past,
        raw_logits=raw_logits,
    )


def ADG_encode(
    prompt_tokens: list[int],
    bit_list: list[int],
    config: ADGConfig,
    max_tokens: int | None = None,
) -> GeneratedText | None:
    """Embed a bit payload into a STEGO sequence using the ADG algorithm.

    For each token, descends the ADG group tree according to the next bits, then
    samples a token inside the terminal group. This is the alternative
    hypothesis H1 — the token choice carries the message.

    Args:
        prompt_tokens: Prompt ids (not text). From a string:
            ADG_encode(encode_prompt(prompt, config), bit_list, config).
        bit_list: Payload bits to embed.
        config: Model, tokenizer, top_k and temperature.
        max_tokens: Hard cap on generated tokens; None means no cap. Returns
            None if the cap is hit before the whole payload is embedded.

    Returns:
        A GeneratedText (stego tokens + per-token TokenInfo), or None if the
        `max_tokens` cap was hit first.
    """
    bit_index = 0
    context_ids = list(prompt_tokens)  # copy: never mutate the caller's list
    stego_tokens = []
    token_infos = []
    past = None

    while bit_index < len(bit_list):
        result = recursive_group_selection(
            bit_list, bit_index, context_ids, config, past=past
        )

        selected_id = result.candidate_ids[torch.multinomial(result.probs, 1)].item()
        log_p_LM = torch.log(result.p_LM[selected_id]).item()
        top_k_logits, top_k_ids = torch.topk(result.raw_logits, config.top_k)

        token_infos.append(
            TokenInfo(
                token_id=selected_id,
                position=len(stego_tokens),
                log_p_LM=log_p_LM,
                bits_encoded=result.bits_encoded,
                tree_path=result.tree_path,
                top_k_logits=top_k_logits.cpu().float().numpy(),
                top_k_ids=top_k_ids.cpu().numpy(),
            )
        )

        stego_tokens.append(selected_id)
        context_ids.append(selected_id)
        bit_index = result.bit_index
        past = result.past

        if max_tokens is not None and len(stego_tokens) >= max_tokens:
            return None

    stego_text = config.tokenizer.decode(stego_tokens)
    return GeneratedText(text=stego_text, tokens=stego_tokens, token_infos=token_infos)


def _find_group(token_id: int, G: list[list[int]]) -> int | None:
    """Return index of group containing token_id, or None."""
    for i, group in enumerate(G):
        if token_id in group:
            return i
    return None


def extract_bits_from_token(
    token_id: int,
    context_ids: list[int],
    config: ADGConfig,
    p_LM=None,
    past=None,
) -> tuple[list[int], Any, list[ADGGroup]]:
    """
    Inverse of recursive_group_selection.
    Extracts the bits encoded by a single token.
    """
    if p_LM is None:
        p_LM, past, _ = get_next_token_probs(context_ids, config, past=past)

    tree_path = []

    token_ids = torch.arange(len(p_LM)).to(config.device)
    G = sub_optimal_grouping(token_ids, p_LM, config.top_k)
    nb_bits = int(math.log2(len(G)))
    bits = []

    # on calcule la masse des top_k tokens dans G (rappel G est une liste de sous groupes)
    all_ids = [idx for group in G for idx in group]
    mass_tot = p_LM[all_ids].sum().item()

    while nb_bits > 0:

        # Find which group contains token_id
        group_idx = _find_group(token_id, G)
        assert group_idx is not None, (
            f"Token {token_id} not found in any group. "
            f"This usually indicates a cache/no-cache inconsistency "
            f"between encode and decode."
        )

        # le numéro de groupe est ajouté à la liste de bits
        bits.extend(int2bits(group_idx, nb_bits))

        # on identifie la liste des token dans le sous groupe
        token_ids = torch.tensor(G[group_idx]).to(config.device)

        # probs est sans doute pas terrible comme nom. C'est la masse du sous groupe
        probs = p_LM[token_ids]

        mass_subgroup = p_LM[token_ids].sum().item()
        eta_tilde = mass_subgroup / mass_tot
        tree_path.append(
            ADGGroup(
                recursion_level=len(tree_path),
                n_groups=len(G),
                group_idx=group_idx,
                eta_tilde=eta_tilde,
            )
        )
        probs = probs / probs.sum()  # probs est renormalisé donc interne au sous groupe

        G = sub_optimal_grouping(token_ids, probs, top_k=None)
        mass_tot = mass_subgroup

        nb_bits = int(math.log2(len(G)))

    return bits, past, tree_path

#  
def ADG_replay(
    observed_tokens: list[int],
    context_head_ids: list[int],
    config: ADGConfig,
) -> list[dict]:
    """Replay the ADG routing of a FIXED token sequence under a given context.

    Walks the ADG routing token by token, like ADG_decode, but the observed
    tokens are imposed (not chosen) and the context is given as ids. This is the
    single entry point for the warden's analysis:
      - S3: pass the true prompt ids -> reproduces the stored routing;
      - S2: pass an approximate prompt (synonym swap) or [BOS] ids.
    Only `context_head_ids` changes between S3 and S2.

    Unlike ADG_decode, it returns the full `tree_path` per token (the LRT input)
    and does not crash when an observed token falls outside p_LM's top-k under
    this context: it flags `in_support=False` and yields an empty routing.

    Args:
        observed_tokens: The intercepted sequence to score (kept fixed).
        context_head_ids: Eve's initial context, as ids (true prompt / approx / [BOS]).
        config: Model, tokenizer, top_k and temperature.

    Returns:
        One dict per token, with the generation token-parquet columns
        (`position`, `token_id`, `bits_encoded`, `tree_path`) plus diagnostics
        `p_max`, `rank_obs`, `in_support`.
    """
    context_ids = [int(t) for t in context_head_ids]  
    past = None
    rows = []

    for pos, tok in enumerate(observed_tokens):
        tok = int(tok)

     
        p_LM, past, _ = get_next_token_probs(context_ids, config, past=past)

        p_max = p_LM.max().item()
        rank_obs = int((p_LM > p_LM[tok]).sum().item())  # 0 = argmax

        # --- le token observé est-il dans le top-k de CE p_LM ? ---
        if config.top_k is None or config.top_k >= p_LM.shape[0]:
            in_support = True  # pas de filtrage
        else:
            topk_ids = torch.topk(p_LM, config.top_k).indices
            in_support = bool((topk_ids == tok).any().item())

        if in_support:
            # p_LM fourni => extract_bits_from_token NE refait PAS de forward
            # (et renvoie le `past` inchangé, qu'on ignore).
            bits, _, tree_path = extract_bits_from_token(
                tok, context_ids, config, p_LM=p_LM, past=past
            )
            path_ser = [
                (g.recursion_level, g.n_groups, g.group_idx, g.eta_tilde)
                for g in tree_path
            ]  # sérialisation = génération
        else:
            bits, path_ser = [], []  # hors-support : routage impossible

        rows.append(
            {
                "position": pos,
                "token_id": tok,
                "bits_encoded": bits,
                "tree_path": path_ser,
                "p_max": p_max,  
                "rank_obs": rank_obs,  
                "in_support": in_support,  
            }
        )

        context_ids.append(tok) 

    return rows


def ADG_decode(
    prompt_tokens: list[int], stego_tokens: list[int], nb_bits: int, config: ADGConfig
) -> list[int]:
    """Decode the payload bits hidden in a stegotext using the ADG algorithm.

    Replays the same per-token routing as the sender, but reads which group each
    observed token falls into instead of choosing it.

    Args:
        prompt_tokens: Prompt ids — the SAME context the sender used (not text).
            From a string: ADG_decode(encode_prompt(prompt, config), ...).
        stego_tokens: Ids of the received stegotext to decode.
        nb_bits: Length of the original payload; the output is truncated to it
            (the last token may carry padding bits beyond the message).
        config: Model, tokenizer, top_k and temperature.

    Returns:
        The decoded bit list, truncated to `nb_bits`.
    """
    context_ids = list(prompt_tokens)       # copy: never mutate the caller's list
    bits = []
    past = None
    for token_id in stego_tokens:
        new_bits, past, _ = extract_bits_from_token(
            token_id, context_ids, config, past=past
        )
        bits.extend(new_bits)
        context_ids.append(token_id)
    return bits[:nb_bits]


def ADG_generate_cover(
    prompt_tokens: list[int],
    nb_bits: int,
    max_tokens: int,
    config: ADGConfig,
) -> GeneratedText | None:
    """Generate a COVER sequence: plain top-k sampling, no message embedded.

    Each token is drawn from the standard top-k distribution; the ADG bits are
    only *read back* afterwards (which group the sampled token fell into). This
    is the null hypothesis H0 — statistically a genuine LM sample, not stego.

    Args:
        prompt_tokens: Prompt ids (not text). From a string:
            ADG_generate_cover(encode_prompt(prompt, config), ...).
        nb_bits: Sampling continues until at least this many bits are read back.
        max_tokens: Hard cap; returns None if reached before `nb_bits` (discard).
        config: Model, tokenizer, top_k and temperature.

    Returns:
        A GeneratedText (cover tokens + per-token TokenInfo), or None if the
        `max_tokens` cap was hit first.
    """
    context_ids = list(prompt_tokens)  # copy: never mutate the caller's list
    bit_stream = []
    cover_tokens = []
    token_infos = []
    past = None

    while len(bit_stream) < nb_bits:
        p_LM, past, raw_logits = get_next_token_probs(context_ids, config, past=past)

        top_probs, top_idx = torch.topk(p_LM, config.top_k)
        top_probs = top_probs / top_probs.sum()
        token_id = top_idx[torch.multinomial(top_probs, 1)].item()

        new_bits, _, tree_path = extract_bits_from_token(
            token_id, context_ids, config, p_LM=p_LM
        )
        bit_stream.extend(new_bits)

        top_k_logits, top_k_ids = torch.topk(raw_logits, config.top_k)

        token_infos.append(
            TokenInfo(
                token_id=token_id,
                position=len(cover_tokens),
                log_p_LM=torch.log(p_LM[token_id]).item(),
                bits_encoded=new_bits,
                tree_path=tree_path,
                top_k_logits=top_k_logits.cpu().float().numpy(),
                top_k_ids=top_k_ids.cpu().numpy(),
            )
        )

        cover_tokens.append(token_id)
        context_ids.append(token_id)

        if len(cover_tokens) >= max_tokens:
            return None

    cover_text = config.tokenizer.decode(cover_tokens)

    return GeneratedText(text=cover_text, tokens=cover_tokens, token_infos=token_infos)
