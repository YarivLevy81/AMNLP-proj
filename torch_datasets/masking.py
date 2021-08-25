import time
from collections import namedtuple, defaultdict
import numpy as np

STOPWORDS = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'also', 'could', 'would'}

MaskedSpanInstance = namedtuple("MaskedSpanInstance",
                              ["index", "begin_label", "end_label"])


def _iterate_span_indices(span):
    return range(span[0], span[1] + 1)


def get_candidate_span_clusters(tokens, max_span_length, include_sub_clusters=False, validate=True):
    token_to_indices = defaultdict(list)
    for i, token in enumerate(tokens):
        token_to_indices[token].append(i)

    recurring_spans = []
    for token, indices in token_to_indices.items():
        for i, idx1 in enumerate(indices):
            for j in range(i + 1, len(indices)):
                idx2 = indices[j]
                assert idx1 < idx2

                max_recurring_length = 1
                for length in range(1, max_span_length):
                    if include_sub_clusters:
                        recurring_spans.append((idx1, idx2, length))
                    if (idx2 + length) >= len(tokens) or tokens[idx1 + length] != tokens[idx2 + length]:
                        break
                    max_recurring_length += 1

                if max_recurring_length == max_span_length or not include_sub_clusters:
                    recurring_spans.append((idx1, idx2, max_recurring_length))

    spans_to_clusters = {}
    spans_to_representatives = {}
    for idx1, idx2, length in recurring_spans:
        first_span, second_span = (idx1, idx1 + length - 1), (idx2, idx2 + length - 1)
        if first_span in spans_to_representatives:
            if second_span not in spans_to_representatives:
                rep = spans_to_representatives[first_span]
                cluster = spans_to_clusters[rep]
                cluster.append(second_span)
                spans_to_representatives[second_span] = rep
        else:
            cluster = [first_span, second_span]
            spans_to_representatives[first_span] = first_span
            spans_to_representatives[second_span] = first_span
            spans_to_clusters[first_span] = cluster

    if validate:
        recurring_spans = [cluster for cluster in spans_to_clusters.values()
                           if validate_ngram(tokens, cluster[0][0], cluster[0][1] - cluster[0][0] + 1)]
    else:
        recurring_spans = spans_to_clusters.values()
    return recurring_spans


def validate_ngram(tokens, start_index, length):
    # If the vocab at the beginning of the span is a part-of-word (##), we don't want to consider this span.
    # if vocab_word_piece[token_ids[start_index]]:
    if tokens[start_index].startswith("##"):
        return False

    # If the token *after* this considered span is a part-of-word (##), we don't want to consider this span.
    if (start_index + length) < len(tokens) and tokens[start_index + length].startswith("##"):
        return False

    if any([(not tokens[idx].isalnum()) and (not tokens[idx].startswith("##")) for idx in range(start_index, start_index+length)]):
        return False

    # We filter out n-grams that are all stopwords (e.g. "in the", "with my", ...)
    if any([tokens[idx].lower() not in STOPWORDS for idx in range(start_index, start_index+length)]):
        return True
    return False


def get_span_clusters_by_length(span_clusters, seq_length):
    already_taken = [False] * seq_length
    span_clusters = sorted([(cluster, cluster[0][1] - cluster[0][0] + 1) for cluster in span_clusters],
                           key=lambda x: x[1], reverse=True)
    filtered_span_clusters = []
    for span_cluster, _ in span_clusters:
        unpruned_spans = []
        for span in span_cluster:
            if any((already_taken[i] for i in range(span[0], span[1]+1))):
                continue
            unpruned_spans.append(span)

        # Validating that the cluster is indeed "recurring" after the pruning
        if len(unpruned_spans) >= 2:
            filtered_span_clusters.append(unpruned_spans)
            for span in unpruned_spans:
                for idx in _iterate_span_indices(span):
                    already_taken[idx] = True

    return filtered_span_clusters


def create_recurring_span_selection_predictions(tokens, max_recurring_predictions, max_span_length, masked_lm_prob):
    masked_spans = []
    num_predictions = 0
    new_tokens = list(tokens)

    already_masked_tokens = [False] * len(new_tokens)
    span_label_tokens = [False] * len(new_tokens)

    num_to_predict = min(max_recurring_predictions,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # start_time = time.time()
    span_clusters = get_candidate_span_clusters(tokens, max_span_length, include_sub_clusters=True)
    span_clusters = get_span_clusters_by_length(span_clusters, len(tokens))
    span_clusters = [(cluster, tuple(tokens[cluster[0][0]:cluster[0][1]+1])) for cluster in span_clusters]
    # end_time = time.time()

    span_cluster_indices = np.random.permutation(range(len(span_clusters)))
    span_counter = 0
    while span_counter < len(span_cluster_indices):
        span_idx = span_cluster_indices[span_counter]
        span_cluster = span_clusters[span_idx][0]
        # self._assert_and_return_identical(token_ids, identical_spans)
        num_occurrences = len(span_cluster)

        unmasked_span_idx = np.random.randint(num_occurrences)
        unmasked_span = span_cluster[unmasked_span_idx]
        span_counter += 1
        if any([already_masked_tokens[i] for i in _iterate_span_indices(unmasked_span)]):
            # The same token can't be both masked for one pair and unmasked for another pair
            continue

        unmasked_span_beginning, unmasked_span_ending = unmasked_span
        for i, span in enumerate(span_cluster):
            if num_predictions >= num_to_predict:
                # logger.warning(f"Already masked {self.max_predictions} spans.")
                break

            if any([already_masked_tokens[j] for j in _iterate_span_indices(unmasked_span)]):
                break

            if i != unmasked_span_idx:
                if any([already_masked_tokens[j] or span_label_tokens[j] for j in _iterate_span_indices(span)]):
                    # The same token can't be both masked for one pair and unmasked for another pair,
                    # or alternatively masked twice
                    continue

                if any([new_tokens[j] != new_tokens[k] for j, k in
                                       zip(_iterate_span_indices(span), _iterate_span_indices(unmasked_span))]):
                    print(
                        f"Two non-identical spans: unmasked {new_tokens[unmasked_span_beginning:unmasked_span_ending + 1]}, "
                        f"masked:{new_tokens[span[0]:span[1] + 1]}")
                    continue

                is_first_token = True
                for j in _iterate_span_indices(span):
                    if is_first_token:
                        new_tokens[j] = "<extra_id>"
                        masked_spans.append(MaskedSpanInstance(index=j,
                                                               begin_label=unmasked_span_beginning,
                                                               end_label=unmasked_span_ending))
                        num_predictions += 1
                    else:
                        new_tokens[j] = "<pad>"

                    is_first_token = False
                    already_masked_tokens[j] = True

                for j in _iterate_span_indices(unmasked_span):
                    span_label_tokens[j] = True

    assert len(masked_spans) <= num_to_predict
    masked_spans = sorted(masked_spans, key=lambda x: x.index)

    j = 0
    for i, token in enumerate(new_tokens):
        if token == '<extra_id>':
            new_tokens[i] = f'<extra_id_{j}>'
            j += 1

    span_label_tokens = []
    if len(masked_spans) == 0:
        print('skiping...')
        return None, None, None

    for j, p in enumerate(masked_spans):
        span_label_tokens = span_label_tokens + [f"<extra_id_{j}>"] + tokens[p.begin_label:p.end_label + 1]
    # </s> added in /a/home/cc/students/cs/sehaik/.local/lib/python3.7/site-packages/transformers/models/t5/tokenization_t5.py line 191
    #span_label_tokens = span_label_tokens + [f"<extra_id_{len(masked_spans)}>"] + ["</s>"]
    span_label_tokens = span_label_tokens + [f"<extra_id_{len(masked_spans)}>"]

    return new_tokens, span_label_tokens, span_clusters
