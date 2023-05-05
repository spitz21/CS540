import os
import math


# These first two functions require os operations and so are completed for you
# Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset


# Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        if d.startswith('.'):
            # ignore hidden files
            continue
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f, 'r', encoding='utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


# The rest of the functions need modifications ------------------------------
# Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = dict.fromkeys(vocab, 0)
    bow.update({})
    f = open(filepath, encoding='utf-8').readlines()

    for word in f:
        word = word.strip()
        if bow.__contains__(word):
            bow[word] += 1
        else:
            if bow.__contains__(None):
                bow[None] += 1
            else:
                bow.update({None: 1})

    return bow


# Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = dict.fromkeys(label_list, 0.0)
    numfiles = 2
    t2016 = 0.0
    t2020 = 0.0

    for dictionary in training_data:
        numfiles += 1
        if dictionary['label'] == '2016':
            t2016 += 1
        else:
            t2020 += 1

    t2016 = math.log((t2016 + smooth) / numfiles)
    t2020 = math.log((t2020 + smooth) / numfiles)
    logprob['2016'] = t2016
    logprob['2020'] = t2020

    return logprob


# Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = dict.fromkeys(vocab, 0.0)
    word_prob.update({None: 0.0})
    wc = 0.0
    vsize = 0.0

    for string in vocab:
        vsize += 1

    for dictionary in training_data:
        if dictionary['label'] == label:
            for key in dictionary['bow']:
                word_prob[key] += dictionary['bow'][key]
                wc += dictionary['bow'][key]

    for key in word_prob:
        word_prob[key] = math.log((word_prob[key] + smooth) / (wc + vsize + 1))

    return word_prob


##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = [f for f in os.listdir(training_directory) if not f.startswith('.')] # ignore hidden files
    # TODO: add your code here

    vocab = create_vocabulary(training_directory, cutoff)
    retval.update({'vocabulary': vocab})

    tdata = load_training_data(vocab, training_directory)
    lprior = prior(tdata, ['2020', '2016'])
    retval.update({'log prior': lprior})

    pw2020 = p_word_given_label(vocab, tdata, '2020')
    retval.update({'log p(w|y=2020)': pw2020})

    pw2016 = p_word_given_label(vocab, tdata, '2016')
    retval.update({'log p(w|y=2016)': pw2016})

    return retval


# Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    sum2016 = 0.0
    sum2020 = 0.0

    for word in open(filepath, encoding='utf-8').readlines():
        word = word.strip()
        if model['log p(w|y=2016)'].__contains__(word):
            sum2016 += model['log p(w|y=2016)'][word]
        else:
            sum2016 += model['log p(w|y=2016)'][None]
        if model['log p(w|y=2020)'].__contains__(word):
            sum2020 += model['log p(w|y=2020)'][word]
        else:
            sum2020 += model['log p(w|y=2020)'][None]

    sum2016 += model['log prior']["2016"]
    sum2020 += model['log prior']["2020"]
    retval.update({'log p(y=2020|x)': sum2020})
    retval.update({'log p(y=2016|x)': sum2016})

    if sum2016 < sum2020:
        retval.update({'predicted y': '2020'})
    else:
        retval.update({'predicted y': '2016'})

    return retval
