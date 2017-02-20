__author__ = 'GCassani'

import os
import argparse
import numpy as np
import random as rnd
from collections import defaultdict, Counter


def read_input_trials(input_file, header=True, sep1=' ', sep2='_'):

    """
    :param input_file:  a .txt file containing two columns. The first column is assumed to contain cues, while the
                        second columns is assumed to contain outcomes.
    :param header:      a boolean specifying whether the file contains a header
    :param sep1:        a string specifying which character divide the two columns of which the input file must consist
    :param sep2:        a string specifying which character is used to separate different cues and outcomes whithin the
                        same line. Default value is assumed to be the underscore
    :return trials:     a list of tuples, consisting of a list and a set. The list contains cues, and allows duplicates;
                        the set contains outcomes, thereby removing all duplicates that may exist in the second column
                        of each line of the input file.
    """

    trials = []

    with open(input_file, 'r+') as training:
        if header:
            next(training)
        for line in training:
            trial_cues = line.strip().split(sep1)[0].split(sep2)
            trial_outcomes = set(line.strip().split(sep1)[1].split(sep2))
            trials.append((trial_cues, trial_outcomes))

    return trials


########################################################################################################################


def print_cue_outcome_associations(avg_associations, n):

    """
    :param avg_associations:    a dictionary of dictionaries, where the bottom-level values are tuples consisting of two
                                numbers
    :param n:                   the number of decimal positions to which each number is rounded
    """

    for k1 in sorted(avg_associations.keys()):
        for k2 in sorted(avg_associations[k1].keys()):
            # print each cue-outcome pair with the corresponding association and its standard deviation
            print '\t'.join([k1, k2, str(round(avg_associations[k1][k2][0], n)),
                             str(round(avg_associations[k1][k2][1], n))])


########################################################################################################################


def get_average_associations(associations, flip=False, print_output=False):

    """
    :param associations:        a dictionary of dictionaries, where the top level keys are cues, the second level keys
                                are outcomes, and the bottom level values are lists of cue-outcome associations obtained
                                over the desired number of iterations
    :param flip:                a binary specifying whether cue-outcome associations should be printed and returned as
                                such or flipped i.e. outcome-cue
    :param print_output:        a boolean specifying whether average associations should be printed after having been
                                computed
    :return avg_associations:   a dictionary of dictionaries where top level keys can be either cues or outcomes
                                (depending on the value of flip), second level keys are outcomes or cues (depending of
                                what are top level keys), and values consist of a tuple containing the average
                                association for each cue-outcome pair and its standard deviation, both computed using
                                the associations obtained on each iteration.

    This function does two things: first, it takes the cue-outcome associations computed for the desired number of
    iterations and computes average and standard deviation of each of them; second, it prints the cue-outcome
    average associations together with the standard deviation.
    """

    avg_associations = defaultdict(dict)

    for k1 in sorted(associations.keys()):
        for k2 in sorted(associations[k1].keys()):
            if flip:
                avg_associations[k2][k1] = (np.mean(associations[k1][k2]), np.std(associations[k1][k2]))
            else:
                avg_associations[k1][k2] = (np.mean(associations[k1][k2]), np.std(associations[k1][k2]))

    if print_output:
        print_cue_outcome_associations(avg_associations, 3)

    return avg_associations


########################################################################################################################


def store_current_associations(curr_associations, associations):

    """
    :param curr_associations:   a dictionary of dictionaries where top level keys are cues, second level keys are
                                outcomes, and bottom level values are cue-outcome associations learned during a single
                                pass through the training data
    :param associations:        a dictionary of dictionaries of the same form of curr_associations, but where bottom
                                level values are lists, containing cue-outcome associations up to the current iteration.
    :return:                    nothing

    This function keeps track of all cue-outcome associations computed on each iteration, storing every ith association
    in a list.
    """

    for cue in curr_associations:
        for outcome in curr_associations[cue]:
            try:
                associations[cue][outcome].append(curr_associations[cue][outcome])
            except KeyError:
                associations[cue][outcome] = [curr_associations[cue][outcome]]


########################################################################################################################


def form_random_hypothesis(outcome, hypotheses, curr_outcomes, curr_cues):

    """
    :param outcome:         a string specifying which outcome is being considered from the current learning trial
    :param hypotheses:      a dictionary containing cue-outcome hypothesis: keys of the dictionaries are outcomes, and
                            values are cues that are currently hypothesized to match the outcomes
    :param curr_outcomes:   a list containing all the outcomes that are present in the current learning trial
    :param curr_cues:       a list containing all the cues that are present in the current learning trial
    :return hypotheses:     the input dictionary, that now contains a new hypothesis involving one of the outcomes that
                            were present in the current learning trial: the outcome has been selected at random among
                            curr_outcomes, while the cue that is now hypothesized to match the outcome has been selected
                            at random from curr_cues
    """

    del hypotheses[outcome]                                 # remove the hypothesis from memory,
    new_o = rnd.randint(1, len(curr_outcomes)) - 1          # select a new outcome at random
    new_outcome = curr_outcomes[new_o]
    new_sel = rnd.randint(1, len(curr_cues)) - 1            # select a new cue at random
    hypotheses[new_outcome] = curr_cues[new_sel]            # form a new cue-outcome hypotheses

    return hypotheses


########################################################################################################################


def compute_total_activation(matrix, outcome, trial_cues, lam):

    """
    :param matrix:      a dictionary of dictionaries, where first-level keys are outcomes, second-level keys are cues
                        and values are numbers indicating the degree of association between each cue and outcome pair
    :param outcome:     a string indicating the outcome being considered
    :param trial_cues:  a dictionary containing the cues that are present in the current learning trial as keys, and
                        the number of times each cue occurs as values
    :return v_total:    the total activation that goes from the cues that are present in the learning trial to the
                        outcome being considered
    """

    v_total = 0
    for cue, count in trial_cues.items():
        try:
            # add to v_total the amount of activation of each cue, weighted by the number of times the cue occurs in the
            # current learning trial
            v_total += matrix[outcome][cue] * count
        except KeyError:
            matrix[outcome][cue] = 0

    # check that the total amount of activation doesn't exceed the value chosen for the lambda parameter: since the
    # lambda parameter specifies the maximum amount of activation each outcome can bear, a v_total that is higher than
    # lambda corresponds to an over-prediction of the presence of the outcome which causes the model to collapse.
    if v_total > lam:
        print trial_cues, outcome
        print v_total
        raise ValueError('Something went wrong! The total amount of activation cannot exceed one: ' +
                         'try lowering the learning rate, alpha, or the cue salience parameter, beta')
    else:
        return v_total

########################################################################################################################


def update_outcome_weights(matrix, outcome, trial_cues, trial_outcomes, lam, ab):

    """
    :param matrix:          a dictionary of dictionaries, where first-level keys are outcomes, second-level keys are
                            cues and values are numbers indicating the degree of association between each cue and
                            outcome pair
    :param outcome:         a string indicating the outcome being considered
    :param trial_cues:      a dictionary containing the cues that are present in the current learning trial as keys, and
                            the number of times each cue occurs as values
    :param trial_outcomes:  a set containing the outcomes that are present in the current learning trial
    :param lam:             a number indicating the maximum amount of activation an outcome can bear
    :param ab:              the learning rate
    :return matrix:         an updated version of the input matrix, where cue-outcome associations have been updated
                            using the Rescorla-Wagner model of learning
    """

    # get the total amount of activation for the outcome being considered given the cues that occur in the current
    # learning trial
    v_total = compute_total_activation(matrix, outcome, trial_cues, lam)

    # if the outcome doesn't occur in the current trial, set the value of lambda to 0
    if outcome not in trial_outcomes:
        lam = 0
    delta_v = ab * (lam - v_total)

    # for each cue from the current learning trial, update its association to the outcome being considered
    for cue in trial_cues:
        try:
            matrix[outcome][cue] += delta_v
        except KeyError:
            matrix[outcome][cue] = delta_v

        if matrix[outcome][cue] > 1:
            print outcome, cue, matrix[outcome][cue]
            raise ValueError('An outcome can only sustain an activation of 1 and it looks like the ' +
                             'activation from a single cue is higher than this threshold. Something ' +
                             'went wrong, try lowering the learning rate, alpha, or the cue salience ' +
                             'parameter, beta.')

    return matrix


########################################################################################################################


def discriminative_learner(training_trials, iterations, alpha=0.2, beta=0.1, lam=1,
                           seed=None, print_output=False):

    """
    :param training_trials:     a list of tuples, consisting of a list and a set. The list contains cues, the set
                                contains outcomes.
    :param iterations:          an integer specifying the number if simulations to run: when learning depends on the
                                order of the training trials, running multiple iterations ensures that the learned
                                associations are not due to chance.
    :param alpha:               cue salience. For simplicity, we assume that every cue has the same salience, so
                                changing the value of this parameter does not affect the relative strength of
                                of cue-outcome associations but only their absolute magnitude.
    :param beta:                learning rate. Again, we make the simplifying assumption that our simulated learners are
                                equally affected by positive and negative feedback. Changing the beta value can have a
                                significant impact on learning outcome, but 0.1 is a standard choice for this model.
    :param lam:                 maximum amount of association that an outcome can receive from all the cues. It simply
                                acts as a scaling factor, so changing its value has the same effects of changing alpha.
    :param seed:                allows to set the seed for the random shuffling of training trials, in case it is
                                important to reproduce exact results. The default is set to None to ensure maximal
                                randomness. In the CogSci 2016 paper we report results with seed=6.
    :param print_output:        a boolean specifying whether average associations should be printed after having been
                                computed
    :return avg_associations:   a dictionary of dictionaries containing average associations and their standard
                                deviations, computed over the desired number of iterations.

    This function implements Naive Discriminative Learning (NDL, see Baayen, Milin, Durdevic, Hendrix, Marelli (2011)
    for a detailed description of the model and its theoretical background. This learner uses the Rescorla-Wagner
    equations to update cue-outcome associations: it is a simple associative network with no hidden layer, that
    incrementally updates associations between cues and outcome.
    """

    associations = defaultdict(dict)

    ab = alpha * beta

    rnd.seed(seed)

    for i in xrange(iterations):

        # the order of presentation impacts learning, so training trials are randomly shuffled before each simulation
        rnd.shuffle(training_trials)

        ith_associations = defaultdict(dict)
        outcomes = set()

        for trial in training_trials:

            trial_cues = Counter(trial[0])
            trial_outcomes = trial[1]
            # keep track of all outcomes that have been encountered so far
            outcomes.update(trial_outcomes)

            #
            for outcome in outcomes:

                ith_associations = update_outcome_weights(ith_associations, outcome, trial_cues, trial_outcomes, lam, ab)

        # store association scores at the end of the current iteration
        store_current_associations(ith_associations, associations)

    # get the average cue-outcome associations over the specified number of iterations
    avg_associations = get_average_associations(associations, print_output=print_output)

    return avg_associations


########################################################################################################################


def hebbian_learner(training_trials, lam=1, print_output=False):

    """
    :param training_trials:     a list of tuples, consisting of a list and a set. The list contains cues, the set
                                contains outcomes.
    :param lam:                 change in association. The value does not affect the relative strength of cue-outcome
                                associations but only their magnitudes, so it operates as a simple linear scaling
                                factor.
    :param print_output:        a boolean specifying whether average associations should be printed after having been
                                computed
    :return avg_associations:   a dictionary of dictionaries containing average associations and their standard
                                deviations, computed over the desired number of iterations.

    This learner was assessed in Yu and Smith (2012) and is a simple associative network with no-hidden layer that
    updates cue-outcome associations according to the Hebb rule, meaning that each cue-outcome association is
    strengthened every time the cue and the outcome co-occur in a learning trial and left unchanged in all other cases.
    This learner is not sensitive to the order of presentation, thus there is no need of multiple iterations and
    random shuffling of the training trials.
    """

    associations = defaultdict(dict)

    for trial in training_trials:

        trial_cues = trial[0]
        trial_outcomes = trial[1]

        # increment cue-outcome association every time the two co-occurs in a training trial. If a cue and an outcome
        # co-occur for the first time, initialize their association score to the update constant lambda
        for cue in set(trial_cues):
            for outcome in set(trial_outcomes):
                try:
                    associations[cue][outcome] += lam
                except KeyError:
                    associations[cue][outcome] = lam

    avg_associations = get_average_associations(associations, flip=True, print_output=print_output)

    return avg_associations


########################################################################################################################


def hypothesis_testing_model(training_trials, iterations, alpha=0.6, alpha_1=(0.81, 0.90, 0.95, 0.99),
                             seed=None, print_output=False):

    """
    :param training_trials: a list of tuples, consisting of a list and a set. The list contains cues, the set contains
                            outcomes.
    :param iterations:      an integer specifying the number if simulations to run: when learning depends on the order
                            of the training trials, running multiple iterations ensures that the learned
                            associations are not due to chance.
    :param alpha:           probability of recalling a hypothesis for the first time after it was formed. The default
                            value is taken from Trueswell et al (2013) where this model was shown to fit behavioral data
                            in a word learning task.
    :param alpha_1:         a tuple containing probabilities of recalling a hypothesis after it was already successfully
                            recalled once. The first value gives the probability of recalling a hypothesis twice, the
                            second value gives the probability of recalling a hypothesis for the third time, and so on.
                            The first value also comes from Trueswell et al (2013) while the following ones were made
                            up: we stopped at five successful recall with p=0.99 to avoid certainty of recall and
                            assuming that after a certain number of correct recalls the probability doesn't change
                            anymore.
    :param seed:            allows to set the seed for the random shuffling of training trials, in case it is
                            important to reproduce exact results. The default is set to None to ensure maximal
                            randomness. In the CogSci 2016 paper we report results with seed=6.
                            CAVEAT: note that the final outcome also depends on the probability of recalling a
                            hypothesis at every learning trial, which is not seeded. Thus, even using the same seed
                            as we used in the paper, different results might be obtained for this learner.
    :param print_output:    a boolean specifying whether average associations should be printed after having been
                            computed
    :return avg_hypotheses: a dictionary of dictionaries containing the proportion of learners that selected each
                            cue-outcome hypothesis over all simulations.

    This model is detailed in Trueswell, Medina, Hafri, and Gletiman (2013) and is fundamentally different from the
    other models we evaluated. Instead of building a network of cue-outcome associations, updating them after each
    learning trial, the Hypothesis Testing Mode (HTM) only forms and evaluates a single cue-outcome hypothesis on every
    learning trial. When no previous knowledge exists, the learner randomly select a cue-outcome mapping. On a
    subsequent learning trial.
    CAVEAT: at line 274, the function only picks the first two cues. This setting is specific to the simulation reported
    in Cassani, Grimm, Gillis, and Daelemans (2016). Remove the line to consider all cues, if you use a different
    dataset than ours.

    """

    hypotheses = defaultdict(dict)
    avg_hypotheses = defaultdict(dict)

    rnd.seed(seed)

    for i in xrange(iterations):

        ith_hypotheses = {}                                                     # contains formed hypothesis
        ith_recalled = defaultdict(dict)                                        # contains recalled hypothesis

        # the order of presentation impacts learning, so training trials are randomly shuffled before each simulation
        rnd.shuffle(training_trials)

        for trial in training_trials:

            trial_cues = trial[0][0:2]
            trial_outcomes = list(trial[1])

            o = rnd.randint(1, len(trial_outcomes)) - 1                     # select an outcome at random
            outcome = trial_outcomes[o]

            if outcome not in ith_hypotheses:                               # IF it is new
                sel = rnd.randint(1, len(trial_cues)) - 1                     # select a cue at random
                ith_hypotheses[outcome] = trial_cues[sel]                     # form a cue-outcome hypothesis
            else:                                                           # IF it is in memory
                if outcome not in ith_recalled:                               # BUT encountered just once
                    p = alpha
                    recall = np.random.binomial(1, p)                           # recall with probability alpha
                else:                                                         # IF it was encountered more times
                    if len(alpha_1) >= ith_recalled[outcome]:
                        p = alpha_1[ith_recalled[outcome]]                      # recall with increased alpha
                    else:
                        p = alpha_1[-1]                                         # BUT never more than .99
                    recall = np.random.binomial(1, p)

                if recall:                                                  # IF the outcome is recalled
                    hypothesis = ith_hypotheses[outcome]                      # retrieve the associated cue
                    if hypothesis in trial_cues:                              # IF the retrieved cue is present in the
                        try:                                                                             # current trial
                            ith_recalled[outcome][hypothesis] += 1              # strengthen its recall alpha
                        except KeyError:
                            ith_recalled[outcome][hypothesis] = 1
                    else:                                                     # IF it isn't
                        form_random_hypothesis(outcome, ith_hypotheses,         # form a new  hypothesis at random
                                               trial_outcomes, trial_cues)
                else:                                                       # IF the outcome is not recalled
                    form_random_hypothesis(outcome, ith_hypotheses,           # form a new  hypothesis at random
                                           trial_outcomes, trial_cues)

        # increment the count of learners that selected a certain cue-outcome hypothesis at the end of learning
        for o in ith_hypotheses:
            try:
                hypotheses[o][ith_hypotheses[o]] += 1
            except KeyError:
                hypotheses[o][ith_hypotheses[o]] = 1

    # compute the proportion of learners that selected each hypothesis
    for o in hypotheses:
        for c in hypotheses[o]:
            avg_hypotheses[o][c] = (hypotheses[o][c] / float(iterations), 0)
            if print_output:
                print o, c, avg_hypotheses[o][c]

    return avg_hypotheses


########################################################################################################################


def probabilistic_learner(training_trials, iterations, t0_prob=10 ** -4, beta=10 ** 4, lam=10 ** -5,
                          seed=None, print_output=False):

    """
    :param training_trials:     a list of tuples, consisting of a list and a set. The list contains cues, the set
                                contains outcomes.
    :param iterations:          an integer specifying the number if simulations to run: when learning depends on the
                                order of the training trials, running multiple iterations ensures that the learned
                                associations are not due to chance.
    :param t0_prob:             posterior probability of outcome given cue at time 0, before cue and outcome co-occur
                                in the learning trials, best computed as 1/beta, as indicated in fazly et al (2010).
    :param beta:                upper bound on the expected number of outcomes.
    :param lam:                 smoothing factors. It's important that lam is smaller than 1/beta, as indicated by
                                Fazly et al (2010).
    :param seed:                allows to set the seed for the random shuffling of training trials, in case it is
                                important to reproduce exact results. The default is set to None to ensure maximal
                                randomness. In the CogSci 2016 paper we report results with seed=6.
    :param print_output:        a boolean specifying whether average associations should be printed after having been
                                computed
    :return avg_associations:   a dictionary of dictionaries containing average associations and their standard
                                deviations, computed over the desired number of iterations.

    This learner is described at length in Fazly, Alishai, and Stevenson (2010) and is inspired by IBM machine
    translation algorithm. It is a simple associative network, with no hidden layers, that computes a full probability
    distribution over all outcomes for each cues. Learning happens by allocating a higher probability mass to the
    outcome that is best supported by the data. Each cue has its own probability distribution that is shifted according
    to the evidence accumulated over subsequent trials: highly skewed probability distributions reflect higher
    confidence in which outcome matches a cue, while more flat distributions reflect higher uncertainty
    """

    outcomes_given_cues = defaultdict(dict)

    rnd.seed(seed)

    for i in xrange(iterations):

        outcomes = set()
        ith_outcomes_given_cues = defaultdict(dict)
        ith_associations = defaultdict(dict)

        # the order of presentation impacts learning, so training trials are randomly shuffled before each simulation
        rnd.shuffle(training_trials)

        for trial in training_trials:

            trial_cues = trial[0]
            trial_outcomes = trial[1]

            # keep track of all outcomes that have been encountered so far
            outcomes = outcomes.union(set(trial_outcomes))

            # assign t0 probability to all new outcome-cue pairings resulting from the current trial
            # CAVEAT: for implementation purposes, this learner stores cue-outcome associations as outcome-cue, since it
            # computes the posterior probability of each outcome given a cue
            for outcome in set(trial_outcomes):
                for cue in set(trial_cues):
                    if cue not in ith_outcomes_given_cues[outcome].keys():
                        ith_outcomes_given_cues[outcome][cue] = t0_prob

            # compute the updates of the associations between cue and outcome. Importantly, associations are only a
            # first step to compute mappings: in order to get to the final stage of learning these associations still
            # have to undergo a further transformation
            for cue in set(trial_cues):
                for outcome in set(trial_outcomes):
                    numerator = ith_outcomes_given_cues[outcome][cue]
                    denominator = 0
                    # the denominator is proportional to the probabilities of each outcome present in the current
                    # learning trial given each cue in the learning trial. If the learner is already confident that an
                    # outcome is the right one, the denominator will be higher (and higher for every such case in which
                    # the current learning trial contains cue-outcome pairs for which the learner is confident that it
                    # already knows the mapping
                    for c in ith_outcomes_given_cues[outcome]:
                        denominator += ith_outcomes_given_cues[outcome][c]
                    a = numerator / float(denominator)

                    if outcome not in ith_associations[cue].keys():
                        # if a cue-outcome pairs is encountered for the first time, its association value is simply the
                        # increment since the same pair, at the previous time point, had an association value of 0
                        ith_associations[cue][outcome] = a
                    else:
                        # if a cue-outcome pairs was already encountered, its association value is incremented by the
                        # computed association change.
                        ith_associations[cue][outcome] += a

            # compute new conditional probabilities of outcomes given cues, for all cues in the learning trial: this is
            # the true estimate of how the learner has learned cue-outcome pairs: the probability mass of each cue is
            # incrementally allocated to the outcome for which more evidence was found in the learning trials. The
            # probability distribution is only updated for cues that are present in the current learning trial, while,
            # of course, all outcomes that occurred with a cue are considered to update the probability distribution
            # over all outcomes for each cue (present in the current trial).
            for cue in set(trial_cues):
                for outcome in ith_associations[cue]:
                    numerator = ith_associations[cue][outcome] + lam
                    smooth = beta * lam
                    denominator = 0
                    # the higher the associations between a cue and all the outcomes it occurred with, the higher the
                    # denominator and the lower the posterior probability of a cue-outcome mapping
                    for o in outcomes:
                        if o in ith_associations[cue].keys():
                            denominator += ith_associations[cue][o]

                    ith_outcomes_given_cues[outcome][cue] = numerator / (denominator + smooth)

        # store posterior probability distributions after each complete iteration on the training trials
        store_current_associations(ith_outcomes_given_cues, outcomes_given_cues)

    # given that cue-outcome pairs were stored as outcome-cue pairs, the argument flip is set to False.
    avg_posteriors = get_average_associations(outcomes_given_cues, print_output=print_output)

    return avg_posteriors


########################################################################################################################


def cross_situational_learning(input_file, n_iter=200, seed=6, print_output=False):

    """
    :param input_file:          a .txt file containing two columns. The first column is assumed to contain cues, while
                                the second columns is assumed to contain outcomes.
    :param n_iter:              an integer stating how many iterations need to be performed to evaluate each learner
    :param seed:                an integer to seed the random permutation of learning trials in the training set, for
                                reproducibility
    :param print_output:        a boolean specifying whether average associations should be printed after having been
                                computed
    :return learning_outcomes:  a dictionary of dictionaries. The first-level keys indicate the four different learning
                                models ('ndl', 'HebbianLearner', 'probabilisticLearner', and 'HTM'). Each of these is a
                                dictionary of dictionaries mapping each outcome to every cue: the values are tuples,
                                each consisting of two numbers: the average associations over the specified number of
                                iterations between a cue and an outcome, and its standard deviation.
    """

    training_trials = read_input_trials(input_file)

    learning_outcomes = defaultdict(dict)

    learning_outcomes['ndl'] = discriminative_learner(training_trials, n_iter, seed=seed, print_output=print_output)
    learning_outcomes['HebbianLearner'] = hebbian_learner(training_trials, print_output=print_output)
    learning_outcomes['probabilisticLearner'] = probabilistic_learner(training_trials, n_iter, seed=seed,
                                                                      print_output=print_output)
    learning_outcomes['HTM'] = hypothesis_testing_model(training_trials, n_iter, print_output=print_output)

    return learning_outcomes


########################################################################################################################

def main():

    parser = argparse.ArgumentParser(description='Run cross-situational learning experiments.')

    parser.add_argument('-i', '--input_file', required=True, dest='input_file',
                        help='Specify the path to the input file.')
    parser.add_argument('-n', '--num_iter', default=200, dest='n_iter',
                        help='Specify the number of iterations.')
    parser.add_argument('-s', '--seed', default=6, dest='seed',
                        help='Specify the seed to replicate random outcomes.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='out',
                        help='Specify the seed to replicate random outcomes.')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise ValueError("The path you provided does not exist. Please, provide the path to an existing file.")
    else:
        learning_outcomes = cross_situational_learning(args.input_file, n_iter=args.n_iter,
                                                       seed=args.seed, print_output=args.out)

    for algorithm in learning_outcomes:
        print algorithm
        print "\tOutcome-Cue: Mean (stdev)"
        for w1 in learning_outcomes[algorithm]:
            for w2 in learning_outcomes[algorithm][w1]:
                mean, stdev = learning_outcomes[algorithm][w1][w2]
                print "\t"+"-".join([w1,w2])+": "+str(mean)+' ('+str(stdev)+')'

########################################################################################################################


if __name__ == '__main__':

    main()
