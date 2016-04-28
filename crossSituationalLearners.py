__author__ = 'GCassani'

import numpy as np
import random as rnd
from collections import defaultdict


def read_input_trials(input_file):

    """
    :param input_file:  a .txt file containing two columns: cues in the first and outcomes in the second, separated by
                        space. Each column can contain multiple items, which should be separated by an underscore ('_')
                        A header might be present, which should start with the word 'Cues'.
    :return trials:     a list containing each line of the input file.
                        [This is memory inefficient, I know, but given the very limited size of the input here I didn't
                        bother too much about this issue.]
    """

    trials = []

    with open(input_file, 'r+') as training:
        for line in training:
            if not line.startswith('Cues'):
                trials.append(line.strip())

    return trials


########################################################################################################################


def print_associations(associations, flip=True):

    """
    :param associations:        a dictionary of dictionaries, where the top level keys are cues, the second level keys
                                are outcomes, and the bottom level values are lists of cue-outcome associations obtained
                                over the desired number of iterations
    :param flip:                a binary specifying whether cue-outcome associations should be printed and returned as
                                such or flipped i.e. outcome-cue
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

    for c in sorted(associations.keys()):
        for o in sorted(associations[c].keys()):
            if flip:
                avg_associations[o][c] = ( np.mean(associations[c][o]), np.std(associations[c][o]) )
            else:
                avg_associations[c][o] = ( np.mean(associations[c][o]), np.std(associations[c][o]) )

    for k1 in sorted(avg_associations.keys()):
        for k2 in sorted(avg_associations[k1].keys()):
            # print each cue-outcome pair with the corresponding average association and its standard deviation
            print '\t'.join([k1, k2, str(round(avg_associations[k1][k2][0], 3)),
                             str(round(avg_associations[k1][k2][1], 3))])

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


def form_random_hypothesis(outcome,hypotheses,curr_outcomes,curr_cues):

    del hypotheses[outcome]                                 # remove the hypothesis from memory,
    new_o = rnd.randint(1, len(curr_outcomes)) - 1          # select a new outcome at random
    new_outcome = curr_outcomes[new_o]
    new_sel = rnd.randint(1, len(curr_cues)) - 1            # select a new cue at random
    hypotheses[new_outcome] = curr_cues[new_sel]            # form a new cue-outcome hypotheses


########################################################################################################################


def discriminative_learner(training_file, iterations, alpha=0.2, beta=0.1, lam=1, seed=None):

    """
    :param training_file:       a .txt. file containing two columns, the first consists of cues, the second of outcome.
                                Multiple cues and outcomes can be passed, in which case they should be separated by an
                                underscore ('_').
    :param iterations:          an integer specifying the number if simulations to run: when learning depends on the order
                                of the training trials, running multiple iterations ensures that the learned
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
    :return avg_associations:   a dictionary of dictionaries containing average associations and their standard
                                deviations, computed over the desired number of iterations.

    This function implements Naive Discriminative Learning (NDL, see Baayen, Milin, Durdevic, Hendrix, Marelli (2011)
    for a detailed description of the model and its theoretical background. This learner uses the Rescorla-Wagner
    equations to update cue-outcome associations: it is a simple associative network with no hidden layer, that
    incrementally updates associations between cues and outcome.
    """

    print
    print "Incremental Naive Discriminative Learner (Baayen et al, 2011): "
    print

    associations = defaultdict(dict)

    # read in the training trials from the input file
    training_trials = read_input_trials(training_file)

    rnd.seed(seed)

    for i in xrange(iterations):

        # the order of presentation impacts learning, so training trials are randomly shuffled before each simulation
        rnd.shuffle(training_trials)

        ith_associations = defaultdict(dict)
        outcomes = set()

        for trial in training_trials:

            # get each single cue and each single outcome from the current training trial
            trial_cues = trial.split(' ')[0].split('_')
            trial_outcomes = trial.split(' ')[1].split('_')

            # keep track of all outcomes that have been encountered so far
            outcomes = outcomes.union(set(trial_outcomes))

            # compute total activation for cues in the current trial: if a cue-outcome pair is new, its activation is 0
            v_total = 0
            for cue in set(trial_cues):
                for outcome in outcomes:
                    try:
                        v_total += ith_associations[cue][outcome]
                    except KeyError:
                        ith_associations[cue][outcome] = 0

            # compute association change for each cue-outcome relation for every cue in the current trial and update
            # the association score accordingly. If a cue is not present in the current learning trial, the change in
            # for each cue-outcome association involving the absent cue is 0 and we don't compute it. On the contrary,
            # change in association from present cues to all outcomes, present and absent, are computed
            for cue in set(trial_cues):
                for outcome in outcomes:
                    if outcome in set(trial_outcomes):
                        # delta V if both cue and outcome are present in the current learning trial
                        delta_v = alpha * beta * (lam - v_total)
                    else:
                        # delta V if the cue is present in the current learning trial but the outcome isn't
                        delta_v = alpha * beta * (0 - v_total)
                    # update each cue-outcome associations for cues that are present in the current learning trial
                    ith_associations[cue][outcome] += delta_v

        # store association scores at the end of the current iteration
        store_current_associations(ith_associations, associations)

    avg_associations = print_associations(associations)

    return avg_associations


########################################################################################################################


def hebbian_learner(training_file, lam=1):

    """
    :param training_file:       a .txt. file containing two columns, the first consists of cues, the second of outcome.
                                Multiple cues and outcomes can be passed, in which case they should be separated by an
                                underscore ('_').
    :param lam:                 change in association. The value does not affect the relative strength of cue-outcome
                                associations but only their magnitudes, so it operates as a simple linear scaling
                                factor.
    :return avg_associations:   a dictionary of dictionaries containing average associations and their standard
                                deviations, computed over the desired number of iterations.

    This learner was assessed in Yu and Smith (2012) and is a simple associative network with no-hidden layer that
    updates cue-outcome associations according to the Hebb rule, meaning that each cue-outcome association is
    strengthened every time the cue and the outcome co-occur in a learning trial and left unchanged in all other cases.
    This learner is not sensitive to the order of presentation, thus there is no need of multiple iterations and
    random shuffling of the training trials.
    """

    print
    print "Hebbian learner (Yu & Smith, 2012): "
    print

    # read in the training trials from the input file
    training_trials = read_input_trials(training_file)

    associations = defaultdict(dict)

    for trial in training_trials:

        trial_cues = trial.split(' ')[0].split('_')
        trial_outcomes = trial.split(' ')[1].split('_')

        # increment cue-outcome association every time the two co-occurs in a training trial. If a cue and an outcome
        # co-occur for the first time, initialize their association score to the update constant lambda
        for cue in set(trial_cues):
            for outcome in set(trial_outcomes):
                try:
                    associations[cue][outcome] += lam
                except KeyError:
                    associations[cue][outcome] = lam

    avg_associations = print_associations(associations)

    return avg_associations


########################################################################################################################


def hypothesis_testing_model(training_file, iterations, alpha=0.6, alpha_1=[0.81, 0.90, 0.95, 0.99], seed=None):

    """
    :param training_file:   a .txt. file containing two columns, the first consists of cues, the second of outcome.
                            Multiple cues and outcomes can be passed, in which case they should be separated by an
                            underscore ('_').
    :param iterations:      an integer specifying the number if simulations to run: when learning depends on the order
                            of the training trials, running multiple iterations ensures that the learned
                            associations are not due to chance.
    :param alpha:           probability of recalling a hypothesis for the first time after it was formed. The default
                            value is taken from Trueswell et al (2013) where this model was shown to fit behavioral data
                            in a word learning task.
    :param alpha_1:         probabilities of recalling a hypothesis after it was already successfully recalled once.
                            The first value gives the probability of recalling a hypothesis twice, the second value
                            gives the probability of recalling a hypothesis for the third time, and so on. The first
                            value also comes from Trueswell et al (2013) while the following ones were made up: we
                            stopped at five successful recall with p=0.99 to avoid certainty of recall and assuming that
                            after a certain number of correct recalls the probability doesn't change anymore.
    :param seed:            allows to set the seed for the random shuffling of training trials, in case it is
                            important to reproduce exact results. The default is set to None to ensure maximal
                            randomness. In the CogSci 2016 paper we report results with seed=6.
                            CAVEAT: note that the final outcome also depends on the probability of recalling a
                            hypothesis at every learning trial, which is not seeded. Thus, even using the same seed
                            as we used in the paper, different results might be obtained for this learner.
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

    print
    print "Hypothesis Testing Model (HTM) (Trueswell et al, 2013): "
    print

    training_trials = read_input_trials(training_file)
    hypotheses = defaultdict(dict)
    avg_hypotheses = defaultdict(dict)

    rnd.seed(seed)

    for i in xrange(iterations):

        ith_hypotheses = {}                                                     # contains formed hypothesis
        ith_recalled = defaultdict(dict)                                        # contains recalled hypothesis

        # the order of presentation impacts learning, so training trials are randomly shuffled before each simulation
        rnd.shuffle(training_trials)

        for trial in training_trials:

            trial_cues = trial.split(' ')[0].split('_')
            trial_cues = trial_cues[0:2]                                    # only pick objectA and objectB
            trial_outcomes = trial.split(' ')[1].split('_')

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
            avg_hypotheses[o][c] = hypotheses[o][c] / float(iterations)
            print o, c, avg_hypotheses[o][c]

    return avg_hypotheses


########################################################################################################################


def probabilistic_learner(training_file, iterations, t0_prob=10**-4, beta=10**4, lam=10**-5, seed=None):

    """
    :param training_file:       a .txt. file containing two columns, the first consists of cues, the second of outcome.
                                Multiple cues and outcomes can be passed, in which case they should be separated by an
                                underscore ('_').
    :param iterations:          an integer specifying the number if simulations to run: when learning depends on the order
                                of the training trials, running multiple iterations ensures that the learned
                                associations are not due to chance.
    :param t0_prob:             posterior probability of outcome given cue at time 0, before cue and outcome co-occur
                                in the learning trials, best computed as 1/beta, as indicated in fazly et al (2010).
    :param beta:                upper bound on the expected number of outcomes.
    :param lam:                 smoothing factors. It's important that lam is smaller than 1/beta, as indicated by
                                Fazly et al (2010).
    :param seed:                allows to set the seed for the random shuffling of training trials, in case it is
                                important to reproduce exact results. The default is set to None to ensure maximal
                                randomness. In the CogSci 2016 paper we report results with seed=6.
    :return avg_associations:   a dictionary of dictionaries containing average associations and their standard
                                deviations, computed over the desired number of iterations.

    This learner is described at length in Fazly, Alishai, and Stevenson (2010) and is inspired by IBM machine
    translation algorithm. It is a simple associative network, with no hidden layers, that computes a full probability
    distribution over all outcomes for each cues. Learning happens by allocating a higher probability mass to the
    outcome that is best supported by the data. Each cue has its own probability distribution that is shifted according
    to the evidence accumulated over subsequent trials: highly skewed probability distributions reflect higher
    confidence in which outcome matches a cue, while more flat distributions reflect higher uncertainty
"""

    print
    print "Probabilistic Learner (Fazly et al, 2010): "
    print

    # read in the training trials from the input file
    training_trials = read_input_trials(training_file)

    outcomes_given_cues = defaultdict(dict)

    rnd.seed(seed)

    for i in xrange(iterations):

        outcomes = set()
        ith_outcomes_given_cues = defaultdict(dict)
        ith_associations = defaultdict(dict)

        # the order of presentation impacts learning, so training trials are randomly shuffled before each simulation
        rnd.shuffle(training_trials)

        for trial in training_trials:

            # get each single cue and each single outcome from the current training trial
            trial_cues = trial.split(' ')[0].split('_')
            trial_outcomes = trial.split(' ')[1].split('_')

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
    avg_posteriors = print_associations(outcomes_given_cues, flip=False)

    return avg_posteriors


########################################################################################################################

def main():

    discriminative_learner('./training_dataSet.txt', 200)
    print
    hebbian_learner('./training_dataSet.txt')
    print
    probabilistic_learner('./training_dataSet.txt', 200)
    print
    hypothesis_testing_model('./training_dataSet.txt', 200)


########################################################################################################################


if __name__ == '__main__':
    main()