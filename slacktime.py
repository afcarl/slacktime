"""The core slacktime model class and some functions for evaluating and optimizing the model."""

import numpy as np
from optunity_modified import minimize_structured, par
from input_texts import InputTexts


class Model(object):

    def __init__(self, switch_cost=0):
        self.textmatrix = np.array([])
        self.ticks = 0
        self.subprocesses = dict()
        self.subprocess_names = []
        self.locked_by = None
        self.switch_ticks = 0
        self.switch_cost = switch_cost
        self.length = 0

    def append_subprocess(self, name, duration, active_word, buffer_len, input_processes):
        self.subprocesses[name] = {
            'duration': duration,
            'active_word': active_word,
            'buffer_len': buffer_len,
            'input_processes': input_processes
        }
        self.subprocess_names.append(name)

    def load_text(self, textarray):
        self.textmatrix = textarray
        self.textmatrix['onset'] -= self.textmatrix['onset'][0]
        self.length = self.textmatrix['onset'][-1]

    def update(self):

        for subprocess_name in self.subprocess_names:
            if subprocess_name == 'input':
                # update active word
                where = np.where(self.textmatrix['onset'] <= self.ticks)
                if len(where[0]) != 0:
                    active_word = int(np.max(where))
                    if len(self.subprocesses[subprocess_name]['active_word']) == 0:
                        self.subprocesses[subprocess_name]['active_word'].append(active_word)
                    elif active_word > self.subprocesses[subprocess_name]['active_word'][-1]:
                        self.subprocesses[subprocess_name]['active_word'].append(active_word)
            else:
                # move to the latest word where accumulated evidence exceeds the duration threshold
                if type(self.subprocesses[subprocess_name]['input_processes']) is list:
                    input_process = self.subprocesses[subprocess_name]['input_processes'][0][0]
                else:
                    input_process = self.subprocesses[subprocess_name]['input_processes'][0]
                where = np.where(self.textmatrix[input_process] >= self.subprocesses[input_process]['duration'])
                if len(where[0]) != 0:
                    active_word = int(np.max(where))
                    if len(self.subprocesses[subprocess_name]['active_word']) == 0:
                        self.subprocesses[subprocess_name]['active_word'].append(active_word)
                    elif active_word > self.subprocesses[subprocess_name]['active_word'][-1]:
                        self.subprocesses[subprocess_name]['active_word'].append(active_word)

            # if evidence has passed threshold for active word in head of queue, pop it
            if len(self.subprocesses[subprocess_name]['active_word']) > 0:
                active_word = self.subprocesses[subprocess_name]['active_word'][0]
                duration = self.subprocesses[subprocess_name]['duration']
                if self.textmatrix[subprocess_name][active_word] > duration:
                    self.subprocesses[subprocess_name]['active_word'].pop(0)

            # if queue is too long, pop head of queue
            while (len(self.subprocesses[subprocess_name]['active_word']) >
                   self.subprocesses[subprocess_name]['buffer_len']):
                self.subprocesses[subprocess_name]['active_word'].pop(0)

            locked = False
            # lock/unlock
            if subprocess_name == 'lemma_sel':
                if self.locked_by == 'lemma_sel':
                    if len(self.subprocesses['lemma_sel']['active_word']) == 0:
                        # count down switch cost and then unlock
                        if self.switch_ticks > 0:
                            self.switch_ticks -= 1
                        else:
                            self.locked_by = None
                elif self.locked_by == 'concept_prep':
                    locked = True
                else:
                    if len(self.subprocesses['lemma_sel']['active_word']) > 0:
                        self.locked_by = 'lemma_sel'
                        self.switch_ticks = self.switch_cost
            elif subprocess_name == 'concept_prep':
                if self.locked_by == 'concept_prep':
                    if len(self.subprocesses['concept_prep']['active_word']) == 0:
                        if self.switch_ticks > 0:
                            self.switch_ticks -= 1
                        else:
                            self.locked_by = None
                elif self.locked_by == 'lemma_sel':
                    locked = True
                else:
                    if len(self.subprocesses['concept_prep']['active_word']) > 0:
                        self.locked_by = 'concept_prep'
                        self.switch_ticks = self.switch_cost

            # let evidence accumulate
            if (len(self.subprocesses[subprocess_name]['active_word']) > 0) and (locked is False):
                active_word = self.subprocesses[subprocess_name]['active_word'][0]
                flow_rate = self.textmatrix['flow_rate'][active_word]
                if type(self.subprocesses[subprocess_name]['input_processes']) is list:
                    for input_process in self.subprocesses[subprocess_name]['input_processes']:
                        input_process_name = input_process[0]
                        input_process_flow = input_process[1]
                        if self.textmatrix[input_process_name][active_word] >= self.subprocesses[input_process_name]['duration']:
                            self.textmatrix[subprocess_name][active_word] += (input_process_flow * flow_rate)
                else:
                    input_process_flow = self.subprocesses[subprocess_name]['input_processes'][1]
                    self.textmatrix[subprocess_name][active_word] += (input_process_flow * flow_rate)

        # advance clock
        self.ticks += 1

    def advance_clock(self, ticks):
        for tick in range(ticks):
            self.update()


def build_model(task, duration_factor=1.0, buffer_len=1, shortcut_flow=0.0, switch_cost=0):
    model_spec = [
        {'name': 'input',
         'duration': 1,
         'active_word': [0],
         'buffer_len': 1,
         'input_processes': (None, 1)},
        {'name': 'phon_feat_ex',
         'duration': 75,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('input', 1)},
        {'name': 'segment',
         'duration': 125,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('phon_feat_ex', 1)},
        {'name': 'phon_code_sel',
         'duration': 90,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('segment', 1)},
        {'name': 'lemma_sel',
         'duration': 150,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('phon_code_sel', 1)},
        {'name': 'concept_prep',
         'duration': 175,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('lemma_sel', 1)},
        {'name': 'lemma_ret',
         'duration': 75,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('concept_prep', 1)},
        {'name': 'phon_code_ret',
         'duration': 80,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('lemma_ret', 1)},
        {'name': 'syllab',
         'duration': 125,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('phon_code_ret', 1)},
        {'name': 'phonetic_enc',
         'duration': 145,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('syllab', 1)},
        {'name': 'output',
         'duration': 1,
         'active_word': [],
         'buffer_len': 1,
         'input_processes': ('phonetic_enc', 1)},
    ]

    if task == 'shadowing':
        for i in range(len(model_spec)):
            if model_spec[i]['name'] == 'syllab':
                model_spec[i]['input_processes'] = [('segment', shortcut_flow), ('phon_code_sel', 1), ('phon_code_ret', 1)]
    elif task == 'interpreting':
        for i in range(len(model_spec)):
            if model_spec[i]['name'] == 'concept_prep':
                model_spec[i]['buffer_len'] = int(buffer_len)

    for i in range(len(model_spec)):
        model_spec[i]['duration'] = duration_factor * model_spec[i]['duration']

    model = Model(switch_cost=switch_cost)
    for process in model_spec:
        model.append_subprocess(**process)

    return model


def default_inputs(folder='input_data', function_word_rate=2.0):
    input_object = InputTexts(folder=folder, function_word_rate=function_word_rate)
    return input_object.input_dict


def build_text(words):
    num = len(words)
    text = np.zeros(num, dtype=[('onset', 'i4'),
                                ('duration', 'i4'),
                                ('flow_rate', 'f4'),
                                ('input', 'f4'),
                                ('phon_feat_ex', 'f4'),
                                ('segment', 'f4'),
                                ('phon_code_sel', 'f4'),
                                ('lemma_sel', 'f4'),
                                ('concept_prep', 'f4'),
                                ('lemma_ret', 'f4'),
                                ('phon_code_ret', 'f4'),
                                ('syllab', 'f4'),
                                ('phonetic_enc', 'f4'),
                                ('output', 'f4')])
    text['onset'] = words[:, 0]
    text['duration'] = words[:, 1]
    text['flow_rate'] = words[:, 2]
    return text


def evaluate(task='shadowing', input_dict=None, function_word_rate=2.0, **params):
    if input_dict is None:
        input_dict = default_inputs(function_word_rate=function_word_rate)

    scores_dict = {
        '100wpm': dict(),
        '125wpm': dict(),
        '150wpm': dict(),
        '175wpm': dict(),
        '200wpm': dict(),
    }
    mean_scores = {
        '100wpm': [],
        '125wpm': [],
        '150wpm': [],
        '175wpm': [],
        '200wpm': [],
    }

    for wpm, titles in input_dict.items():
        for title, words in titles.items():
            model = build_model(task, **params)
            text = build_text(words)
            model.load_text(text)
            model.advance_clock(model.length + 3000)
            unique, counts = np.unique(model.textmatrix['output'], return_counts=True)
            score = (len(words) - float(counts[0])) / len(words)
            scores_dict[wpm][title] = score

    for wpm, titles in scores_dict.items():
        for title, score in titles.items():
            mean_scores[wpm].append(score)
    for wpm, scores in mean_scores.items():
        mean_scores[wpm] = np.mean(scores)

    return mean_scores, scores_dict


def loss_function(**params):
    errors = []
    pp_scores = {
        'shadowing': {
            '100wpm': .95,
            '125wpm': .95,
            '150wpm': .92,
            '175wpm': .86,
            '200wpm': .81
        },
        'interpreting': {
            '100wpm': .87,
            '125wpm': .81,
            '150wpm': .74,
            '175wpm': .69,
            '200wpm': .59
        }
    }
    for task in ['shadowing', 'interpreting']:
        model_scores, _ = evaluate(task, **params)
        errors += [model_scores[wpm] - pp_score for wpm, pp_score in pp_scores[task].items()]

    return np.sqrt(np.mean(np.square(errors)))


def minimize_loss(params,
                  iterations=1,
                  cores=1,
                  method='random search'):
    pmap = par.create_pmap(cores)
    minimal_params, details, solver = minimize_structured(loss_function,
                                                          params,
                                                          method,
                                                          num_evals=iterations,
                                                          pmap=pmap)
    return minimal_params, details, solver
