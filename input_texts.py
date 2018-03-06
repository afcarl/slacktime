"""This class returns an object containing stimulus texts ready for use in computational models (e.g., slacktime)."""

import numpy as np
from textgrid import *
from os import listdir
from function_words import function_words


class InputTexts(object):

    def __init__(self, folder, function_word_rate):
        folder = folder + '/'
        self.input_dict = {
            '100wpm': dict(),
            '125wpm': dict(),
            '150wpm': dict(),
            '175wpm': dict(),
            '200wpm': dict()
        }
        self.input_list = list()
        for filename in listdir(folder):
            if filename[-9:] == '.TextGrid':
                textgridfile = TextGrid.load(folder + filename)
                wpm = filename[-15:-9]
                text = filename[:-16]
                tiers = textgridfile._find_tiers()
                tier = tiers[0]
                grid = tier.make_simple_transcript()

                input_stream = list()
                i = 0
                for interval in grid:
                    onset = int(float(interval[0]) * 1000)
                    offset = int(float(interval[1]) * 1000)
                    word = interval[2]
                    flow_rate = function_word_rate if word in function_words else 1.0
                    if word != '':
                        i += 1
                        input_stream.append((onset, offset - onset, flow_rate))
                input_stream = np.array(input_stream)

                self.input_dict[wpm][text] = input_stream
                self.input_list.append((wpm, text, input_stream))
        return
