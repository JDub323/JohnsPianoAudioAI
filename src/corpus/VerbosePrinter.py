# class which only prints stuff if verbose is true
# to avoid lots of boilerplate in my code

class VerbosePrinter:
    def __init__(self, configs):
        self.will_print = configs.verbose

    def print(self, msg):
        if self.will_print:
            print(msg)
