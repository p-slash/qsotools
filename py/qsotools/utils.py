import logging

class Progress(object):
    """docstring for Progress"""
    def __init__(self, total, percThres=5):
        self.i = 0
        self.total = total
        self.percThres = percThres
        self.last_progress = 0
        self.start_time = time.time()

    def increase(self):
        self.i+=1
        curr_progress = int(100*self.i/self.total)
        print_condition = (curr_progress-self.last_progress >= self.percThres) or (self.i == 0)

        if print_condition:
            etime = (time.time()-self.start_time)/60 # min
            logging.info(f"Progress: {curr_progress}%. Elapsed time {etime:.1f} mins.")
            self.last_progress = curr_progress