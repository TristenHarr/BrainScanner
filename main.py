import numpy as np

import brainflow
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

from brainflow.board_shim import BoardShim, BrainFlowInputParams
import time
from multiprocessing import Process, Queue, Lock
import matplotlib.pyplot as plt
import pandas as pd
import os


class ThoughtStream(object):

    def __init__(self, serial_port, board_id):
        self.serial_port = serial_port
        self.board_id = board_id

        self.params = None
        self.board = None
        self.q = Queue()
        self.lock = Lock()

    def connect_board(self):
        self.params = BrainFlowInputParams()
        self.params.ip_port = 0
        self.params.ip_protocol = 0
        self.params.timeout = 0
        self.params.serial_port = self.serial_port
        self.board = BoardShim(self.board_id, self.params)

    def collect_thoughts(self, collection_duration, save_path=None):
        self.connect_board()
        self.lock.acquire()
        gather_p = Process(target=ThoughtStream._gather_thoughts, args=(self.q, self.lock, self.board, collection_duration,))
        gather_p.start()
        process_p = Process(target=ThoughtStream._process_thoughts, args=(self.q, self.lock, save_path,))
        process_p.start()
        while gather_p.is_alive() or process_p.is_alive():
            time.sleep(1)

    def stream_thoughts(self, load_path):
        self.lock.acquire()
        simulate_p = Process(target=ThoughtStream._simulate_thoughts, args=(self.q, self.lock, load_path,))
        simulate_p.start()
        process_p = Process(target=ThoughtStream._process_thoughts, args=(self.q, self.lock, None,))
        process_p.start()
        while simulate_p.is_alive() or process_p.is_alive():
            time.sleep(1)

    @staticmethod
    def _simulate_thoughts(q, lock, load_path):
        df = pd.read_csv(load_path)
        sleeps = df[["Ts"]].diff()["Ts"].values
        sleeps[0] = 0.0
        for i, row in enumerate(df.values):
            row = np.reshape(row, (32, 1))
            q.put(row)
            time.sleep(sleeps[i])
        lock.release()
        print("SIMULATE DONE")


    @staticmethod
    def _gather_thoughts(q, lock, board, collection_duration):
        board.prepare_session()
        board.start_stream()
        count = 0
        now = time.time()
        flag = True
        while flag:
            if board.get_board_data_count() > 0:
                data = board.get_board_data()
                q.put(data)
                count += len(data[0])
                elapsed = time.time() - now
                if elapsed >= collection_duration:
                    flag = False
        board.stop_stream()
        board.release_session()
        lock.release()
        print("GATHER DONE")

    @staticmethod
    def _extract_thoughts(data):
        data = data.transpose()
        return pd.DataFrame(data, columns=[
            "S",  # Sample #
            "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",  # Channels
            "C9", "C10", "C11", "C12", "C13", "C14", "C15",
            "Ac0", "Ac1", "Ac2",  # Accelerometer
            "O1", "02", "03", "O4", "05", "06", "07",  # Other
            "An0", "An1", "An2",  # Analog
            "Ts",  # Timestamp
            "O8"  # Other
            ])

    @staticmethod
    def _process_thoughts(q, lock, save_path):
        running = True
        while running:
            if not q.empty():
                df = q.get()
                df = ThoughtStream._extract_thoughts(df)
                if save_path is not None:
                    if os.path.exists(save_path):
                        df.to_csv(save_path, mode="a", header=False, index=False)
                    else:
                        df.to_csv(save_path, index=False)
                df = df[["S",
                         "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
                         "C9", "C10", "C11", "C12", "C13", "C14", "C15",
                         "Ts"
                         ]]
                # print(df)
            elif lock.acquire(False):
                running = False
                lock.release()
        print("PROCESS DONE")


if __name__ == "__main__":
    thoughts = ThoughtStream(serial_port="/dev/cu.usbserial-DM0258D3", board_id=2)
    thoughts.collect_thoughts(collection_duration=10, save_path="tmp.txt")
    # thoughts.stream_thoughts("tmp.txt")