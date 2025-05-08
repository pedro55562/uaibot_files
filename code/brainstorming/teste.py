from multiprocessing import Process, Event, Queue
import time

# Função que será executada em paralelo
def process_A(stop_event):
    i = 0
    while not stop_event.is_set():
        print(f" From A : {1*i}")
        i+=1
        time.sleep(1)
    print("processo finalizado corretamente")


if __name__ == '__main__':
    stop_event = Event()

    p = Process(target=process_A, args=(stop_event,))
    p.start()

    time.sleep(10)

    stop_event.set()
    p.join()
