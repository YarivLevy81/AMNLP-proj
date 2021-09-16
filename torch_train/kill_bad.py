import os
import subprocess
import time
import tensorflow.compat.v1 as tf


while True:
    p = subprocess.check_output(['squeue'])
    p = str(p).split()
    my_p = []
    for i, w in enumerate(p):
        if p[i] == 'sehaik' and p[i+1]=='R':
            my_p.append((f'slurm/finetune.{p[i-3]}.err', p[i-3]))
    tf.print(my_p)

    for log, id in my_p:
        err = subprocess.check_output(['tail',f'{log}'])
        words = str(err).split()
        step = 0
        f1 = 0
        for i,w in enumerate(words):
            if words[i]=='f1' and words[i+1]=='score:':
                step = int(words[i-1][1:])
                f1 = float(str(words[i+2]).split('\\nstep')[0])
        tf.print(log, step, f1)
        if ((step >= 2000 and f1 < 20) or (step>=1000 and f1<5)):
            tf.print(f'scancel {id}')
            os.system(f'scancel {id}')
    time.sleep(180)