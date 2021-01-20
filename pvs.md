---
titlepage: true
title: "Parallele und Verteilte Systeme: WS20/21 -- Übung 3 -- Gruppe B"
author: Lisa Piekarski, Klara Wichmann, Jakob Béla Ruckel
date: "2021-01-21"
listings-no-page-break: true
...

If not indicated otherwise, tests were performed on a
4 Core / 4 Thread machine, running 1 master and
3 workers, multiplying 1000x1000 matrices.

# Comparison of Variants

| variant | time (seconds) |
|---------+----------------|
| serial  |        7.21179 |
| A       |        2.67452 |
| C       |        2.66445 |



# Variant A (blocking)

| step                         | time (seconds) | time (%total) |
|------------------------------+----------------+---------------|
| sending data (master)        |        0.01577 |         0.509 |
| calculation  (each worker)   |        3.08063 |        99.476 |
| sending result (each worker) |        0.00047 |         0.015 |
| total                        |        3.09687 |       100.000 |


(i5-9400F, 6 Cores)

| number of workers | time (seconds) | speedup |
|-------------------+----------------+---------|
|          (serial) |        4.79994 |       1 |
|                 1 |        1.32140 |    3.63 |
|                 2 |        1.24119 |    3.87 |
|                 4 |        0.97739 |    4.91 |



# Variant C (Broadcast, Scatter, Gather)

| step                         | time (seconds) | time (%total) |
|------------------------------+----------------+---------------|
| sending data (master)        |        0.00934 |         0.351 |
| calculation  (each worker)   |        2.65221 |         99.63 |
| sending result (each worker) |        0.00050 |         0.019 |
| total                        |        2.66205 |       100.000 |

(i5-9400F, 6 Cores)

| number of workers | time (seconds) | speedup |
|-------------------+----------------+---------|
|          (serial) |        4.79994 |       1 |
|                 1 |        1.06059 |    3.63 |
|                 2 |        1.06773 |    3.87 |
|                 4 |        0.99163 |    4.84 |
