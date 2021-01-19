---
titlepage: true
title: "Parallele und Verteilte Systeme: WS20/21 -- Übung 3 -- Gruppe B"
author: Lisa Piekarski, Klara Wichmann, Jakob Béla Ruckel
date: "2021-01-19"
listings-no-page-break: true
...

If not indicated otherwise, tests were performed on a
4 Core / 4 Thread machine, running 1 master and
3 workers, multiplying 1000x1000 matrices.

# Comparison of Variants

| variant | time (seconds) |
|---------+----------------|
| serial  |       10.23009 |
| dist_A  |        3.08063 |



# Variant A (blocking)

| step                         | time (seconds) | time (%total) |
|------------------------------+----------------+---------------|
| sending data (master)        |        0.01577 |         0.509 |
| calculation  (each worker)   |        3.08063 |        99.476 |
| sending result (each worker) |        0.00047 |         0.015 |
| total                        |        3.09687 |       100.000 |

| number of workers | time (seconds) | speedup |
|-------------------+----------------+---------|
|          (serial) |       10.23009 |       1 |
|                 1 |        8.06448 |    1.27 |
|                 2 |        4.21712 |    2.43 |
|                 3 |        3.19937 |    3.20 |