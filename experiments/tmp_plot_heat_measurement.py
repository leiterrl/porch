import matplotlib.pyplot as plt
import numpy as np

errors = [0.01, 0.02, 0.05, 0.1, 0.2]
pinn_err = 4.7835367666948514e-08
de_err = [
    2.6398280169814825e-05,
    0.00015040456491988152,
    0.00033858013921417296,
    0.0036834983620792627,
    0.013910114765167236,
]
de_es_err = [
    7.207469820968981e-08,
    4.174174783599938e-09,
    8.75967089086771e-06,
    3.76453130002119e-07,
    8.716880984138697e-06,
]

fig = plt.figure(figsize=(12, 9))

plt.axhline(y=pinn_err, color="r", linestyle="-", label="PINN")
plt.plot(errors, de_err, "-*", label="de-PINN")
plt.plot(errors, de_es_err, "-*", label="de-es-PINN")
plt.legend()
plt.yscale("log")

plt.savefig("plots/error_sens_test.png")
