# %%
import tensorflow as tf
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

tag = "validating/validation_error"

test_path = "/import/sgs.local/scratch/leiterrl/h_param_wave_eq_pinn_equal/"


def get_event_file(path: str):
    search_path = os.path.join(path, "*tfevents*")
    for name in glob.glob(search_path):
        return name


def get_validation_errors(path: str):
    sum_it = tf.compat.v1.train.summary_iterator(get_event_file(path))
    data = []

    for e in sum_it:
        for v in e.summary.value:
            if v.tag == "validating/validation_error":
                # print(f"Step: {e.step} val_error: {v.simple_value}")
                data.append((e.step, v.simple_value))

    return np.array(data)


# %%
dir_4 = "/import/sgs.local/scratch/leiterrl/wave_eq_rom_error_sensitive_4"
dir_8 = "/import/sgs.local/scratch/leiterrl/wave_eq_rom_error_sensitive_8"
dir_12 = "/import/sgs.local/scratch/leiterrl/wave_eq_rom_error_sensitive_12"


def get_total_errors(path: str):
    total_errors = np.empty([20001, 1])

    cnt = 0
    for dir_entry in os.scandir(path):
        # if cnt > 20:
        #     break
        if dir_entry.is_dir():
            errors = get_validation_errors(dir_entry.path)
            if errors.shape[0] == 20001 and not np.isnan(errors).any():
                print(dir_entry.path)
                # print(total_errors.shape)
                # print(errors[:, 1].shape)
                total_errors = np.hstack([total_errors, errors[:, 1][:, np.newaxis]])
                cnt += 1
    return total_errors


# print(total_errors)

# print(get_validation_errors(test_path))

# %%
errors_equal = get_total_errors(dir_4)
# errors_opt = get_total_errors(dir_opt)
# errors_lra = get_total_errors(dir_lra)

mean_equal = errors_equal.mean(axis=1)
# mean_opt = errors_opt.mean(axis=1)
# mean_lra = errors_lra.mean(axis=1)

std = errors_equal.std(axis=1)

# mean = np.log(total_errors).mean(axis=1)
# std = np.log(total_errors).std(axis=1)

# mean = np.nanmean(total_errors, axis=1)
# std = np.nanstd(total_errors, axis=1)

cm = 1 / 2.54  # centimeters in inches
width_cm = 17
height_cm = 17 * 0.4
# fig, ax = plt.subplots(1, 1, figsize=[width_cm, height_cm])
plt.figure(figsize=[width_cm, height_cm])
plt.yscale("log")
plt.ylim([1e-5, 1e0])
plt.plot(mean_equal)
# plt.plot(mean_opt)
# plt.plot(mean_lra)
# plt.plot(mean_equal + std)
# plt.plot(mean_equal - std)
plt.show()

# %%
