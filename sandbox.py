import argparse
from concurrent.futures import ThreadPoolExecutor

import submitit

parser = argparse.ArgumentParser()
parser.add_argument("--array", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


# printhello
def add(a: int, b: int):
    return a + b


def foo(x, y, z):
    results = x + y + z
    return results

a = [1, 2, 3, 4]
b = [10, 20, 30, 40]

if args.debug:
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(add, x, y) for x, y in zip(a, b)]
        outputs = [j.result() for j in jobs]
    print(f"Outputs: {outputs}")
else:
    log_folder = "log_test/%j"
    executor = submitit.Autoexecutor(folder=log_folder)
    slurm_config = {
        "slurm_qos": "h100_dev",
        "slurm_account": "unicorns",
        "timeout_min": 4,
    }

    if args.array:
        slurm_config["slurm_array_parallelism"] = 2

    executor.update_parameters(**slurm_config)

    if args.array:
        jobs = executor.map_array(add, a, b)
        print(f"job ids: {[j.job_id for j in jobs]}")
        outputs = [j.result() for j in jobs]
        print(f"Outputs: {outputs}")
    else:
        job = executor.submit(add, 5, 7)
        print(f"job id {job.job_id}")
        output = job.result()
        print(f"Output {output}")
