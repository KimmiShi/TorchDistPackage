import os
import argparse
from subprocess import PIPE, STDOUT, Popen
import copy
import re
import time

def exec_cmd(cmd_with_args: list, shell=False, env=None) -> str:
    results = ""
    with Popen(cmd_with_args, shell=shell, stdout=PIPE, stderr=STDOUT, env=env) as output:
        for line in iter(output.stdout.readline, b""):
            results += line.rstrip().decode() + "\n"

    return results

def launch_job(sbatch_file):
    sbatch_cmd = ["sbatch", sbatch_file]
    sbatch_env = copy.deepcopy(os.environ)
    results = exec_cmd(sbatch_cmd, env=sbatch_env)
    print("start launch job\n")
    print(results)
    print('----'*10, flush=True)
    if "Submitted batch job" not in results:
        return False, f'submit sbatch job "{sbatch_cmd}" failed, please check it.'

    new_jobid = re.search(r"\b(\d+)\b", results).group(1)  # get new jobid
    return True, new_jobid

def get_slurm_jobinfo(jobid):
    sacct_cmd = (
        f'sacct -j {jobid} --format="JobID%100, JobName%100, UID%20,'
        " User%30, State%20, QuotaType%20, ExitCode%10, Cluster%20,"
        " VirtualPartition%30, Partition%30, AllocCPUS%10, AllocGPUS%10,"
        ' AllocNodes%10, NodeList%255, NTasks%30"'
    )

    res = exec_cmd(sacct_cmd, shell=True)
    tmp = res.splitlines()

    job_info = {}
    job_info["jobid"] = tmp[2][:100]
    job_info["jobname"] = tmp[2][100:201]
    job_info["uid"] = tmp[2][201:222]
    job_info["user"] = tmp[2][222:253]
    job_info["state"] = tmp[2][253:274]
    job_info["quotatype"] = tmp[2][274:295]
    job_info["exitcode"] = tmp[2][295:306]
    job_info["cluster"] = tmp[2][306:327]
    job_info["virtual_partition"] = tmp[2][327:358]
    job_info["partition"] = tmp[2][358:389]
    job_info["alloc_cpus"] = tmp[2][389:400]
    job_info["alloc_gpus"] = tmp[2][400:411]
    job_info["alloc_nodes"] = tmp[2][411:422]
    job_info["nodelist"] = tmp[2][422:678]
    job_info["ntasks"] = tmp[-1][678:709]

    for key in job_info:
        if key == "state":
            job_info[key] = job_info[key].strip().split(" ")[0]
        value = job_info[key].replace(" ", "")
        if value == "Unknown":
            value = None
        job_info[key] = value

    return job_info


def scancel_slurm_job(job_id: str):
    """
    scancel current slurm job.
    """

    # scancel jobid
    scancel_cmd = ["scancel", f"{job_id}"]
    exec_cmd(scancel_cmd)

def determine_job_is_alive(slurm_job_id: str):
    jobinfo = get_slurm_jobinfo(slurm_job_id)
    curjob_state = jobinfo["state"]

    if curjob_state not in ["RUNNING", "PENDING", "COMPLETED"]:
        exit_code = jobinfo["exitcode"]
        print(f"Job {slurm_job_id} is {curjob_state}, exit code: {exit_code}", flush=True)
        scancel_slurm_job(slurm_job_id)
        return False
    elif curjob_state == 'COMPLETED':
        print('Job COMPLETED')

    return True

def job_is_finished(slurm_job_id: str):
    jobinfo = get_slurm_jobinfo(slurm_job_id)
    curjob_state = jobinfo["state"]

    return curjob_state == 'COMPLETED'

def monitor_job(sbatch_file, query_interval=10, ):
    job_id = None

    def job_running():
        if job_id is None:
            return False
        else:
            return determine_job_is_alive(job_id)

    # launch job
    while not job_running():
        flg, new_job_id = launch_job(sbatch_file)
        if flg:
            job_id=new_job_id

        time.sleep(query_interval)

    # monitor until finish
    while not job_is_finished(job_id):
        if not job_running():
            flg, new_job_id = launch_job(sbatch_file)
            if flg:
                job_id=new_job_id
        time.sleep(query_interval)

    print('job monitor finished')

parser = argparse.ArgumentParser()

if __name__ == "__main__":
    parser.add_argument("--cfg", type=str, help="sbatch file path")
    # parser.add_argument("--output", type=str, help="sbatch log file name")
    # parser.add_argument("--partition", type=str, help="slurm partition")
    # parser.add_argument("--ntasks", type=str, help="slurm ntasks")
    # parser.add_argument("--gpu_per_node", type=str, default=8, help="slurm gpu_per_node")
    args = parser.parse_args()
    monitor_job(args.cfg)