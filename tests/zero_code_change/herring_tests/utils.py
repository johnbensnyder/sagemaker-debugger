# Standard Library
import os
import subprocess
import sys


def launch_herring_job(script_file_path, script_args, num_workers, config_file_path, mode):
    command = (
        [
            "mpirun",
            "-np",
            str(num_workers),
            "-x",
            "HERRING_USE_SINGLENODE=1",
            "herringrun",
            "-c",
            "/opt/conda",
        ]
        + [sys.executable, script_file_path]
        + script_args
    )
    env_dict = os.environ.copy()
    env_dict["SMDEBUG_CONFIG_FILE_PATH"] = f"{config_file_path}"
    env_dict["PYTHONPATH"] = "/home/ubuntu/sagemaker-debugger/"
    subprocess.check_call(command, env=env_dict)
