"""
Easy-to-use methods for getting the singleton SessionHook.
This is abstracted into its own module to prevent circular import problems.

Sample usage (in AWS-PyTorch repo):

import smdebug.pytorch as smd
hook = smd.hook()
"""

# First Party
import smdebug.core.singleton_utils as sutils
from smdebug.core.singleton_utils import del_hook, set_hook  # noqa
from smdebug.core.utils import error_handling_agent


@error_handling_agent.catch_smdebug_errors()
def get_hook(json_config_path=None, create_if_not_exists: bool = False) -> "Hook":
    from smdebug.pytorch.hook import Hook
    from smdebug.core.config_validator import ConfigValidator

    validator = ConfigValidator(framework="pytorch")
    if validator.validate_training_Job():
        return sutils.get_hook(
            json_config_path=json_config_path,
            hook_class=Hook,
            create_if_not_exists=create_if_not_exists,
        )
    else:
        return None
