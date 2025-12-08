import os
from logging import getLogger
from huggingface_hub import HfApi

logger = getLogger("ditty_hf_utils")


def push_to_hub(self, repo_id, token=None, accelerator=None, private=True):
    """
    Push model to HuggingFace Hub with FSDP support.

    This function handles gathering FSDP sharded state dicts before pushing.
    Meant to be monkey-patched onto HuggingFace models.
    """
    if accelerator is None:
        self.save_pretrained(f"/tmp/ditty_push_{repo_id.replace('/', '_')}")
        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=f"/tmp/ditty_push_{repo_id.replace('/', '_')}",
            repo_id=repo_id,
            token=token,
        )
        return

    accelerator.wait_for_everyone()

    if accelerator.distributed_type == "FSDP":
        state_dict = accelerator.get_state_dict(self)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(self)
            unwrapped.save_pretrained(
                f"/tmp/ditty_push_{repo_id.replace('/', '_')}",
                state_dict=state_dict,
            )
    else:
        if accelerator.is_main_process:
            self.save_pretrained(f"/tmp/ditty_push_{repo_id.replace('/', '_')}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=f"/tmp/ditty_push_{repo_id.replace('/', '_')}",
            repo_id=repo_id,
            token=token,
        )
        logger.info(f"Pushed model to https://huggingface.co/{repo_id}")
