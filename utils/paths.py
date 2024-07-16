import os


def get_ckpts(ckpts_folder, ckpt):
    ckpt_paths = []
    if ckpts_folder is not None:
        # Get all ckpts in ckpts_folder
        ckpt_paths = [
            os.path.join(ckpts_folder, path) for path in os.listdir(ckpts_folder)
        ]
    else:
        # Get the final ckpt from path
        ckpt_paths = [ckpt]
    return ckpt_paths
