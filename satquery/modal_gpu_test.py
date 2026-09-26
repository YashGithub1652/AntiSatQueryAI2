import modal

app = modal.App("satquery-gpu-test")

image = modal.Image.debian_slim().pip_install("torch")

@app.function(image=image, gpu="A10G", timeout=300)
def gpu_test():
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
    }

@app.local_entrypoint()
def main():
    print(gpu_test.remote())
