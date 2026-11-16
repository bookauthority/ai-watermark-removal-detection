CHAPTER_NUMBER = 10
EXPERIMENT_ID = "exp-10-multimodal-ensemble-detection-v1"

# We implement three workloads from the specification. The operational
# tiered case study is omitted because we cannot meaningfully measure
# latency/throughput without fabricating values.
WORKLOAD_IDS = [
    "wl_image_in_distribution",
    "wl_image_shifted",
    "wl_video_multimodal",
]

# Simple configuration for synthetic dataset sizes per workload
WORKLOAD_CONFIG = {
    "wl_image_in_distribution": {
        "num_samples": 400,
        "shift": 0.0,
    },
    "wl_image_shifted": {
        "num_samples": 400,
        "shift": 0.8,
    },
    "wl_video_multimodal": {
        "num_samples": 200,
        "shift": 0.3,
    },
}

# Fixed random seed for determinism
GLOBAL_RANDOM_SEED = 42
