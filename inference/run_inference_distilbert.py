import torch, time, json
from transformers import AutoTokenizer, AutoModel
from pynvml import *
from datetime import datetime, timedelta

# CONFIG
ALPACA_PATH = "../alpaca_data.json"
OUTPUT_PATH = "../distilbert_logs.jsonl"
DURATION_HOURS = 8

# Init
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").eval().cuda()

with open(ALPACA_PATH, "r") as f:
    dataset = json.load(f)

def get_power():
    return nvmlDeviceGetPowerUsage(handle) / 1000.0

def run(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
    start = time.time()
    with torch.no_grad():
        _ = model(**inputs)
    return time.time() - start, len(tokenizer.tokenize(text))

start_time = datetime.now()
end_time = start_time + timedelta(hours=DURATION_HOURS)

with open(OUTPUT_PATH, "w") as out:
    while datetime.now() < end_time:
        for entry in dataset:
            text = entry["instruction"] + " " + entry["input"] if entry["input"] else entry["instruction"]
            timestamp = datetime.now().isoformat()
            duration, token_len = run(text)
            power = get_power()
            out.write(json.dumps({
                "timestamp": timestamp,
                "model": "DistilBERT",
                "input_len": token_len,
                "inference_time_sec": duration,
                "power_watts": power,
                "text_snippet": text[:100]
            }) + "\n")
            out.flush()

nvmlShutdown()
