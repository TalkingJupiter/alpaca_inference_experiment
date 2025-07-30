import torch, time, json
from transformers import AutoTokenizer, AutoModel
from pynvml import *
from datetime import datetime

# === CONFIGURATION ===
ALPACA_PATH = "../alpaca_data.json"
NUM_EXAMPLES = 3

# === LOAD ALPACA ===
print("📥 Loading Alpaca dataset...")
with open(ALPACA_PATH, "r") as f:
    data = json.load(f)

samples = [
    item["instruction"] + (" " + item["input"] if item["input"] else "")
    for item in data[:NUM_EXAMPLES]
]

# === INIT GPU POWER LOGGING ===
print("⚙️ Initializing NVML...")
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

# === LOAD MODELS ===
print("📦 Loading BERT...")
bert_model = AutoModel.from_pretrained("bert-base-uncased").eval().cuda()
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("📦 Loading DistilBERT...")
distil_model = AutoModel.from_pretrained("distilbert-base-uncased").eval().cuda()
distil_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# === HELPERS ===
def log_power():
    return nvmlDeviceGetPowerUsage(handle) / 1000.0  # watts

def run_inference(model, tokenizer, text, label):
    print(f"\n🚀 {label} Inference | Text: {text[:60]}...")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    duration = time.time() - start

    hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, 768)
    #mean_vector = hidden_state.mean(dim=1).squeeze().tolist()
    return duration, len(tokenizer.tokenize(text)), None
    print(f"⏱️ {label} took {duration:.6f}s | Embedding size: {len(mean_vector)}")
    return duration, len(tokenizer.tokenize(text)), mean_vector

# === RUN TEST & LOG ===
output_path = "test_output_alpaca.jsonl"
with open(output_path, "w") as f:
    for text in samples:
        ts = datetime.now().isoformat()
        dur, n_tokens, vec = run_inference(bert_model, bert_tokenizer, text, "BERT")
        power = log_power()
        entry = {
            "timestamp": ts,
            "model": "BERT",
            "input_len": n_tokens,
            "inference_time_sec": dur,
            "power_watts": power,
            "text_snippet": text[:100],
            #"output_vector": vec
        }
        f.write(json.dumps(entry) + "\n")
        print("✅ Logged BERT entry")

    for text in samples:
        ts = datetime.now().isoformat()
        dur, n_tokens, vec = run_inference(distil_model, distil_tokenizer, text, "DistilBERT")
        power = log_power()
        entry = {
            "timestamp": ts,
            "model": "DistilBERT",
            "input_len": n_tokens,
            "inference_time_sec": dur,
            "power_watts": power,
            "text_snippet": text[:100],
            #"output_vector": vec
        }
        f.write(json.dumps(entry) + "\n")
        print("✅ Logged DistilBERT entry")

print(f"\n🎉 Test complete. Logged to {output_path}")
nvmlShutdown()
