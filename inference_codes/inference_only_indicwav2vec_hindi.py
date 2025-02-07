import torch
import torchaudio
from datasets import DatasetDict, load_dataset, load_metric
from transformers import AutoModelForCTC, AutoProcessor

# Load metrics
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

# Load model and processor
DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi").to(DEVICE_ID)
processor = AutoProcessor.from_pretrained("ai4bharat/indicwav2vec-hindi")

# Load dataset
raw_datasets = DatasetDict()
raw_datasets["eval"] = load_dataset(
    "json",
    data_files="/m/triton/scratch/elec/puhe/p/palp3/MUCS/MUCS_train_test_dataset_dict_v2.json",
    field="test",
)

# Function to preprocess and load audio files
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["audio_paths"])
    batch["speech"] = speech_array.squeeze().numpy()
    return batch

# Map function to dataset
test_dataset = raw_datasets["eval"]["train"].map(speech_file_to_array_fn)

# Predictions and references
predictions = []
references = []

for data in test_dataset:
    # Preprocess input audio using the processor
    inputs = processor(data["speech"], sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE_ID)
    
    # Perform inference
    with torch.no_grad():
        logits = model(inputs).logits.cpu()
    
    # Print the shape of logits for debugging
    print(f"Logits shape: {logits.shape}")

    # Directly use logits with batch_decode
    prediction = processor.batch_decode(logits.numpy())[0]
    
    # Store prediction and reference for metric computation
    predictions.extend(prediction)
    references.append(data["transcriptions"])
    
    # Print each prediction for review (optional)
    print(f"Prediction: {prediction[0]}")
    print(f"Reference: {data['transcriptions']}")
    print("###############\n")
    print(type(prediction))
    print(type(data['transcriptions']))

# Calculate and print metrics
print("WER: {:2f}".format(100 * wer_metric.compute(predictions=predictions, references=references)))
print("CER: {:2f}".format(100 * cer_metric.compute(predictions=predictions, references=references)))
