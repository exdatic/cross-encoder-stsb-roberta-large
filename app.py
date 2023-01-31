from sentence_transformers import CrossEncoder
import torch


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = CrossEncoder('cross-encoder/stsb-roberta-large', device=device)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    sentences = model_inputs.get('sentence_pairs', None)
    if sentences == None:
        return {'message': "No sentences provided"}
    
    # Run the model
    result = model.predict(sentences)

    return {"output": result.tolist()}
