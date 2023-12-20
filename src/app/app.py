import gradio as gr
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# Load the saved full pipeline from the file
# Load the saved full pipeline from the file
full_pipeline_path = '../../model/Gradient Boosting_pipeline.pkl'

with open(full_pipeline_path, 'rb') as f_in:
    full_pipeline = pickle.load(f_in)

# Define the predict function
# Define the predict function
def predict(danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, 
      liveness, valence, tempo, duration_ms, mode):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'danceability': [danceability] if danceability else [0],  # Replace None with default value
        'energy': [energy] if energy is not None else [0],  # Replace None with default value
        'key': [key] if key else [0],  # Replace None with default value
        'loudness': [loudness] if loudness else [0],  # Replace None with default value
        'speechiness': [speechiness] if speechiness else [0],  # Replace None with default value
        'acousticness': [acousticness] if acousticness else [0],  # Replace None with default value
        'instrumentalness': [instrumentalness] if instrumentalness else [0],  # Replace None with default value
        'liveness': [liveness] if liveness else [0],  # Replace None with default value
        'valence': [valence] if valence else [0],  # Replace None with default value
        'tempo': [tempo] if tempo else [0],  # Replace None with default value
        'duration_ms': [duration_ms] if duration_ms else [0],  # Replace None with default value
        'mode': [mode] if mode else [0],  # Replace None with default value
    })

        # Make predictions using the loaded logistic regression model
        #predict probabilities
    predictions = full_pipeline.predict_proba(input_data)
    #take the index of the maximum probability
    index=np.argmax(predictions)
    higher_pred_prob=round((predictions[0][index])*100)


    #return predictions[0]
    print(f'[Info] Predicted probabilities{predictions},{full_pipeline.classes_}')
    
    return f'{full_pipeline.classes_[index]} with {higher_pred_prob}% confidence'
    
# Setting Gradio App Interface
with gr.Blocks(css=".gradio-container {background-color:grey }",theme=gr.themes.Base(primary_hue='blue'),title='Uriel') as demo:
    gr.Markdown("# Spotify Genre Prediction #\n*This App allows the user to predict genre by entering values in the given fields. Any field left blank takes the default value.*")
    
    # Receiving ALL Input Data here
    gr.Markdown("**Demographic Data**")
    with gr.Row():
        danceability = gr.Number(label="Danceability ~ describes how suitable a track is for dancing based on musical elements.")
        energy = gr.Number(label="Energy ~ measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.")
        key = gr.Number(label="Key ~ The estimated overall key of the track, If no key was detected, the value is -1")
        loudness = gr.Number(label="Loudness ~ Overall loudness of a track in decibels (dB), range between -60 and 0 db.")

    with gr.Row():
        speechiness = gr.Number(label="Speechiness ~ indicates the modality (major or minor), represented by 1 and minor is 0")
        acousticness = gr.Number(label="Acousticness ~ A confidence measure from 0.0 to 1.0 of whether the track is acoustic")
        instrumentalness = gr.Number(label="Instrumentalness ~ Predicts whether a track contains no vocals, Rap or spoken word tracks")
        liveness = gr.Number(label="Liveness ~ Detects the presence of an audience in the recording, measure from 0.0 to 1.0")

    with gr.Row():
        valence = gr.Number(label="Valence ~ A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.")
        tempo = gr.Number(label="Tempo ~ The overall estimated tempo of a track in beats per minute (BPM)")
        duration_ms = gr.Number(label="Duration_ms ~ double Duration of song in milliseconds")
        mode = gr.Number(label="Mode ~ Mode indicates the modality (major or minor) of a track, represented by 1 and minor is 0")

    # Output Prediction
    output = gr.Text(label="Outcome")
    submit_button = gr.Button("Predict")
    
    submit_button.click(fn= predict,
                        outputs= output,
                        inputs=[danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, mode]
    
    ),
    
    # Add the reset and flag buttons
    
    def clear():
        output.value = ""
        return 'Predicted values have been reset'
         
    clear_btn = gr.Button("Reset", variant="primary")
    clear_btn.click(fn=clear, inputs=None, outputs=output)
        
 
demo.launch(inbrowser = True)