FROM colo_mlops:train

# Run train
ENTRYPOINT ["python", "src/models/predict_model.py"]
