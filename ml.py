import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Function to process JSON logs
def process_json_folder(json_folder, max_files=5000):
    data = []
    files_processed = 0

    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            file_path = os.path.join(json_folder, file)
            print(f"Processing file: {file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    if not isinstance(json_data, list):
                        json_data = [json_data]

                    for event in json_data:
                        if not isinstance(event, dict):
                            continue  

                        event_data = event.get("Event", {}).get("EventData", {}).get("Data", [])

                        event_dict = {}
                        for item in event_data:
                            if isinstance(item, dict) and "@Name" in item:
                                event_dict[item["@Name"]] = item.get("#text", "")

                        event_id_raw = event.get("Event", {}).get("System", {}).get("EventID", {})
                        try:
                            event_id = int(event_id_raw) if isinstance(event_id_raw, str) else int(event_id_raw.get("#text", ""))
                        except (TypeError, ValueError):
                            event_id = -1  

                        attack_event_ids = {1, 3, 5, 7, 8, 10, 13, 22, 4624, 4625, 4634, 4648, 4649, 
                                            4657, 4663, 4672, 4673, 4674, 4688, 4689, 4697, 4698, 
                                            4699, 4700, 4701, 4702, 4720, 4722, 4725, 4726, 4732, 
                                            4733, 4740, 4768, 4769, 4771, 4776, 4781, 5031, 5140, 
                                            5142, 5143, 5145, 5156, 5157, 5158, 7045, 8004}
                        label = 1 if event_id in attack_event_ids else 0  

                        data.append({
                            "EventID": event_id,
                            "Image": event_dict.get("Image", "NA"),
                            "CommandLine": event_dict.get("CommandLine", "NA"),
                            "ParentImage": event_dict.get("ParentImage", "NA"),
                            "User": event_dict.get("User", "NA"),
                            "Label": label
                        })
                except (json.JSONDecodeError, TypeError) as e:
                    continue  
            
            files_processed += 1
            if files_processed >= max_files:
                break

    return pd.DataFrame(data)

# Set JSON folder path
json_folder = "/Users/akashmishra/Documents/trainEvtx/json"  

# Process JSON files
df = process_json_folder(json_folder, max_files=10000)

# Handle missing values
df.fillna("NA", inplace=True)  
df["EventID"] = df["EventID"].astype(int)

# Remove duplicates and reset index
df = df.drop_duplicates().reset_index(drop=True)

# Print Missing Values Check
print("\n🔍 DEBUG: Missing Values in Key Columns")
print(df.isnull().sum())

# Count attack vs benign events
attack_count = df[df["Label"] == 1].shape[0]
benign_count = df[df["Label"] == 0].shape[0]

print(f"\n🔍 Total Events Processed: {len(df)}")
print(f"✅ Attack Events: {attack_count}")
print(f"🟢 Benign Events: {benign_count}")

# Combine text-based features
df["CombinedText"] = df["CommandLine"] + " " + df["Image"] + " " + df["ParentImage"] + " " + df["User"]

# Convert Text to TF-IDF Features
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df["CombinedText"])

# Encode EventID
label_encoder = LabelEncoder()
df["Encoded_EventID"] = label_encoder.fit_transform(df["EventID"])

# Ensure X_text and X_event have the same shape
X_event = df["Encoded_EventID"].values.reshape(-1, 1)
X_combined = np.hstack((X_text.toarray(), X_event))

# Labels (Attack or Benign)
y = df["Label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, stratify=y, random_state=42)

# **Apply SMOTE to balance dataset**
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train XGBoost Model with regularization
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, 
                          min_child_weight=5, subsample=0.7, colsample_bytree=0.7, 
                          reg_lambda=3.0, reg_alpha=2.0, random_state=42)

xgb_model.fit(X_train_resampled, y_train_resampled)

# **Cross-Validation to Check Overfitting**
cv_scores = cross_val_score(xgb_model, X_combined, y, cv=5, scoring='accuracy')
print(f"\n✅ Cross-Validation Accuracy: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

# Predict & Evaluate
y_pred = xgb_model.predict(X_test)

# Print Model Performance
print("\n🚀 Model Performance:\n")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.6f}")

# Check Train vs Test Accuracy
train_pred = xgb_model.predict(X_train_resampled)
train_accuracy = accuracy_score(y_train_resampled, train_pred)

print(f"🔍 Training Accuracy: {train_accuracy:.6f}")
print(f"🔍 Testing Accuracy: {accuracy:.6f}")

if train_accuracy > 0.98 and accuracy < 0.90:
    print("⚠️ Overfitting detected! Consider tuning model further.")

# Save processed data to CSV
# df.to_csv("processed_events.csv", index=False)
# print("✅ Processed events saved as CSV.")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
