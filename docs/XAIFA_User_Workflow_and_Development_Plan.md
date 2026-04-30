# XAIFA: User Workflow and Development Plan

## 1. Project Overview

XAIFA stands for Explainable AI Failure Analyzer. The goal of XAIFA is to provide a web-based dashboard where users can upload a trained image classification model and a test dataset. The system will automatically analyze model predictions, identify failed cases, generate explainable AI visualizations, compare explanation methods, cluster failure patterns, and recommend improvements.

The project is designed to move beyond simple model accuracy. Instead of only showing whether a model is correct or incorrect, XAIFA explains why failures may be happening and what actions can improve the model.

## 2. Core Project Idea

The user will upload their trained model, test data, and class labels. XAIFA will handle the remaining workflow automatically:

1. Load the uploaded model.
2. Preprocess the uploaded dataset.
3. Run predictions.
4. Calculate model performance.
5. Collect failed predictions.
6. Generate Grad-CAM, SHAP, and LIME explanations.
7. Create fusion explanations.
8. Compare solo, pair-wise, and combined explanation methods.
9. Cluster similar failure cases.
10. Generate improvement recommendations.
11. Display all results on a web dashboard.

## 3. What The User Uploads

### 3.1 Trained Model

The user uploads a trained image classification model.

Supported formats:

- TorchScript model: `.pt`
- PyTorch model weights: `.pth`

The preferred format is TorchScript `.pt` because it can be loaded more directly without needing the original model architecture code.

For `.pth` files, the user must select a supported model architecture such as:

- SimpleCNN
- ResNet18
- ResNet50
- MobileNetV2
- VGG16
- EfficientNet

### 3.2 Test Dataset

The user uploads a labelled test dataset. Two dataset formats can be supported.

Folder format:

```text
test_dataset/
  class_0/
    img1.png
    img2.png
  class_1/
    img3.png
    img4.png
```

CSV format:

```text
image_path,true_label
img1.png,cat
img2.png,dog
img3.png,car
```

### 3.3 Class Labels

The user provides class names for model output classes.

Example:

```text
0: cat
1: dog
2: car
```

### 3.4 Model Configuration

The user provides or selects:

- input image size, for example `28x28`, `32x32`, or `224x224`
- image type: grayscale or RGB
- normalization values
- batch size
- target layer for Grad-CAM, or auto-detect last convolution layer

## 4. What XAIFA Does

### 4.1 Model Loading

XAIFA loads the uploaded model based on file type.

For TorchScript `.pt`:

```python
model = torch.jit.load(model_path)
```

For `.pth`:

1. User selects supported architecture.
2. XAIFA initializes the architecture.
3. XAIFA loads the weights.
4. XAIFA validates the model input/output shape.

### 4.2 Dataset Preprocessing

XAIFA prepares the uploaded dataset by:

- reading images
- resizing images to the required input size
- converting images to grayscale or RGB
- applying normalization
- matching each image with its true label
- creating batches for prediction

### 4.3 Prediction Analysis

XAIFA runs all test images through the uploaded model and stores:

- image ID
- true label
- predicted label
- confidence score
- correct/incorrect status

The system calculates:

- total samples
- correct predictions
- failed predictions
- overall accuracy
- class-wise accuracy
- confusion matrix

### 4.4 Failure Collection

XAIFA filters all incorrect predictions and creates a failure dataset.

Each failure record contains:

- original image
- true class
- predicted class
- prediction confidence
- failure ID

### 4.5 XAI Explanation Generation

For every failed image, XAIFA generates explanations using:

- Grad-CAM
- SHAP
- LIME

The system produces:

- raw heatmap
- heatmap overlay on original image
- explanation metadata
- confidence values before and after masking important regions

### 4.6 Fusion Explanation

XAIFA combines multiple explanation heatmaps to create fusion explanations.

Supported combinations:

- Grad-CAM
- SHAP
- LIME
- Grad-CAM + SHAP
- Grad-CAM + LIME
- SHAP + LIME
- Grad-CAM + SHAP + LIME

Example fusion formula:

```python
fusion = 0.4 * gradcam + 0.3 * shap + 0.3 * lime
```

All heatmaps are normalized before fusion.

### 4.7 Explanation Quality Evaluation

Model accuracy and explanation quality are different. Grad-CAM, SHAP, and LIME do not have accuracy in the same way a classifier has accuracy.

XAIFA will calculate an explanation reliability score using faithfulness-based metrics.

Main idea:

1. Get original model confidence.
2. Identify important pixels or regions from an explanation heatmap.
3. Remove or mask those important regions.
4. Run the model again.
5. Measure how much the confidence drops.

If confidence drops strongly, the explanation likely found important regions.

Example:

```text
Original confidence: 0.91
Confidence after masking important region: 0.52
Confidence drop: 0.39
```

Higher confidence drop means better explanation reliability.

Metrics:

- confidence drop
- faithfulness score
- stability score
- overall XAI score

### 4.8 Comparative Table

XAIFA will compare solo, pair-wise, and combined explanation methods.

Example table:

| Method | Confidence Drop | Faithfulness Score | Stability Score | Overall XAI Score |
|---|---:|---:|---:|---:|
| Grad-CAM | 0.39 | 0.72 | 0.80 | 0.70 |
| SHAP | 0.43 | 0.77 | 0.74 | 0.72 |
| LIME | 0.35 | 0.68 | 0.78 | 0.67 |
| Grad-CAM + SHAP | 0.46 | 0.80 | 0.76 | 0.75 |
| Grad-CAM + LIME | 0.44 | 0.78 | 0.79 | 0.75 |
| SHAP + LIME | 0.45 | 0.79 | 0.75 | 0.74 |
| Grad-CAM + SHAP + LIME | 0.50 | 0.84 | 0.77 | 0.79 |

### 4.9 Failure Clustering

XAIFA groups failed cases into clusters based on:

- heatmap features
- model confidence
- true label and predicted label
- explanation scores
- similar visual failure behavior

Possible cluster types:

- confident but wrong
- weak or uncertain prediction
- scattered attention
- narrow attention
- similar class confusion
- unstable explanation pattern

### 4.10 Recommendation Generation

XAIFA generates improvement suggestions based on detected failure patterns.

Examples:

| Detected Pattern | Recommendation |
|---|---|
| Broad or scattered attention | Add rotation, shift, blur, and noise augmentation |
| Weak confidence | Add more training samples or improve model capacity |
| Confident but wrong | Check ambiguous labels and similar classes |
| Unstable explanations | Add dropout, normalization, or regularization |
| Same class pair confusion | Add targeted examples for confused classes |

## 5. Dashboard Workflow

The dashboard will include the following pages.

### 5.1 Upload Page

User uploads:

- model file
- test dataset
- class labels
- model configuration

### 5.2 Model Analysis Page

Displays:

- model load status
- total test samples
- overall accuracy
- class-wise accuracy
- confusion matrix
- number of failures

### 5.3 Failure Gallery Page

Displays all failed predictions with:

- original image
- true label
- predicted label
- confidence
- view details button

### 5.4 Process View Page

Displays the complete explanation process for a selected failed image:

- original image
- prediction result
- Grad-CAM heatmap
- SHAP heatmap
- LIME heatmap
- fusion heatmap
- explanation scores

### 5.5 XAI Comparison Page

Displays:

- solo method comparison
- pair-wise fusion comparison
- combined fusion comparison
- comparative table
- charts for XAI scores

### 5.6 Cluster Analysis Page

Displays:

- cluster cards
- cluster sample images
- common true/predicted classes
- cluster explanation
- cluster-wise recommendation

### 5.7 Recommendation Report Page

Displays:

- final model weakness summary
- most common failure patterns
- best performing explanation method
- recommended actions
- optional exportable report

## 6. Complete System Workflow

```text
User uploads model + dataset + labels
          |
          v
XAIFA loads and validates model
          |
          v
XAIFA preprocesses dataset
          |
          v
XAIFA runs predictions
          |
          v
XAIFA calculates model performance
          |
          v
XAIFA collects failed predictions
          |
          v
XAIFA generates Grad-CAM, SHAP, and LIME explanations
          |
          v
XAIFA creates solo, pair-wise, and combined fusion heatmaps
          |
          v
XAIFA evaluates explanation reliability scores
          |
          v
XAIFA clusters similar failure cases
          |
          v
XAIFA generates recommendations
          |
          v
User views results on dashboard
```

## 7. Proposed Technical Architecture

### 7.1 Frontend

Recommended technology:

- React
- Vite
- Tailwind CSS
- Recharts or Chart.js

Frontend responsibilities:

- upload model and dataset
- display analysis status
- show heatmaps and overlays
- show tables and charts
- show recommendations

### 7.2 Backend

Recommended technology:

- FastAPI
- Python
- PyTorch

Backend responsibilities:

- model upload and loading
- dataset processing
- prediction
- failure detection
- XAI generation
- fusion
- scoring
- clustering
- recommendation generation

### 7.3 Machine Learning and XAI Layer

Libraries:

- PyTorch
- torchvision
- scikit-learn
- SHAP
- LIME
- OpenCV
- NumPy
- Matplotlib or Pillow

### 7.4 Storage

Initial storage can be file-based:

```text
models/
uploads/
outputs/
  predictions/
  failures/
  heatmaps/
    gradcam/
    shap/
    lime/
    fusion/
  clusters/
  reports/
```

SQLite can be added later for storing run history and metadata.

## 8. Development Phases

### Phase 1: Repository Cleanup

- restructure the current script-based repo
- create backend and frontend folders
- add requirements file
- create output folders automatically
- convert scripts into reusable services

### Phase 2: Backend Foundation

- create FastAPI app
- add health check API
- add upload API
- add model loading service
- add dataset parsing service

### Phase 3: Prediction and Failure Analysis

- run predictions on uploaded dataset
- calculate accuracy
- calculate class-wise accuracy
- generate confusion matrix
- collect failed cases

### Phase 4: XAI Methods

- implement Grad-CAM
- implement SHAP
- implement LIME
- save heatmaps and overlays

### Phase 5: XAI Scoring and Fusion

- implement confidence-drop scoring
- implement faithfulness score
- implement stability score
- implement fusion heatmaps
- create solo vs pair vs combined comparison

### Phase 6: Clustering and Recommendations

- extract XAI features
- cluster failed cases
- label cluster patterns
- generate recommendations

### Phase 7: Frontend Dashboard

- create upload page
- create dashboard overview
- create failure gallery
- create process view
- create comparison table
- create cluster page
- create recommendation report page

### Phase 8: Final Polish

- add documentation
- add demo data and pretrained model
- improve UI
- test full workflow
- prepare synopsis and presentation material

## 9. Recommended Project Scope

To keep the project realistic and strong, XAIFA should initially support:

- image classification models
- PyTorch models
- TorchScript `.pt` uploads
- selected `.pth` architectures
- Grad-CAM, SHAP, and LIME
- dashboard-based visualization

The system should not claim to support every possible AI model in the first version. The correct scope is:

XAIFA supports uploaded PyTorch image classification models and provides automated explainable failure analysis.

## 10. Final One-Line Description

XAIFA is a web-based Explainable AI Failure Analyzer where users upload a trained image classification model and test dataset, and the system automatically detects failures, explains them using Grad-CAM, SHAP, and LIME, compares explanation methods, clusters failure patterns, and recommends model improvements.
